"""
Efficiency benchmark: FSDrive-LoRA vs VCoTD-distill vs VCoTD-E2E

Measures per-sample inference latency, FPS, parameter count, and speedup.

Three inference chains:
  A. FSDrive-LoRA  : Qwen2-VL-2B + LoRA → autoregressive generation (~66 tokens)
  B. VCoTD-distill : Qwen2-VL visual encoder (feature hook) + 4.3M student (1 forward)
  C. VCoTD-E2E     : EfficientViT-M4 + 4.15M student (1 forward)

Usage:
    cd FSDrive-main
    python benchmark_efficiency.py
"""

import os, time, json, pickle, glob, argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

N_WARMUP = 10
N_BENCH  = 50

NUSCENES_ROOT = '/media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/nuscenes'
CAMERA_TYPES  = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# ── helpers ───────────────────────────────────────────────────────────────────

def gpu_timer(fn, n_warmup=N_WARMUP, n_bench=N_BENCH):
    """Returns (mean_ms, std_ms) after warmup."""
    device = torch.device('cuda')
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_sample_images(nuscenes_root, data, tokens):
    """Load raw PIL images for the first valid sample. Returns (token, list[PIL])."""
    idx_map = {}
    for path in glob.glob(os.path.join(nuscenes_root, '*', 'samples', '*', '*.jpg')):
        idx_map[os.path.basename(path)] = path

    for token in tokens:
        if token not in data:
            continue
        d = data[token]
        pils = []
        ok = True
        for cam in CAMERA_TYPES:
            fn = os.path.basename(d['cams'][cam]['data_path'])
            fp = idx_map.get(fn)
            if fp is None or not os.path.exists(fp):
                ok = False; break
            pils.append(Image.open(fp).convert('RGB'))
        if ok:
            return token, pils   # list of 6 PIL Images
    raise RuntimeError("No valid sample found with all 6 cameras.")


def load_hist(data, token, T_obs=6):
    d = data[token]
    his = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
    if his.shape[0] >= T_obs:
        return his[-T_obs:]
    pad = torch.zeros(T_obs - his.shape[0], 2)
    return torch.cat([pad, his], dim=0)   # (T_obs, 2)


# ── A: FSDrive-LoRA ───────────────────────────────────────────────────────────

def bench_fsdrive_lora(data, tokens, device):
    print("\n[A] Loading FSDrive-LoRA (Qwen2-VL-2B + LoRA)...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    base_path  = 'model/FSDrive_pretrain'
    lora_path  = 'model/lora_traj_5ep'

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation='eager',
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)

    total_params = count_params(model)
    print(f"    Params: {total_params:.1f}M")

    token, pil_imgs = load_sample_images(NUSCENES_ROOT, data, tokens)
    pil_imgs = [img.resize((224, 224)) for img in pil_imgs]
    hist = load_hist(data, token).to(device)

    # Build processor inputs once
    messages = [{
        'role': 'user',
        'content': [
            *[{'type': 'image', 'image': img} for img in pil_imgs],
            {'type': 'text', 'text': 'Predict the future trajectory of the ego vehicle.'},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_imgs, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def run():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=80, do_sample=False)

    print(f"    Benchmarking ({N_WARMUP} warmup + {N_BENCH} runs)...")
    mean_ms, std_ms = gpu_timer(run)
    fps = 1000.0 / mean_ms

    del model
    torch.cuda.empty_cache()

    return {'name': 'FSDrive-LoRA', 'params_M': total_params,
            'latency_ms': mean_ms, 'std_ms': std_ms, 'fps': fps}


# ── B: VCoTD-distill ─────────────────────────────────────────────────────────

def bench_vcotd_distill(data, tokens, device):
    print("\n[B] Loading VCoTD-distill (Qwen2-VL visual encoder + 4.3M student)...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from create_data.extract_teacher_features import (
        FeatureHook, load_and_resize_image, LAYER_DEEP,
    )
    from model.vcotd_student import build_student_from_checkpoint

    base_path = 'model/FSDrive_pretrain'

    teacher = Qwen2VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation='eager',
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    visual_enc = teacher.visual
    L_d = min(LAYER_DEEP, len(visual_enc.blocks) - 1)
    hook = FeatureHook()
    hook.register(visual_enc.blocks[L_d])

    enc_params = count_params(visual_enc)

    # Load student
    ckpt = torch.load('saves/vcotd_distill_single/checkpoint_best.pt',
                      map_location=device, weights_only=False)
    student = build_student_from_checkpoint(ckpt).to(device)
    student.eval()

    stu_params   = count_params(student)
    total_params = enc_params + stu_params
    print(f"    Encoder: {enc_params:.1f}M  Student: {stu_params:.2f}M  Total: {total_params:.1f}M")

    token, pil_imgs = load_sample_images(NUSCENES_ROOT, data, tokens)
    pil_imgs = [img.resize((224, 224)) for img in pil_imgs]
    hist = load_hist(data, token, student.T_obs).unsqueeze(0).to(device)

    messages = [{
        'role': 'user',
        'content': [
            *[{'type': 'image', 'image': img} for img in pil_imgs],
            {'type': 'text', 'text': 'Describe the scene.'},
        ],
    }]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=pil_imgs, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def run():
        hook.value = None
        with torch.no_grad():
            try:
                teacher(**inputs)
            except Exception:
                pass
        feat = hook.value.unsqueeze(0).float().to(device)
        with torch.no_grad():
            student.predict(feat, hist)

    print(f"    Benchmarking ({N_WARMUP} warmup + {N_BENCH} runs)...")
    mean_ms, std_ms = gpu_timer(run)
    fps = 1000.0 / mean_ms

    hook.remove()
    del teacher, student
    torch.cuda.empty_cache()

    return {'name': 'VCoTD-distill', 'params_M': total_params,
            'latency_ms': mean_ms, 'std_ms': std_ms, 'fps': fps}


# ── C: VCoTD-E2E ─────────────────────────────────────────────────────────────

def bench_vcotd_e2e(data, tokens, device):
    print("\n[C] Loading VCoTD-E2E (EfficientViT + student)...")
    from model.light_encoder import LightEncoder
    from model.vcotd_student_e2e import VCoTDStudentE2E

    ckpt = torch.load('saves/vcotd_e2e/checkpoint_best.pt',
                      map_location=device, weights_only=False)
    cfg  = ckpt.get('args', {})

    encoder = LightEncoder(
        model_name=cfg.get('encoder_model', 'efficientvit_m4'), pretrained=False,
    ).to(device)
    encoder.load_state_dict(ckpt['encoder_state'])
    encoder.eval()

    student = VCoTDStudentE2E(
        d_enc=encoder.d_enc,
        d_teacher=cfg.get('d_teacher', 1280),
        d=cfg.get('d', 256), K=cfg.get('K', 4),
        h=cfg.get('h', 4), d_ff=cfg.get('d_ff', 1024),
        T_obs=cfg.get('T_obs', 6), T_pred=cfg.get('T_pred', 6),
        vis_tokens=cfg.get('vis_tokens', 16),
        K_det=cfg.get('K_det', 20),
    ).to(device)
    student.load_state_dict(ckpt['student_state'])
    student.eval()

    enc_params = count_params(encoder)
    stu_params = count_params(student)
    total_params = enc_params + stu_params
    print(f"    Encoder: {enc_params:.2f}M  Student: {stu_params:.2f}M  Total: {total_params:.2f}M")

    to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    token, pil_imgs_e2e = load_sample_images(NUSCENES_ROOT, data, tokens)
    imgs = torch.stack([to_tensor(img) for img in pil_imgs_e2e]).unsqueeze(0).to(device)
    hist = load_hist(data, token).unsqueeze(0).to(device)

    def run():
        with torch.no_grad():
            feats = encoder(imgs)
            student.predict(feats, hist)

    print(f"    Benchmarking ({N_WARMUP} warmup + {N_BENCH} runs)...")
    mean_ms, std_ms = gpu_timer(run)
    fps = 1000.0 / mean_ms

    del encoder, student
    torch.cuda.empty_cache()

    return {'name': 'VCoTD-E2E', 'params_M': total_params,
            'latency_ms': mean_ms, 'std_ms': std_ms, 'fps': fps}


# ── print table ───────────────────────────────────────────────────────────────

def print_table(results):
    baseline_ms = results[0]['latency_ms']
    print("\n" + "="*80)
    print(f"{'Model':<20} {'Params(M)':>10} {'Latency(ms)':>14} {'FPS':>8} {'Speedup':>10}")
    print("-"*80)
    for r in results:
        speedup = baseline_ms / r['latency_ms']
        print(f"{r['name']:<20} {r['params_M']:>10.1f} "
              f"{r['latency_ms']:>8.1f}±{r['std_ms']:.1f} "
              f"{r['fps']:>8.1f} "
              f"{speedup:>9.1f}x")
    print("="*80)
    print(f"Baseline: {results[0]['name']} ({results[0]['latency_ms']:.1f} ms)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--skip_lora', action='store_true', help='Skip FSDrive-LoRA (slow to load)')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  CUDA: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")

    data       = pickle.load(open('create_data/cached_nuscenes_info.pkl', 'rb'))
    split_info = json.load(open('create_data/full_split.json'))
    tokens     = split_info['val']

    results = []

    if not args.skip_lora:
        results.append(bench_fsdrive_lora(data, tokens, device))

    results.append(bench_vcotd_distill(data, tokens, device))
    results.append(bench_vcotd_e2e(data, tokens, device))

    print_table(results)

    # Save JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved → benchmark_results.json")


if __name__ == '__main__':
    main()
