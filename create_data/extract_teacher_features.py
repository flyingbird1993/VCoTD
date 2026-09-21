"""
Offline teacher feature extraction for VCoTD distillation training.

Loads the FSDrive pre-trained Qwen2-VL model, registers forward hooks on the
visual encoder at layers L_s=8, L_m=16, L_d=24 (0-indexed: 7, 15, 23), and
caches the three-level feature representations to disk.

Each sample produces a .pt file containing:
    {
        'shallow': Tensor (N, 1280),   # layer  8 features (ViT hidden_size=1280)
        'middle' : Tensor (N, 1280),   # layer 16 features
        'deep'   : Tensor (N, 1280),   # layer 24 features
    }

where N = total visual tokens from all 6 surround cameras.

Usage:
    cd FSDrive-main
    python create_data/extract_teacher_features.py \
        --model_path saves/qwen2_vl-2b/pretrain \
        --split train \
        --output_dir ./teacher_features \
        --image_size 224 \
        --batch_size 1
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Configuration ────────────────────────────────────────────────────────────

CAMERA_TYPES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# Visual encoder layer indices for feature extraction (1-indexed in paper, 0-indexed here)
LAYER_SHALLOW = 7   # paper's layer  8
LAYER_MIDDLE  = 15  # paper's layer 16
LAYER_DEEP    = 23  # paper's layer 24


def parse_args():
    parser = argparse.ArgumentParser(description="Extract teacher features from Qwen2-VL")
    parser.add_argument('--model_path', type=str, default='model/FSDrive_pretrain',
                        help='Path to pre-trained Qwen2-VL checkpoint (FSDrive pretrain)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to extract')
    parser.add_argument('--output_dir', type=str, default='./teacher_features',
                        help='Directory to save extracted feature files')
    parser.add_argument('--nuscenes_root', type=str,
                        default='./LLaMA-Factory/data/nuscenes',
                        help='nuScenes dataset root')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resize each camera image to this square size before encoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--dtype', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Model dtype for memory efficiency')
    parser.add_argument('--resume', action='store_true',
                        help='Skip tokens that already have saved feature files')
    return parser.parse_args()


# ── Hook utilities ────────────────────────────────────────────────────────────

class FeatureHook:
    """Captures the output hidden states of a Transformer block."""

    def __init__(self):
        self.value: torch.Tensor | None = None
        self._handle = None

    def register(self, module: torch.nn.Module):
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # Qwen2VLVisionBlock returns tensor of shape (total_tokens, d_vlm)
        if isinstance(output, torch.Tensor):
            self.value = output.detach().cpu().float()
        elif isinstance(output, (tuple, list)):
            self.value = output[0].detach().cpu().float()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ── Image preprocessing ───────────────────────────────────────────────────────

def load_and_resize_image(path: str, size: int) -> Image.Image:
    """Load a camera image and resize to (size, size)."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return img


# ── Main extraction logic ─────────────────────────────────────────────────────

def extract_features_for_token(
    token: str,
    data_dict: dict,
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    hooks: dict,
    image_size: int,
    nuscenes_root: str,
    device: torch.device,
):
    """
    Run Qwen2-VL visual encoder on 6 surround cameras for one nuScenes sample.

    Returns:
        dict with 'shallow', 'middle', 'deep' tensors, each (N_total, d_vlm)
        or None on failure
    """
    # Load all 6 camera images
    images = []
    for cam in CAMERA_TYPES:
        img_path = data_dict['cams'][cam]['data_path']
        # Handle path prefix replacement
        img_path = img_path.replace('/localdata_ssd/nuScenes', nuscenes_root, 1)
        if not os.path.exists(img_path):
            # Try alternate path format
            img_path = os.path.join(nuscenes_root, img_path.lstrip('/'))
        if not os.path.exists(img_path):
            return None
        images.append(load_and_resize_image(img_path, image_size))

    # Build conversation-style input for the Qwen2-VL processor.
    messages = [{
        'role': 'user',
        'content': [
            *[{'type': 'image', 'image': img} for img in images],
            {'type': 'text', 'text': 'Describe the scene.'},
        ],
    }]

    # Preprocess: tokenize + prepare pixel_values
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(
            text=[text],
            images=images,
            return_tensors='pt',
            padding=True,
        )
    except Exception as e:
        return None

    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # Reset hook values
    for h in hooks.values():
        h.value = None

    # Forward pass through the model (only need encoder, but full forward is simpler)
    with torch.no_grad():
        try:
            model(**inputs)
        except Exception:
            # Some inputs may fail (e.g., shape mismatch); skip gracefully
            return None

    # Collect hooked features
    features = {}
    for name, hook in hooks.items():
        if hook.value is None:
            return None
        # hook.value: (total_tokens_all_images, d_vlm)
        features[name] = hook.value.clone()

    return features


def main():
    args = parse_args()

    # ── Load dataset metadata ─────────────────────────────────────────────
    print("Loading cached nuScenes info...")
    data = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split = json.load(open('./create_data/full_split.json', 'r'))
    tokens = split[args.split]
    print(f"  Split '{args.split}': {len(tokens)} samples")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading Qwen2-VL from {args.model_path} ...")
    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ── Validate visual encoder depth ────────────────────────────────────
    visual_encoder = model.visual  # Qwen2VisionTransformerPretrainedModel
    num_blocks = len(visual_encoder.blocks)
    print(f"Visual encoder depth: {num_blocks} blocks")

    # Adjust layer indices if encoder depth differs from default 32
    # Default paper setting: L_s=8, L_m=16, L_d=24 (1-indexed)
    L_s = min(LAYER_SHALLOW, num_blocks - 1)
    L_m = min(LAYER_MIDDLE,  num_blocks - 1)
    L_d = min(LAYER_DEEP,    num_blocks - 1)
    print(f"Feature extraction layers (0-indexed): shallow={L_s}, middle={L_m}, deep={L_d}")

    # ── Register forward hooks ────────────────────────────────────────────
    hooks = {
        'shallow': FeatureHook(),
        'middle':  FeatureHook(),
        'deep':    FeatureHook(),
    }
    hooks['shallow'].register(visual_encoder.blocks[L_s])
    hooks['middle'].register(visual_encoder.blocks[L_m])
    hooks['deep'].register(visual_encoder.blocks[L_d])

    # ── Setup output directory ────────────────────────────────────────────
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving features to: {output_dir}")

    # ── Extraction loop ───────────────────────────────────────────────────
    success, skipped, failed = 0, 0, 0

    for token in tqdm(tokens, desc=f"Extracting {args.split}"):
        out_path = output_dir / f"{token}.pt"

        # Resume: skip already-extracted samples
        if args.resume and out_path.exists():
            skipped += 1
            continue

        if token not in data:
            failed += 1
            continue

        features = extract_features_for_token(
            token=token,
            data_dict=data[token],
            model=model,
            processor=processor,
            hooks=hooks,
            image_size=args.image_size,
            nuscenes_root=args.nuscenes_root,
            device=device,
        )

        if features is None:
            failed += 1
            continue

        # Save as CPU float16 tensors (~half the storage of float32)
        torch.save({k: v.half() for k, v in features.items()}, out_path)
        success += 1

    # ── Cleanup ───────────────────────────────────────────────────────────
    for h in hooks.values():
        h.remove()

    print(f"\nDone: success={success}, skipped(resume)={skipped}, failed={failed}")
    print(f"Feature files saved to: {output_dir}")

    # ── Report approximate storage usage ─────────────────────────────────
    if success > 0:
        sample_file = next(output_dir.glob('*.pt'))
        sample_feats = torch.load(sample_file, map_location='cpu', weights_only=True)
        N = sample_feats['deep'].shape[0]
        d_vlm = sample_feats['deep'].shape[1]
        bytes_per_sample = sum(value.numel() * value.element_size()
                               for value in sample_feats.values())
        print(f"Token count per sample: N={N}")
        print(f"Approx. storage per sample: {bytes_per_sample / 1e6:.1f} MB")
        print(f"Approx. total storage: {bytes_per_sample * success / 1e9:.1f} GB")


if __name__ == '__main__':
    main()
