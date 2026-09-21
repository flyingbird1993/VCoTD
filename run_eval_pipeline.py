"""
Wait for training to finish, then run inference + UniAD + STP3 evaluation.
Usage:
    python run_eval_pipeline.py --training_log <path> --ckpt <path>
"""
import time
import subprocess
import sys
import os
import argparse

def tail(path, n=5):
    try:
        with open(path) as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception:
        return ''

def run(cmd, cwd=None):
    print(f'\n[PIPELINE] Running: {" ".join(cmd)}\n', flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    return result.returncode

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training_log', type=str, required=True)
    p.add_argument('--ckpt',         type=str, default='saves/vcotd_e2e/checkpoint_best.pt')
    p.add_argument('--output_json',  type=str, default='results_vcotd_e2e_final.json')
    p.add_argument('--gt_folder',    type=str, default='tools/data/metrics')
    p.add_argument('--method',       type=str, default='VCoTD-E2E')
    p.add_argument('--nuscenes_root',type=str,
                   default='/media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/nuscenes')
    args = p.parse_args()

    python = '/home/flyingbird/anaconda3/envs/qwen_vl_vadb/bin/python'
    cwd    = '/home/flyingbird/Work/AutoDriver/FSDrive-main'

    # 1. Wait for training to complete
    print('[PIPELINE] Waiting for training to complete...', flush=True)
    while True:
        last = tail(args.training_log, 3)
        if 'Training complete' in last:
            print('[PIPELINE] Training complete!', flush=True)
            print(last, flush=True)
            break
        if last:
            print(f'[PIPELINE] Still training... last log:\n{last}', flush=True)
        time.sleep(60)

    # 2. Inference
    print('\n[PIPELINE] Step 1: Inference', flush=True)
    rc = run([python, 'infer_vcotd_e2e.py',
              '--ckpt', args.ckpt,
              '--nuscenes_root', args.nuscenes_root,
              '--split', 'val',
              '--output_path', args.output_json,
              '--batch_size', '64'], cwd=cwd)
    if rc != 0:
        print(f'[PIPELINE] Inference failed (rc={rc})', flush=True)
        sys.exit(1)

    # 3. UniAD evaluation
    print('\n[PIPELINE] Step 2: UniAD evaluation', flush=True)
    run([python, 'tools/evaluation/evaluation.py',
         '--metric', 'uniad',
         '--result_file', args.output_json,
         '--method', args.method,
         '--gt_folder', args.gt_folder], cwd=cwd)

    # 4. STP3 evaluation
    print('\n[PIPELINE] Step 3: STP3 evaluation', flush=True)
    run([python, 'tools/evaluation/evaluation.py',
         '--metric', 'stp3',
         '--result_file', args.output_json,
         '--method', args.method,
         '--gt_folder', args.gt_folder], cwd=cwd)

    print('\n[PIPELINE] All done!', flush=True)

if __name__ == '__main__':
    main()
