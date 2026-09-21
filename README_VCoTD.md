# VCoTD: Lightweight Visual CoT Planning via Progressive Feature Distillation

This directory contains the paper-aligned implementation of VCoTD on top of
the official FutureSightDrive (FSDrive) codebase.

> Reproducibility status: the architecture and losses have been reconciled
> with the current VCoTD manuscript, but the manuscript's reported 0.57 m
> average L2 result has not been independently reproduced by the local
> artifacts. See [Paper and artifact differences](#paper-and-artifact-differences).

## Method scope

VCoTD replaces FSDrive's autoregressive future-image generation with a compact
four-layer Transformer that predicts all six future waypoints in one forward
pass:

```text
six surround-view images
  -> frozen Qwen2-VL visual encoder
  -> deep feature sequence
  -> global average pooling + Linear(d_vlm, 256)
  -> concatenate six ego-history tokens
  -> four Transformer encoder layers
  -> two-layer trajectory decoder
  -> six future (x, y) waypoints
```

During training, teacher layers 8, 16, and 24 supervise student layers 1, 2,
and 4 with three complementary objectives:

- `L_global`: pooled shallow-feature alignment;
- `L_spatial`: normalized token-energy profile alignment;
- `L_detail`: interpolated deep-feature alignment.

The full per-sample objective is:

```text
L_distill = 1.0 * L_global + 1.0 * L_spatial + 0.5 * L_detail
alpha_i   = 0.3 + (1.0 - 0.3) * gamma_i
L_total   = mean_i(L_pred_i + alpha_i * L_distill_i)
```

`L_pred` is the mean Euclidean displacement over valid future steps. `gamma`
is a learned training-time distillation-strength gate. It does not select
cameras, tokens, Transformer layers, or inference exits. The inference-only
`predict()` path skips the gate and the distillation feature taps.

## Code map

| Path | Purpose |
|---|---|
| `model/vcotd_student.py` | Paper model, hierarchical losses, adaptive weight, checkpoint compatibility |
| `create_data/extract_teacher_features.py` | Offline teacher feature extraction at layers 8/16/24 |
| `train_vcotd.py` | Single-GPU and DDP training, validation, checkpointing |
| `run_train.sh` | Portable tmux launcher for the paper GAP configuration |
| `run_vcotd_ablations.sh` | Five incremental paper ablations |
| `infer_vcotd.py` | Raw-image inference with the frozen Qwen visual encoder |
| `infer_vcotd_cached.py` | Cached-feature diagnostic inference |
| `run_eval_only.py` | Mask-aware checkpoint validation against cached features |
| `tests_vcotd/` | Unit tests for the paper-critical behavior |

`planning_reasoner/` is a separate structured-planning research branch. It is
not used for the VCoTD paper results or commands in this README.

## Environment

Use the FSDrive environment described in the root `README.md`. The local code
has been exercised with Python 3.10 and PyTorch in the `qwen_vl_vadb` Conda
environment.

```bash
conda activate qwen_vl_vadb
cd FSDrive-main
```

The following files are expected:

```text
create_data/cached_nuscenes_info.pkl
create_data/full_split.json
LLaMA-Factory/data/nuscenes/       # directory or symlink
model/FSDrive_pretrain/            # frozen FSDrive/Qwen2-VL checkpoint
```

The local `full_split.json` contains 23,388 training and 6,019 validation
tokens.

## Extract teacher features

Feature extraction is offline and must be run once for each split:

```bash
python create_data/extract_teacher_features.py \
  --model_path model/FSDrive_pretrain \
  --nuscenes_root LLaMA-Factory/data/nuscenes \
  --split train \
  --output_dir teacher_features \
  --dtype bf16 \
  --resume

python create_data/extract_teacher_features.py \
  --model_path model/FSDrive_pretrain \
  --nuscenes_root LLaMA-Factory/data/nuscenes \
  --split val \
  --output_dir teacher_features \
  --dtype bf16 \
  --resume
```

Each file contains `shallow`, `middle`, and `deep` tensors. The feature width is
read from the cache at training time; do not hardcode the manuscript value.
The current local cache uses `(1536 tokens, 1280 channels)` per level and
occupies about 324 GiB in total.

## Train

Single GPU, paper-aligned GAP model:

```bash
python train_vcotd.py \
  --mode distill \
  --teacher_feat_dir teacher_features \
  --output_dir saves/vcotd_paper_gap \
  --visual_aggregation gap \
  --vis_tokens 1 \
  --epochs 12 \
  --batch_size 64 \
  --lr 1e-4
```

For two GPUs and a global batch size of 64, use 32 samples per process:

```bash
torchrun --standalone --nproc_per_node=2 train_vcotd.py \
  --mode distill \
  --teacher_feat_dir teacher_features \
  --output_dir saves/vcotd_paper_gap \
  --visual_aggregation gap \
  --vis_tokens 1 \
  --epochs 12 \
  --batch_size 32 \
  --lr 1e-4
```

The tmux launcher derives the project path from its own location:

```bash
bash run_train.sh
bash run_train.sh --attach
bash run_train.sh --resume
```

Its environment overrides are `VCOTD_ENV`, `VCOTD_SESSION`,
`VCOTD_BATCH_SIZE`, `VCOTD_FEATURE_DIR`, `VCOTD_OUTPUT_DIR`, and `VCOTD_LOG`.

New checkpoints store `model_config` and training arguments. Legacy one-token
and 16-token checkpoints are loaded by inferring their shape-critical settings
and retain the old uniform-sampling semantics. The `uniform` mode is retained
only for legacy compatibility; it is not the architecture described in the
manuscript. A legacy one-token checkpoint is therefore not silently relabeled
as a paper GAP checkpoint.

## Ablations

The launcher maps directly to the incremental rows in the manuscript:

| Run | Distillation terms | Alpha |
|---|---|---|
| `no_distillation` | none | none |
| `global` | global | fixed 1.0 |
| `global_spatial` | global + spatial | fixed 1.0 |
| `hierarchical_fixed` | global + spatial + detail | fixed 1.0 |
| `full_adaptive` | global + spatial + detail | learned per sample |

Run all five experiments:

```bash
VCOTD_PYTHON="$CONDA_PREFIX/bin/python" \
  bash run_vcotd_ablations.sh all --epochs 12 --batch_size 64
```

Run one experiment or a short smoke run:

```bash
bash run_vcotd_ablations.sh full_adaptive \
  --epochs 1 --batch_size 2 \
  --max_train_samples 64 --max_val_samples 64 --num_workers 0
```

Never use `--max_val_samples` for a reported result.

`--mode distill_kd` is an optional response-KD extension and is not part of the
paper method. It requires `--teacher_wp_dir` and logs teacher-waypoint coverage.

## Inference

### Raw images: complete inference pipeline

This is the deployment path described in the manuscript. It still requires one
frozen Qwen2-VL visual-encoder pass, but no autoregressive future-image decoder:

```bash
python infer_vcotd.py \
  --student_ckpt saves/vcotd_paper_gap/checkpoint_best.pt \
  --teacher_model_path model/FSDrive_pretrain \
  --nuscenes_root LLaMA-Factory/data/nuscenes \
  --mode distill \
  --split val \
  --output_path results_vcotd_paper_gap.json
```

### Cached features: student diagnostic

```bash
python infer_vcotd_cached.py \
  --ckpt saves/vcotd_paper_gap/checkpoint_best.pt \
  --feat_dir teacher_features/val \
  --split val \
  --output_path results_vcotd_cached.json
```

Cached inference measures the student after feature extraction. It must not be
reported as raw-image end-to-end latency or as teacher-free deployment.

## Evaluation

Fast mask-aware L2 validation from a checkpoint:

```bash
python run_eval_only.py \
  --checkpoint saves/vcotd_paper_gap/checkpoint_best.pt \
  --teacher_feat_dir teacher_features
```

The script reports both denominators:

- `official-mask`: masked errors divided by all validation samples, matching
  the local FSDrive/UniAD port;
- `valid-only`: masked errors divided by the number of valid samples at each
  horizon.

For the FSDrive L2 and collision evaluator, first generate a result JSON and
then run:

```bash
python tools/evaluation/evaluation.py \
  --metric stp3 \
  --result_file results_vcotd_paper_gap.json \
  --method VCoTD
```

Record the metric implementation, checkpoint hash, split size, precision,
hardware, warm-up count, and repeated-run latency statistics with every result.

## Tests

```bash
python -m unittest discover -s tests_vcotd -v
python -m unittest discover -s planning_reasoner/tests -v
```

The VCoTD tests cover masked GAP, full student-sequence distillation, padding
masks, per-sample and fixed alpha, masked Euclidean trajectory loss, the
inference-only path, data collation, and legacy checkpoint inference.

## Paper and artifact differences

The manuscript and the checked local artifacts are not fully consistent. These
differences must be resolved before claiming reproduction:

| Item | Manuscript | Verified local artifact/code |
|---|---:|---:|
| Teacher feature width | 1536 | 1280 |
| Visual encoder depth | 28 blocks | 32 blocks in extraction logs |
| Train split | 28,130 | 23,388 tokens |
| Student-side trainable parameters | about 16M | 4,299,149 for the GAP model |
| Visual input tokens | one GAP token | one GAP token in paper path; legacy 16-token mode also exists |
| Reported average L2 | 0.57 m | not reproduced; legacy local log records 1.9299 m |

The exact local model count includes the student, trajectory decoder,
complexity estimator, and both distillation projections. The complete raw-image
pipeline is larger because it includes the frozen visual encoder. The existing
benchmark artifact reports 669.6M parameters and 95.9 ms for that chain, but it
is a single local benchmark record rather than a controlled reproduction.

## Known limitations

- The adaptive gate has no complexity label, calibration objective, or explicit
  regularizer. Because `alpha * L_distill` is positive, its direct gradient can
  favor the lower alpha bound. Treat `gamma` as an uncalibrated learned
  distillation-strength gate, not as a validated scene-complexity score.
- Adaptive weighting is training-only. VCoTD has no dynamic-depth routing,
  early exit, camera selection, or token selection at inference.
- The cached-feature path is not teacher-free raw-image inference.
- The depth-to-visual-CoT-stage correspondence is a design motivation, not a
  causal identification of individual hidden states.
- Full planning and collision results must be rerun after this reconciliation;
  legacy result files were produced by earlier loss and evaluation code.

## GitHub release hygiene

Do not commit local data, feature caches, model weights, optimizer checkpoints,
or training logs. On this machine, `teacher_features/`, `model/`, and `saves/`
contain approximately 324 GiB, 298 GiB, and 2.7 GiB respectively. The included
`.gitignore` preserves source files under `model/` while excluding the large
artifacts.

Before release, publish separately:

1. a checksum and download instructions for each released checkpoint;
2. the exact environment lock file and GPU/CUDA versions;
3. canonical full-validation L2/collision outputs for every ablation;
4. a reconciled manuscript configuration matching the released feature width,
   split, parameter count, and encoder depth.
