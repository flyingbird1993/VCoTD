# Planning Reasoner

This package is the isolated implementation path for the upgraded planning method. Legacy VCoTD scripts remain unchanged.

## Current Status

| Layer | Status | Evidence |
|---|---|---|
| Canonical data and mask contract | Implemented | Unit tests and real cached-feature sample |
| Canonical JSON L2 evaluator | Implemented | 6019/6019 current VCoTD predictions parsed |
| Experiment provenance manifest | Implemented | Config, seed, environment, file SHA256 |
| B0 constant velocity | Completed | `outputs/planning_reasoner/b0_constant_velocity` |
| B1-B4 training path | Completed | Formal Smooth-L1 input attribution table |
| Road/Interaction/Future fixed-depth model | Completed | Scratch and residual ablations on 34,149 samples |
| B2 motion-prior residual path | Completed | R1/R2/R3, zero-initialized residuals |
| Interaction/Future cache targets | Implemented | Standard 10 classes and future occupancy |
| Road map targets | Completed | 34,149 one-channel legacy drivable targets and 32 overlay checks |
| Adaptive router | Intentionally absent | Starts only after fixed-depth gains are verified |

## Reproducible Order

Run commands from `FSDrive-main` with the `qwen_vl_vadb` environment.

```bash
python -m unittest discover -s planning_reasoner/tests -v
python -m planning_reasoner.scripts.smoke_test
python -m planning_reasoner.scripts.evaluate_predictions \
  --predictions results_vcotd.json \
  --ground-truth tools/data/metrics/gt_traj.pkl \
  --masks tools/data/metrics/gt_traj_mask.pkl
python -m planning_reasoner.scripts.evaluate_b0
```

Then train B1 through B4 without changing their order (formal runs are already stored under `outputs/planning_reasoner`):

```bash
python -m planning_reasoner.scripts.train_baseline --config planning_reasoner/configs/b1_history.yaml
python -m planning_reasoner.scripts.train_baseline --config planning_reasoner/configs/b2_history_command.yaml
python -m planning_reasoner.scripts.train_baseline --config planning_reasoner/configs/b3_vision.yaml
python -m planning_reasoner.scripts.train_baseline --config planning_reasoner/configs/b4_full.yaml
```

Prepare and inspect real road supervision before fixed-depth training:

```bash
python -m planning_reasoner.scripts.prepare_road_targets --dataroot /path/to/nuscenes
python -m planning_reasoner.scripts.inspect_road_targets \
  --targets artifacts/planning_reasoner/road_targets_64x64.pkl
python -m planning_reasoner.scripts.train_structured
```

The current dataset exposes legacy semantic-prior PNG maps, so the real Road target has one drivable channel. Lane/divider channels require the separate map-expansion package. The checked overlays confirm forward-up orientation and no obvious mirror.

Residual experiments use the B2 checkpoint as a frozen motion prior:

```bash
python -m planning_reasoner.scripts.train_structured --config planning_reasoner/configs/r1_b2_road_residual.yaml
python -m planning_reasoner.scripts.train_structured --config planning_reasoner/configs/r2_b2_road_interaction_residual.yaml
python -m planning_reasoner.scripts.train_structured --config planning_reasoner/configs/r3_b2_full_residual.yaml
```

R1/R2/R3 are fixed-depth ablations, not adaptive routing. R3 currently improves the B2 Avg L2 by 0.0130 m at the best epoch, but all results are single-seed evidence.

## Output Contract

Every completed experiment directory contains:

- `manifest.json`: resolved config, seed, environment, source revision when available, and input hashes;
- `history.json`: epoch-level training and validation history;
- `best.pt`: best checkpoint selected by canonical Avg L2;
- `best_predictions.json`: token-keyed six-point predictions;
- `best_metrics.json`: official-mask and valid-only metrics.

See `ARCHITECTURE_CONTRACT.md` for tensor semantics and loss definitions.
