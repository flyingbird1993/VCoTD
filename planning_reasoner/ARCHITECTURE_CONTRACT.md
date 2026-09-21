# Fixed-Depth Architecture Contract

## Canonical Coordinates And Time

- Ego history: `(B, 6, 2)`, padded on the left, 0.5 seconds per step.
- Ego future: `(B, 6, 2)`, current time removed, 0.5 seconds per step.
- Coordinates: ego-local metres using the cache convention; the first component is lateral and the second is longitudinal in downstream rasterization.
- Command: `(B, 3)` ordered as `[right, left, forward]`.
- All validity fields are Boolean masks. Training and JSON evaluation use the same mask-aware L2 implementation.

## Visual Tokens

The current Qwen cache stores six camera grids as `(1536, 1280)`, interpreted as `6 x 16 x 16` tokens. The data adapter pools each camera independently to `8 x 8`, producing:

- visual tokens: `(B, 384, 1280)`;
- positions: `(B, 384, 3)` containing normalized camera, x, and y coordinates.

The adapter rejects an unexpected token count. It never samples the flattened six-camera sequence as a one-dimensional signal.

## Three Structured Stages

1. Road stage
   - 64 learned road queries at the default `8 x 8` latent resolution.
   - The currently available legacy nuScenes maps provide one real semantic-prior drivable channel.
   - Lane/lane-connector/divider channels remain an extension that requires the separate map-expansion JSON package.
   - Absolute trajectory exit `P_road`.
2. Interaction stage
   - 20 learned agent queries.
   - Reads canonical context plus all Road latent tokens.
   - Predicts one of 10 nuScenes classes plus background, current center, velocity, and six future offsets.
   - Residual trajectory exit `P_interaction = P_road + delta_interaction`.
3. Future stage
   - `6 x 8 x 8` learned space-time queries.
   - Reads canonical context plus Road and Interaction latent tokens.
   - Predicts six future occupancy maps.
   - Residual final exit `P_future = P_interaction + delta_future`.

The fixed-depth model has about 5.65M parameters with the default config, excluding the cached feature encoder. The unstructured B1-B4 planner has about 2.22M parameters.

The residual configuration adds a 2.22M-parameter B2 motion prior. It is loaded from `b2_history_command/best.pt`, frozen during R1/R2/R3, and followed by zero-initialized stage residual heads. Thus epoch 0 is exactly the B2 trajectory, which is checked before training. The structured trainable parameter count remains about 5.66M after freezing the prior.

## Supervision

- Every trajectory exit uses the configured mask-aware Smooth L1 objective in the formal runs. Gaussian NLL remains an isolated pilot objective and is not mixed with the reported table.
- Road and future occupancy use BCE plus soft Dice after target resizing.
- Interaction queries use detached Hungarian matching with class and center costs, followed by class, center, velocity, and masked-future losses.
- Agent targets are selected by valid standard nuScenes class and distance to ego, then padded with class `-1`.
- Current future occupancy is a center-occupancy auxiliary target. A box-footprint occupancy variant remains a later controlled ablation.

## Experiment Boundary

This package deliberately contains no adaptive router. The admissible experiment order is:

`protocol -> B0 -> B1 -> B2 -> B3 -> B4 -> S1 -> S2 -> S3 -> R1 -> R2 -> R3 -> router`

Routing work begins only if the frozen-prior residual sequence shows multi-seed incremental accuracy or semantic gains and the full-depth model is stable. This keeps architectural benefit separate from adaptive-compute benefit.
