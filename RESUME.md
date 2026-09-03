# Resume point — v1, paused for GPU enablement

*Paused 2026-09-03 so the RTX 5050 can be brought up. Nothing is lost; the training run had
just started and produced no checkpoint.*

## Done and committed

- **`datasets/v001/`** — all 13 elements built (35 MB, 1810 element-frames, 2753 shapes,
  256px clean alpha). Gitignored, on disk, survives the reboot. Rebuild with
  `roto dataset datasets/v001 --size 256` (~30 min, CPU-bound, no GPU benefit).
- **`src/roto/keys/`** — keyframe selection by curve simplification. Done and measured.
- **`src/roto/model/`** — network, data pipeline, crop-space geometry mapping, training loop.
  Runs; the full run is what got interrupted.
- **`v1/results/`** — the keyframe numbers below, as JSON.

## The headline result so far — keyframes, with no model at all

Phase 3 of the plan, run against ground-truth tracks on all 13 elements:

| tolerance | keys vs artist | precision | recall | F1 |
|---|---|---|---|---|
| 0.1 px | 29 629 vs 22 929 (×1.29) | 0.728 | 0.855 | **0.766** |
| 0.2 px | 23 758 vs 22 929 (**×1.04**) | 0.744 | 0.737 | 0.724 |
| 0.3 px | 19 673 (×0.86) | 0.739 | 0.613 | 0.653 |
| 0.5 px | 17 479 (×0.76) | 0.729 | 0.537 | 0.598 |
| 1.0 px | 13 106 (×0.57) | 0.717 | 0.391 | 0.484 |

Against the previous learned keyframe head — F1 0.33–0.62, precision 0.21–0.44, **2.5× too
many keys** — this is a decisive change of failure mode. Precision roughly doubles, F1 beats
the old ceiling, and at 0.2 px the key count lands within 4% of what the artist actually set.
It required no training, no GPU, and has one interpretable knob.

**Recommended operating point: 0.1 px for F1, 0.2 px for key economy.**

## What to do when the GPU is up

1. `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   — the 5050 is Blackwell (sm_120); installed torch is 2.12+cu130, which should cover it.
2. Turn the two CPU compromises back on in `src/roto/model/net.py`:
   - **Self-attention among shape queries.** Currently absent — queries cross-attend to the
     image but not to each other, so two shapes can claim the same contour with nothing to
     stop them. This is v1's biggest quality compromise.
   - **512px crops** instead of 256 (rebuild the dataset with `--size 512`).
3. Retrain: `train('datasets/v001', 'v1/model', TrainConfig(steps=12000))`.

## What was still unwritten when we stopped

- `src/roto/model/reconstruct.py` — predicted points → DP keys → rebuilt `RotoDoc` → render →
  soft-IoU against the element's own clean alpha.
- The figures: per element, one image showing original splines / clean alpha / reconstructed
  splines with its IoU, for up to 10 elements.
- `v1/tech.md` and the PM-facing overview.

## One thing already known about the model

Predicting control points in the IR's local normalised space **stalls at ~260 px** and is not
a capacity problem: local coordinates are absolute document positions spanning roughly ±0.5,
while a tracking crop of a small element covers as little as 0.059 of that. Predicting in crop
space instead ([0,1] across the alpha) fixed the conditioning immediately — 78 → 54 px in 400
steps and still falling. This is written up in `src/roto/model/geometry.py`; don't undo it.
