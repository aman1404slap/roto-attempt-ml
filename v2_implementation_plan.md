# v2 Charter — Implementation Plan (file-level)

Order of work top to bottom. One PR per numbered item; each lands with its test.

## 1. Dataset v002 (roto/dataset/build.py + configs)
- RenderConfig defaults: `fill_open_zero_width=False`, `stroke_gain=hairline_floor(1px)`,
  `open_clamp="duplicate"`; interpolation = clamped-CR law (already in ir.py).
- Transforms: projective 8-param everywhere (already in code; assert no 6-DOF path remains).
- Splits at build time, written into meta.json: `holdout_frames = every 7th`;
  `holdout_layers = ["FAM__r1_w_c", "FAM__green_2"]` (one small, one dense — confirm names).
- CLI: `roto dataset datasets/v002 --size 256`. Gate: `scripts/ledger.py datasets/v002` all green,
  including artist-self-score == 1.000000 on every layer, every frame.
- Tests: ledger rows for alignment-shift exactness and probe/local_to_crop agreement (from the
  zero-margin doc) if not already merged.

## 2. Re-baseline (scripts/train_v2.py, configs/)
- Runs: `v002_control` (v1 config, 40k, seeds 1,2) and `v002_final` (headline config, 40k,
  seeds 1,2). Backfill `final_long` seed 3 on v001.
- Report: the §3 table (mean/worst-layer/worst-frame IoU, point mean/p95, jitter, key ratio,
  key F1 constrained, holdout gap, frames<0.90 count). This table's format is frozen.

## 3. .sfx writer (roto/sfx/write.py)
- Serialize IR→XML matching read.py's schema; zlib+4-byte-header variant behind a flag.
- Ledger row: `read(write(doc))` bit-exact on all 6 shots. Then export one reconstructed layer
  (v002_final best checkpoint) for the Silhouette seat validation (external dependency: license).

## 4. S1 — key-timing head (design note first: 1 page, review, then build)
Reference design to start the note:
- Head: per-shape query token → small MLP over per-frame features. Get per-frame features by
  cross-attending each query to the encoder tokens of a K-frame window (reuse aligned window
  machinery), output logit per (shape, frame): p_key.
- Loss: BCE against artist key indicator, `pos_weight ≈ (1-r)/r` with r = key rate (~0.07),
  restricted to live frames. Labels in loss only (L1-compliant).
- Picker integration: local tolerance `tol_f = tol_base * (1 - a·p_key_f)`, a≈0.5 — keys become
  cheaper where the model expects one; DP unchanged otherwise. Tune `a`, `tol_base` on the
  holdout frames ONLY.
- Gate: key F1 ≥ 0.50 at keys ∈ [0.75,1.3]×, render within noise of v002_final.

## 5. S2 — lifespans & point counts predicted (net.py heads + masks)
- Lifespan: per (shape, frame) alive logit, BCE, hysteresis-threshold at decode.
- Point count: classification over allowed counts per shape type; decode = argmax; loss CE.
- Loss shift: curve/polyline term weight ↑ (primary), dot-L1 ↓ (0.25×); add style stat:
  |predicted point count − archive count| as soft L1 (not hard match — L2-compliant).
- Gate: render within 0.01 of S1.

## 6. S3 — dynamic queries (design note required; do not start before S2 gate)
- Note must cover: canonical ordering (sort by transform group, then centroid y,x at reference
  frame) as the first attempt; query init from encoder tokens (learned pooling), fixed max S;
  losses = composite render (union of per-shape polylines rasterized at low res) + chamfer to
  artist curves + style statistics (shape count, key sparsity, point density as distribution
  penalties); Hungarian only as fallback ablation.
- Gate: soft IoU ≥ 0.95 on the two held-out layers.

## 7. Infra (parallel to 4-6, triggered by archive subset arrival)
- Approved cloud account from Jon; sync datasets to shared storage; `scripts/launch.py` fans a
  config list to one spot A10G/L4 instance each; results folders synced back. No other changes.

## Standing rules while implementing
Two seeds per quoted number; deltas < noise floor are not results; frozen metric definitions;
any interface change bumps a version + one re-baseline; questions about ambiguity go back to the
charter owner once, batched — not resolved by local improvisation.
