# Spline Rotoscoping — Handoff & POC Plan

Prepared for: Lokesh · Last updated: 29 Aug 2026
Status: Step 1 (data trust) complete on the 6-shot test drop; ready to start POC training on the labeled .sfx data.

---

## 1. What this project is

Slapshot Roto produces raster alpha mattes (per-frame pixel masks). Production, however, buys **spline programs**: a small set of Bézier/B-spline shapes per body part, animated with sparse keyframes, editable by artists. The goal is a model that converts an element's alpha sequence (plus RGB when available) into a native, artist-style spline program. Our unfair advantage is the Hotspring archive of real artist Silhouette projects (.sfx) — thousands of worked answers showing how professionals decompose subjects, place points, and choose keyframe times.

The unit of work is an **element**: one deliverable matte (one channel of one EXR sequence), typically tens of shapes — never the whole shot at once. The model never selects objects; the input alpha *is* the selection (Slapshot handles that upstream).

Two training tiers. **Tier 1:** input = clean alpha we render ourselves from the artist splines; target = the artist's spline program. **Tier 2 (later):** input = Slapshot's real predicted alpha on the same footage (imperfect); target unchanged — teaches the model to remove inference noise instead of copying it. Tier 1 requires no external data beyond the archive: the answer key generates its own exam question.

## 2. The data, concretely

Each archive shot pairs an .sfx project with delivered matte EXR sequences:

```
<shot>/
├── scene/…​.sfx            # the artist project ("the recipe")
└── matte01/…​.NNNN.exr     # delivered mattes ("the result"), frames 1001+
```

Format facts we validated on real data (do not re-derive these, they're confirmed):

- .sfx has two dialects: plain XML, or a 4-byte header followed by zlib-compressed XML. Both occur in the test drop (3 of each).
- Coordinates are center-origin, y-down, both axes normalized by image height: `px = x·H + W/2`, `py = y·H + H/2`.
- Shape paths live as `<Key frame=N>` entries, each holding the complete point list. Bézier points are (anchor, in-handle, out-handle) in absolute coords; B-splines are closed uniform cubic control polygons. In the measured sample, artists authored B-splines exclusively — the model's target stays native B-spline (convert to Bézier only inside a differentiable renderer if needed for losses).
- sfx frame 0 ↔ EXR frame 1001 (session startFrame).
- EXR mattes pack multiple layers into R/G/B channels; a channel is frequently the union of several sub-layers.
- Shapes carry Add/Subtract modes (Subtract = holdout holes), keyed opacity (0/100 hold keys used as on/off lifespans — ~45% of shapes have them), stroked open paths (hair strands, often one shape alive for a single frame), and per-frame 4×4 tracking matrices on layers (row-vector convention, translation in the 4th row) that compose down the layer hierarchy. Motion is deliberately factorized: layer transforms carry gross motion, sparse shape keys carry residual deformation. The model must preserve this factorization.
- Typical economy (nfl_0200, 191 frames): ~11 points per shape, ~14 keyframes per shape. This sparsity is the product.

## 3. What already exists: roto_toolkit.py

Single file, numpy + opencv only. Commands:

```
python roto_toolkit.py inventory  test_data                      # per-shot summary
python roto_toolkit.py parse      test_data/<shot>               # .sfx -> roto_ir.json
python roto_toolkit.py verify     test_data/<shot> --viz-dir viz # render + score vs EXRs
python roto_toolkit.py verify-all test_data --report report.csv  # whole drop
```

Internals worth knowing: `parse_sfx_to_ir` (both dialects → IR), `render_layer` (IR → alpha at any frame/scale, supersampled), `candidate_layers` + greedy union matcher (auto-discovers which layer set produces which matte channel), `iou_dice` / `soft_iou`, `INTERP_MODE` flag ("linear" | "catmullrom" — both implemented), `STROKE_WIDTH_GAIN` for stroked paths. The IR JSON stores, per shape: label, shape_type, mode, invert, keyed opacity, strokeWidth, and `path_keys = [frame, interp, closed, points]`; per layer: keyed 4×4 matrix, TRS props, children, shapes.

The renderer plays two roles going forward: **data factory** (manufactures the clean input alphas for Tier 1) and **grader** (renders model predictions for evaluation).

## 4. Verification results (why the labels are trustworthy)

Round-trip = render the parsed splines, compare against the delivered EXRs.

- nfl_0200: **all 191 frames** verified at half res — mean IoU 0.9803 (person+chair) / 0.9597 (hair detail); worst single frame 0.9721 / 0.9458. Other shots verified on sampled frames: MAT 0.988, FAM 0.88–0.99 across six channels, nfl_0080 matte01 up to 0.978, sh0260 up to 0.965 (auto-discovered 2-layer union).
- The residual gap is **pure edge anti-aliasing, not geometry**. Proof chain: (a) Catmull-Rom vs linear interpolation moves IoU only ~0.1% (both implemented and A/B tested on all 191 frames); (b) exact-keyframe frames (no interpolation active) still score 0.984, so interpolation can't be the cause; (c) at full 2880×1978 res on the *worst* frame, only 1,116 pixels disagree and every one is at distance 0.0 from the matte edge; (d) with 1px edge tolerance, IoU = 1.0000/0.9999 mean over all 191 frames; (e) pixel-level: their edge reads 0.00→0.88→1.00 across a row, ours 0.00→1.00 at the same pixel — one partial-coverage pixel, same boundary. Cause: Silhouette writes fractional coverage (grey AA edge), we currently threshold after supersampling. Fix is to keep the soft values (small change, raises headline IoU to true accuracy).
- **Data QC catch — sh0230:** the delivered matte contains animated hair strands for ~20 frames beyond the last keyframe in the archived project. The matte was rendered from a different save than the project on file. Exclude this pairing from training; at archive scale we need automated project↔matte pairing checks (the per-frame content-vs-project comparison in the toolkit is exactly that detector).
- Known renderer gaps (flagged, not blockers for fill-based shots): soft brush/feather profiles on strokes, motion blur (nfl_0080 "MB" pass scores 0.08 pending this), Silhouette's exact stroke-width units (empirical gain 0.0625 calibrated).

## 5. POC plan — start here

The recommended spine, smallest-risk-first. Phases 0–2 use only the labeled .sfx data we already have.

**Phase 0 — Training packets (~1–2 days).** Write `emit_packets.py` on top of the toolkit. For each verified element: render the clean alpha sequence (start at 512-ish crops around the element, keep the crop transform in meta), slice the IR to just that element's layers/shapes, save `{alpha/, target_ir.json, meta.json}`. Skip flagged pairings (sh0230) and MB passes. The 6 shots yield roughly a dozen elements — enough for pipeline debugging and overfit tests. Also emit derived tensors the loss needs: per-shape point arrays per key, key-time lists, lifespan intervals, per-frame composed layer matrices.

**Phase 1 — Overfit sanity check (~2–3 days).** Before any real model: a tiny network that memorizes 2–3 elements perfectly. If it can't drive train loss to ~0 and reproduce the artist program on data it has seen, the representation/losses are broken, not the model. This gate saves weeks.

**Phase 2 — Teacher-forced animation model (the actual POC).** Give the model the artist's breakdown at a reference frame (shape count, point counts, initial point positions — straight from the IR). It must predict, per element: (a) the layer transform track, (b) each shape's keyframe times, (c) point positions at those keys, (d) opacity lifespans. Sketch: a video encoder over the alpha crop sequence (a small 3D-conv or per-frame 2D encoder + temporal transformer is fine at POC scale); a decoder that cross-attends encoder features and emits per-shape tokens autoregressively — order shapes by the archive's layer order for a deterministic teacher-forcing sequence. Losses: point-position L1/L2 at artist keys; key-time loss (treat key placement as per-frame binary prediction per shape, or regress sorted key times — start with the binary formulation, it's simpler); lifespan cross-entropy; transform track regression; plus a **render consistency loss** — evaluate the predicted program at randomly sampled *non-key* frames under the same interpolation the renderer uses, render (DiffVG if gradients through rendering are wanted; our deterministic renderer suffices for evaluation-only), and compare to the clean alpha. That last loss is what stops the model from placing keys that look right only where supervised.

**Evaluation protocol (freeze this before training):** per element — matte fidelity (soft-IoU of rendered prediction vs clean alpha at all frames, reported separately for key and non-key frames), key-count ratio vs artist (sparsity), key-time agreement (within ±2 frames), point-count parity, and temporal stability (frame-to-frame point velocity smoothness). Report per element, never averaged into one opaque number at POC stage.

**Phase 3 — remove the training wheels (post-POC).** Drop the given breakdown; the model generates its own shape decomposition, judged mainly by composite render fidelity (decomposition-independent), with the artist program as guidance rather than strict matching. Then Tier 2: swap inputs to real Slapshot alphas, targets unchanged. Both need the bigger archive subset — not the 6 shots.

**Parallel infrastructure track (can proceed independently):** soft-edge rendering (keep AA values), feather/blur + motion-blur support, exhaustive all-frames verification of the remaining 5 shots, and the pairing-check QC as a standalone gate for archive ingest.

## 6. Open questions with Jon (asked, some pending)

Whether a newer save of sh0230 exists and whether any manifest links matte renders to exact project versions; whether the layer→channel mapping has an authoritative record (render-node settings) or we keep IoU-based discovery; whether Silhouette applies Catmull-Rom on path channels despite `interp="linear"` tags (empirically linear matched marginally better on nfl_0200); whether MB passes and single-frame paint-stroke hair are in scope as POC targets; when RGB plates for a subset become available (needed for the RGB branch and Tier 2, not for Tier 1).

## 7. Quickstart

```
pip install numpy opencv-python
unzip test_data.zip
python roto_toolkit.py inventory test_data
python roto_toolkit.py verify test_data/nfl_0200_bg01_v001_compplate_roto_v001 --scale 0.5 --stride 1 --viz-dir viz
```

Data-handling note: the shots are client production material — keep everything on approved machines/infra only; no personal cloud IDEs or third-party uploads without sign-off.
