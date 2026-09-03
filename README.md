# roto

Read Silhouette `.sfx` roto projects, redraw them exactly, and package them as supervised
training data for a model that predicts splines from alpha.

**Status: the renderer and the dataset builder are done and tested. No model exists yet.**
The plan for the model is [docs/PLAN.md](docs/PLAN.md).

---

## What this repo does today

**1. Reads Silhouette projects.** Both container formats, all four Silhouette versions present
in the archive (`2020`, `2022.5`, `2025.5`, `5`), no licence needed, plain Python. The
archive's shapes are **B-splines**, not Béziers, and they are preserved as such — the
deliverable is a file the artist edits, so converting on ingest would manufacture parameters
they never chose.

**2. Redraws them and scores against the delivered mattes.** This is the step that says the
renderer can be trusted. On certified channels it reaches **0.9843–0.9980** overlap, and
**0.9947** at full resolution on the main test element, with every disagreeing pixel within
one pixel of the outline — no wrong pixels in the interior. Details in
[docs/FINDINGS.md](docs/FINDINGS.md).

**3. Builds training elements.** Clean alpha rendered from the artist's own splines, paired
with that element's spline program as dense arrays. The delivered EXRs are **not** used here —
they are reference only. The program round-trips back to the alpha exactly, and that is an
automated test rather than a claim.

## What it does not do

- No model, no training loop, no losses. Removed deliberately; see *History* below.
- No `.sfx` **writing**. [src/roto/sfx/write.py](src/roto/sfx/write.py) is a documented seam
  that raises `NotImplementedError`. Writing a `.sfx` needs no ML — it is serialisation from
  the IR — but no file we write has ever been opened in Silhouette, and validating one
  requires a seat we do not have.
- No plate/RGB handling. Every input is alpha.

---

## Setup

Python ≥ 3.10 (developed on 3.12), numpy and opencv-python. Nothing else — no torch.

```bash
pip install -e ".[dev]"
```

Then unpack the archive so shots land under `data/extracted/test_data/`:

```bash
mkdir -p data/extracted && unzip data/test_data.zip -d data/extracted
```

`data/` is gitignored — it is client archive material and stays out of version control.

Shot layouts differ between vendors and neither the `.sfx` path nor the matte folder names
are hardcoded; the `.sfx` is found recursively and any directory containing `.exr` files is
treated as a matte folder. Both conventions in the drop work:

```
<shot>/scene/<name>.sfx            <shot>/<name>_SFX_script_v02/<name>.sfx
<shot>/matte01/<name>.1001.exr     <shot>/<name>_matte_L110_v02/<name>.1001.exr
```

## Commands

```
roto inventory  <data_root>                     per-shot summary
roto parse      <shot_dir> [-o out.json]        .sfx -> readable JSON IR
roto verify     <shot_dir>                      render and score against delivered mattes
roto verify-all <data_root> [--report csv]      the same, every shot
roto elements   [--all]                         list training elements (uses no EXR)
roto dataset    <out_dir>                       render clean alpha + spline program
```

`verify`/`verify-all` take `--scale` (default `0.35`), `--stride` and `--viz-dir`.
**Scoring below full resolution understates every number by up to 0.02**, all of it on the
outline, because a downscaled render against a downscaled EXR carries a half-pixel convention
mismatch. Use `--scale 1.0` for a number you intend to quote.

### Typical session

```bash
roto inventory data/extracted/test_data          # what is in the archive
roto verify-all data/extracted/test_data --scale 1.0 --stride 10 --report reports/verify.csv
roto elements                                    # what is trainable
roto dataset datasets/v001/                      # build it
```

A smoke run: `roto dataset /tmp/ds --element <id> --stride 20`.

## Output layout

`roto dataset` writes one directory per element:

```
<out>/<element_id>/
  alpha/<frame>.png    16-bit clean alpha, rendered from the artist splines
  target_ir.json       the element's spline program, native B-splines
  tensors.npz          frames, matrix_frames, key_frames, points_norm,
                       opacity, lifespan, layer_matrix, n_shapes
  meta.json            provenance, crop transform, per-shape index
  preview.png          first / middle / last frame, for eyeballing
```

Control points stay in **native local normalised coordinates** — they live under the layer
transform, so pre-baking the crop translation into them would be meaningless. `meta.json`
carries `px_per_norm` for a pixel-unit loss and spells out the full composition formula.

## Elements

An element is **one top-level layer of one `.sfx`**, discovered from the file rather than a
hand-maintained table. The six shots carry **18**; **13 are v1 targets across 4 shots**
(2,753 shapes, 25,127 artist keys). Exclusions are by measurement:

| rule | threshold | fires on |
|---|---|---|
| `over_keyed` | keys per live frame > 0.75 | MAT_0130/Red Matte (1.34), sh0230/L100 (1.59), sh0230/L110 (0.95) |
| `paint_strokes` | open-stroke fraction > 0.9 | sh0230/L110 (1.00), nfl_0080/MB 2 (1.00) |
| `too_few_shapes` | < 3 shapes | FAM_0060/red (2) |

## Layout

```
src/roto/
  ir.py            canonical IR — every reader, renderer and model target converts through it
  sfx/read.py      .sfx -> IR, all four dialects
  sfx/write.py     IR -> .sfx — seam only, not implemented
  sfx/json_ir.py   IR <-> readable JSON
  render/          deterministic rasteriser (curves, raster)
  matte/exr.py     defensive EXR decode        ─┐ verification path,
  eval/            channel assignment, metrics  ┘ never in the training path
  data/            shot discovery; recovered EXR channel map (verify only)
  dataset/         manifest -> clean alpha + spline program
  program/         spline program <-> dense arrays, and back
docs/              plan, findings, standup, client handoff docs
scripts/           discover_elements.py — regenerates the EXR channel map
tests/             57 tests
```

## Tests

```bash
pytest            # 57 tests, ~100s (they render real archive frames)
```

Two are load-bearing:

- **`test_golden.py`** asserts that parsing real `.sfx` files and rendering them reproduces
  the mattes the vendor actually shipped. Thresholds sit ~1 point below measured values. If
  one needs *raising* that is an improvement; if one needs lowering, something broke.
- **`test_dataset.py`** asserts `decode(encode(program))` renders back to the element's own
  alpha, at strides 1, 7 and 20. The tolerance is `1e-4`, which is 16-bit PNG quantisation on
  the stored alpha (1/65535 per pixel) and nothing else — the tensors round-trip exactly. If
  it needs raising, the representation lost something.

## History

This repo previously supervised training on the **delivered EXR mattes**, which forced an
element to be something an EXR could certify. That required a layer→RGB-channel mapping that
was never delivered (it lived in a Nuke script), recovered by brute-force search, and it
capped usable data at 9 of 23 deliverables.

The current strategy renders clean alpha from the artist's own splines instead, so input and
target agree by construction and the model is never asked to explain a vendor's compositing
quirk alongside the artist's craft. Three blockers dissolved with it: the missing channel
mapping, the sh0230 project/matte disagreement, and a vendor colour transform wrongly applied
to alpha on two shots. The EXR path is kept — it is the evidence the renderer is right.

An exploratory training stack (a tiny memorising model, losses, an overfit gate) was removed
so the new architecture starts unencumbered. Its measured findings survive in
[docs/FINDINGS.md](docs/FINDINGS.md) and [docs/STANDUP.md](docs/STANDUP.md) — chiefly that
geometry is close to solved while keyframe **precision** is not, which is why
[docs/PLAN.md](docs/PLAN.md) treats key selection as curve simplification rather than
classification.
