# roto

Reconstruct editable roto splines from a matte.

Given the **matte** for one roto layer — a filled silhouette, one frame — produce the **shapes**
that drew it: B-spline control points on sparse **keyframes**, the form an artist actually
edits.

```
Silhouette .sfx  ──render──▶  matte  ──▶  model  ──▶  shapes + keyframes
```

## Vocabulary

Silhouette's own terms are used throughout, in code and docs.

| term | meaning |
|---|---|
| **shot** | one Silhouette project (`.sfx`) and the roto in it |
| **layer** | a top-level group of shapes; each renders to one matte |
| **shape** | one B-spline. A layer holds anywhere from 4 to 1,036 of them |
| **matte** | the alpha for one layer at one frame — what the model is given |
| **keyframe** | a frame on which the artist positioned a shape's control points |
| **control point** | one point defining a B-spline |

A layer's shapes overlap heavily — measured up to **2.5×** the layer's own silhouette area —
because artists stack overlapping shapes to cover a form. A matte is their union, not a
partition, which is why a matte cannot simply be split back into its shapes.

## Status

v1 is complete: **13 layers across 4 shots reconstruct at 0.9218 soft IoU.** See
[v1/OVERVIEW.md](v1/OVERVIEW.md) for what that means and [v1/tech.md](v1/tech.md) for how.

v1 is an **overfit** — trained and measured on the same 13 layers. It answers "can we regenerate
roto we have been shown", not "can we roto an unseen shot". That is v2.

## Setup

Python ≥ 3.10, numpy, opencv-python, torch (CPU is enough), matplotlib.

```bash
pip install -e ".[dev]"
mkdir -p data/extracted && unzip data/test_data.zip -d data/extracted
```

`data/` is gitignored — archive material stays out of version control.

## Commands

```
roto inventory [data_root]              what is in the archive
roto parse     <shot_dir> [-o out]      .sfx -> readable JSON
roto layers    [--all]                  layers available for training
roto dataset   <out_dir>                render mattes + spline programs
```

Full run, start to finish:

```bash
roto layers                                     # 13 of 18 layers selected
roto dataset datasets/v001 --size 256           # ~30 min, CPU-bound
python -c "from roto.model import train, TrainConfig; \
           train('datasets/v001','v1/model', TrainConfig(steps=12000))"
python scripts/eval_keys.py datasets/v001       # keyframe search on its own
python scripts/report_v1.py                     # scores + figures
```

## Which layers are used

A layer is selected unless a measured rule excludes it:

| rule | threshold | fires on |
|---|---|---|
| `over_keyed` | > 0.75 keys per live frame | 3 layers, at 0.95–1.59 |
| `paint_strokes` | > 0.9 of shapes are open strokes | 2 layers, both at 1.00 |
| `too_few_shapes` | < 3 shapes | 1 layer, at 2 |

13 of 18 top-level layers pass: 1,810 mattes, 2,753 shapes, 22,929 artist keyframes.

The `over_keyed` threshold reads a real gap rather than cutting an arbitrary tail — 14 layers
sit at 0.03–0.59 and four at 0.95–1.59, with nothing between.

## Layout

```
src/roto/
  ir.py            the intermediate representation everything converts through
  shots.py         locating each shot's .sfx
  metrics.py       iou, soft_iou
  sfx/             .sfx reader, JSON form, writer seam
  render/          deterministic rasteriser (curves, raster)
  dataset/         layer manifest -> matte + spline program per layer
  program/         spline program <-> dense arrays
  keys/            keyframe selection by curve simplification
  model/           network, training, reconstruction, figures
scripts/           eval_keys.py, report_v1.py
tests/             61 tests
v1/                deliverables (gitignored): docs, checkpoint, results, figures
```

## Tests

```bash
pytest            # 61 tests, ~75s — they render real archive frames
```

Two carry the most weight:

- **`test_dataset.py`** asserts `decode(encode(program))` renders back to the layer's own
  matte, at strides 1, 7 and 20. Tolerance is `1e-4`, which is 16-bit PNG quantisation on the
  stored matte and nothing else. If it needs raising, the representation lost something.
- **`test_keys.py`** measures keyframe selection against tracks with planted, known keyframes,
  so it is scored on ground truth rather than on its own output.

## Not in scope yet

- **Writing a real `.sfx`.** [src/roto/sfx/write.py](src/roto/sfx/write.py) is a documented
  seam that raises `NotImplementedError`. This needs no machine learning — it is serialisation
  from the IR — but nothing we write has ever been opened in Silhouette, and checking that
  needs a licence seat.
- **Generalising to unseen shots.**
- **Harder inputs.** The matte is rendered from the answer, so it is perfectly clean. A plate,
  or a mask from another tool, is not.
