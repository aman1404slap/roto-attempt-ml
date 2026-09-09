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
`v1/` is frozen; its published numbers are reproducible at commit `2d1d53e`.

v1.1 answers the v1 code review, and then a review of v1.1, on the same 13 layers:
**0.9742 soft IoU**, 0.92 px point error, at 0.80× the artist's keyframe count — or **0.9712
with a layer-motion head that also works** (0.9567 rendered, from 0.7938), which is the version
to build v2 on. See [v1.1/OVERVIEW.md](v1.1/OVERVIEW.md),
[v1.1/review-response.md](v1.1/review-response.md) and
[v1.1/v1.1-review-response.md](v1.1/v1.1-review-response.md).

v1.2 answers the handover written after v1.1 ([v1.1-to-v1.2-plan.md](v1.1-to-v1.2-plan.md)),
which asks for something other than a better number: split every pixel of disagreement into
**pipeline error**, which must be provably zero, and **model error**, which is minimised and
reported *worst case*. It is a measurement round and it proposes no new headline model. What it
established: the run-to-run noise floor is **±0.0069 soft IoU at 12k**, larger than most
differences v1.1's ladder reports; the exactness ledger has 15 rows and found two that were
silently wrong; and dropping the teacher-forced motion track entirely costs **0.0049** end to
end, not the 0.043 the head's own isolated number implies. See [v1.2/OVERVIEW.md](v1.2/OVERVIEW.md)
and [v1.2/plan-response.md](v1.2/plan-response.md).

The aggregate is the least informative number in that table. **The two layers that were broken
went 0.7316 → 0.9533 and 0.7739 → 0.9614**, exactly as the v1 review predicted they would when
it named shape-query collision as the cause; the mean moved only +0.051 because eight of the
thirteen layers were already above 0.96 and had nothing left to give. Separately, 87% of the
aggregate gain is training 3.3× longer on v1's unmodified configuration — at 12k steps nothing
in the ladder has converged, which `v1_control_long` proves rather than assumes.

v1.1 also corrects how the system is measured, referees four render conventions against
Silhouette's own EXRs, and fixes three bugs found by scoring the case that must come out
perfect — the strongest habit to carry forward. Because metric fixes landed, the same v1
checkpoint no longer scores what v1 published, so v1.1 compares against a re-baselined row
rather than 0.9218.

Both are an **overfit** — trained and measured on the same 13 layers. They answer "can we
regenerate roto we have been shown", not "can we roto an unseen shot". That is v2.

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
  exr.py           Silhouette's delivered mattes, for refereeing render conventions
  sfx/             .sfx reader, JSON form, writer seam
  render/          deterministic rasteriser (curves, raster)
  dataset/         layer manifest -> matte + spline program per layer
  program/         spline program <-> dense arrays
  keys/            keyframe selection by curve simplification, key-value refit
  model/           network, training, reconstruction, smoothing, curve loss, figures
scripts/           eval_keys.py, report_v1.py, and the v1.1 set:
                   train_v11.py / report_v11.py / summarise_v11.py,
                   sweep_operating_point.py, fig_worst_layers.py,
                   exp_conventions.py, exp_peak_frames.py,
                   exp_offset_jitter.py, exp_affine_target.py
                   ...and the v1.2 set: ledger.py (the exactness ledger),
                   train_v12.py / report_v12.py / summarise_v12.py,
                   exp_scoring_ceiling.py, exp_noise_floor.py, fig_worst_frames.py
tests/             123 tests
v1/                deliverables (gitignored): docs, checkpoint, results, figures
v1.1/              same, for the review response: ladder, sweeps, referees
v1.2/              same, for the handover: the ledger, the noise floor, worst-case tables
```

## Tests

```bash
pytest                    # 123 tests, ~85s — they render real archive frames
python scripts/ledger.py  # the exactness ledger; exits non-zero if a row is red
```

Four carry the most weight:

- **`test_dataset.py`** asserts `decode(encode(program))` renders back to the layer's own
  matte, at strides 1, 7 and 20. Tolerance is `1e-4`, which is 16-bit PNG quantisation on the
  stored matte and nothing else. If it needs raising, the representation lost something.
- **`test_keys.py`** measures keyframe selection against tracks with planted, known keyframes,
  so it is scored on ground truth rather than on its own output.
- **`test_v11.py`** pins the transform representation both ways: that the 6-number affine form
  the pipeline shipped with is *lossy* on a perspective transform, and that the 8-number
  projective form replacing it is exact. That asymmetry was a real ceiling on a real head, and
  a test is the only thing that stops it coming back.
- **`test_v12.py`** checks the two conversions nothing was watching — that window alignment uses
  the offsets the targets were built with, and that the transform loss's probe points land where
  the renderer puts them, both to 1e-9 crop px on the real archive. It also pins the
  de-teacher-forcing path the only way that means anything: hand it the artist's own transform
  track and it must reproduce the teacher-forced reconstruction exactly.

## Not in scope yet

- **Writing a real `.sfx`.** [src/roto/sfx/write.py](src/roto/sfx/write.py) is a documented
  seam that raises `NotImplementedError`. This needs no machine learning — it is serialisation
  from the IR — but nothing we write has ever been opened in Silhouette, and checking that
  needs a licence seat.
- **Generalising to unseen shots.**
- **Harder inputs.** The matte is rendered from the answer, so it is perfectly clean. A plate,
  or a mask from another tool, is not.
