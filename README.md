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

**v2, at stage S3A.** Training runs on AWS, triggered over an API; see
[docs/service.md](docs/service.md) for the service and
[aws-gpu-migration.md](aws-gpu-migration.md) for why it moved and what is still open.

The v2 ladder is one named configuration per charter S4 stage, in
[`roto.v2.rungs`](src/roto/v2/rungs.py):

| rung | what it adds | outcome |
|---|---|---|
| `s0` | geometry and motion, structure given | 0.9717 on-screen soft IoU, 0.830 px, 1.119× keys |
| `s1` | the key-timing head | run, closed, **does not pass** — over-random key F1 0.0914 against a 0.15 bar |
| `s2a` | the loss reweight alone | preparation for S3; costs geometry nothing measurable |
| `s2b` | the lifespan head | the rung the render gate is about |
| `s2c` | the point-count head, `n_points` out of the query input | the training wheel off |
| `s3a` | **one shared bank of 256 slots, replacing the per-element query table** | the live rung |

**S3A is the open question and the default.** The S2 checkpoints reconstruct at 0.9652 with
their query rows and **0.0377** with those rows freshly initialised, so the table was carrying
~96% of the result and S3's whole job is to replace it. The model generalises perfectly within
a shot (held-frame gap 0.0027) and not at all across shots (0.2031) —
[luthra-understands/s3-attempts.md](luthra-understands/s3-attempts.md) is where that stands.

**A larger archive starts at S3A, not at S0.** The rungs below it each answered one question
about one change against a fixed dataset, and more data does not reopen them.

Design notes, in order: [v2-s1-design-note.md](v2-s1-design-note.md),
[v2-s2-design-note.md](v2-s2-design-note.md), [v2-s3-design-note.md](v2-s3-design-note.md).
The rules are [v2_charter.md](v2_charter.md); the running record is
[v2-tracker.md](v2-tracker.md).

### v1 is deleted

v1 through v1.3 shipped and were removed on 2026-09-15. The boundary that made that a delete
rather than an untangling is documented in [`roto/v2/__init__.py`](src/roto/v2/__init__.py) and
enforced by `tests/test_v2_boundary.py`, which now asserts the modules stay gone. What each
round established, and the numbers it published, are in git history at `af51dfb` and earlier.

Three things v1 established that v2 still rests on, recorded because they are load-bearing and
no longer have a document of their own:

- **Re-rendering the targets correctly was worth +0.0048 soft IoU** — more than v1.1's whole
  architecture ladder against the same anchor. The ground mattered more than the model.
- **Every scorer that builds its own `RenderConfig` silently reasserts the wrong conventions.**
  That bug class appeared three times. It is why the dataset owns its conventions.
- **A head can pass its own gate having learned nothing.** The v1.3 key head scored 0.803 where
  firing on every live frame scores 0.797, because key density ranges 13× across layers. Every
  key metric carries its trivial baselines for that reason.

## Setup

Python 3.12, numpy, opencv-python, torch. A CUDA GPU for training; CPU is enough to build a
dataset or read the archive.

```bash
pip install -e ".[dev]"
```

For the service — Django, Postgres in Docker, and the API — see
[docs/service.md](docs/service.md).

`data/`, `datasets/` and `runs/` are gitignored: archive material and generated artifacts stay
out of version control, and in the deployed setup they live in S3.

## Commands

Everything runs two ways. The **shell** path needs nothing but the package, and is how an
experiment is driven:

```bash
export PYTHONPATH=src                                   # or pip install -e .

python -m roto.v2 dataset datasets/v003 --tier tier1    # build a dataset
python -m roto.v2 ledger  datasets/v003                 # the exactness ledger; non-zero on RED
python -m roto.v2 subset                                # the shot selection and its reasons

python scripts/train_v2.py s3a --seed 1                 # train one rung, one seed
python scripts/score_v2.py s3a --seeds 1 2              # score into the frozen table
```

The `scripts/` entry points put `src` on the path themselves; `python -m roto.v2` needs it set
or the package installed.

The **service** path is the same work, triggered over an API and run on ECS:

```bash
python manage.py build_dataset --version v004 --tier tier1
python manage.py train_run --create --rung s3a --seed 1
python manage.py score_run --rung s3a --seeds 1 2
```

They share the rung catalogue and the training loop — the service adds S3 sync and a row in a
database, and nothing else. That is deliberate: `src/roto/` has no Django in it and no knowledge
that the service exists.

## Which shots are used

v2 **tags rather than excludes** (charter S6, D4). v1 dropped a layer when a measured rule
fired; v2 records the grade and lets the run decide, because a threshold calibrated on 18
archive layers is not a property of the task. The selection and every reason is
[`roto.v2.subset`](src/roto/v2/subset.py), printable with `python -m roto.v2 subset`.

The archive is **50 shots**: 4 excluded outright, 3 deferred for size, 10 Tier 1, 5 Tier 2.
`datasets/v003` is Tier 1 — 10 shots, 21 elements, 1,412 element-frames.

Pairing QC grades each element gold or silver and can fail a whole shot; the grade travels with
the element into the manifest rather than deciding for it.

## Layout

```
src/roto/          the science. No Django, no boto3.
  ir.py            the intermediate representation everything converts through
  metrics.py       iou, soft_iou
  exr.py           Silhouette's delivered mattes, for refereeing render conventions
  smoothing.py     track smoothing, shared
  sfx/             .sfx reader, JSON form, writer seam
  render/          deterministic rasteriser (curves, raster)
  program/         spline program <-> dense arrays
  keys/            keyframe selection by curve simplification, key-value refit
  dataset/         crop, arrays, layers — version-independent mechanics
  v2/              the round: ingest, manifest, qc, build, splits, dataset, ledger,
                   traindata, net, losses, train, config, rungs, reconstruct,
                   score, report, provenance, subset, cli

roto_app/          the service. Django, orchestration only.
  models/run.py    one row per training run
  views/           the API: create, status, stop
  services/        paths (the S3 layout), launcher (ecs | local), train_service
  helpers/         aws (sync), ecs (RunTask/Describe/Stop), utils
  management/commands/  build_dataset, train_run, score_run — what a container runs

scripts/           train_v2.py, score_v2.py, and the S2/S3 experiments:
                   exp_s2_baselines.py, exp_s2_visibility.py, exp_s3_baselines.py,
                   sweep_s2_lifespan.py, fig_s2_attempt.py, fig_different_answer.py
tests/             the science suite — 108 tests, no database, no Django
roto_app/tests/    the service suite — 17 tests
ci/, .circleci/    build, deploy and pipeline, in the house shape
Dockerfile         one image; the command override decides which job it does
docker-compose.yml local Postgres, and nothing else
```

## Tests

Two suites, because there are two codebases and the boundary between them is the point.

```bash
pytest                            # the science: 108 tests, ~25s — they render real frames
python manage.py test roto_app    # the service: 22 tests, needs the compose Postgres
PYTHONPATH=src python -m roto.v2 ledger datasets/v003   # the exactness ledger
```

Five carry the most weight:

- **`test_v2_dataset.py`** asserts the built dataset is what it claims: the program round-trips
  to the element's own alpha, and the split recorded at build time is the split training reads.
- **`test_keys.py`** measures keyframe selection against tracks with planted, known keyframes,
  so it is scored on ground truth rather than on its own output.
- **`test_v2_s2.py`** pins the S2 heads' semantics — lifespan, point count, and the decode that
  turns them into shapes — including that a head must *vary with the frame*. A head wired to the
  raw query embedding would train, report a falling loss, and predict a constant.
- **`test_v2_boundary.py`** asserts v1 stays deleted, that v2 names none of its modules, and
  that importing v2's dataset path pulls in neither torch nor any v1 module.
- **`test_v2_provenance.py`** asserts the thing everything else rests on: a run stamped `local`
  cannot reach a results table, a mixed-fingerprint table says so, and a run from before
  stamping still tables rather than being voided.

## Not in scope yet

- **Generalising to unseen shots.** This is what S3 is for and it is not solved: the model
  generalises within a shot (held-frame gap 0.0027) and not across shots (0.2031). The dataset
  withholds whole shots at build time, so the claim is measurable; it is not yet earned.
- **Harder inputs.** The matte is rendered from the answer, so it is perfectly clean. A plate,
  or a mask from another tool, is not.

- **Scale.** The dataset is 10 shots of an archive of 50, and the plan targets 1,000. Two
  things break before that and neither is fixed by a bigger instance: alphas are materialised
  into RAM up front (~75 GB at 1,000 shots), and a 40k-step schedule that is 170 passes over
  today's data is 1.7 over that one. Both are named in
  [aws-gpu-migration.md](aws-gpu-migration.md) §5.1 and §1.

Writing a real `.sfx` **is** in scope: [src/roto/sfx/write.py](src/roto/sfx/write.py) emits both
containers and all four dialects, and `read(write(IR))` is bit-exact on every archive shot. What
is still open is one file **opened in Silhouette**, which needs a licence seat.
