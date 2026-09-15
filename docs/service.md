# The training service

**Written for: whoever picks this repo up next — including devops setting up the pipeline.**

Training moved off a laptop and behind an API. This note says how the service is shaped, how to
run it locally, and what is deliberately not in it.

The infrastructure request is [devops-requirements.md](../devops-requirements.md); the reasoning
behind the move is [aws-gpu-migration.md](../aws-gpu-migration.md).

---

## What this is

```
POST /api/runs  ──▶  Run row (Postgres)  ──▶  ecs:RunTask  ──▶  GPU task
                                                                  │
                                                     python3 ./manage.py train_run --run-id …
                                                                  │
                                              sync dataset ▸ train ▸ sync results
                                                                  │
GET /api/runs/<id>  ◀── stage, status, timings, artifact URI  ◀───┘
```

A 30-minute training job cannot run inside a web request, so the service is two halves: one
that accepts work and one that does it. They meet at a `Run` row and an ECS task definition.

**One image, three jobs.** Which job a container does is decided by the command ECS overrides
with — `train_run`, `build_dataset`, `score_run` — so there is one thing to build, push and keep
in step with the code.

## The two codebases, and the line between them

| | |
|---|---|
| `src/roto/` | the science. Plain Python, no Django, no boto3, no knowledge that this service exists. Tested by `pytest`. |
| `roto_app/` | the service. Django, orchestration only. Tested by `manage.py test roto_app`. |

The line is load-bearing in both directions:

- **The service never reaches into the science except through its public entry points** —
  `roto.v2.rungs.RUNS`, `roto.v2.train.train`, `roto.v2.dataset.build_dataset`,
  `roto.v2.score.score_run`. Everything else it does is sync files and update a row.
- **The science never imports the service.** `scripts/train_v2.py` still works on a machine
  with no database and no `.env`, which is what makes an experiment runnable without
  infrastructure.

Two consequences worth knowing:

**`boto3` stays out of the training path.** The contract is sync to local disk, run the
existing code unchanged, sync back. An S3-native data layer would be a rewrite in exchange for
nothing — the training code reads plain local paths and should keep doing so.

**The web process never imports torch.** Validating a rung name against the catalogue must not
require the training stack, or the web service needs the GPU image. `roto.v2.rungs` imports
`roto.v2.config` — a dataclass and nothing else — and a test in `roto_app/tests/test_api.py`
holds that line.

## Local setup

Local exists to answer one question: **does this code run on a GPU without crashing.** It never
produces a number that gets quoted. That is enforced, not remembered — see "The two fences"
below.

```bash
# 1. Postgres (the only containerised piece — CUDA is on the host, use it)
docker compose up -d db

# 2. Config
cp .env.example .env          # set ROTO_APP_TASK_API_KEY; STORAGE_ROOT=abc needs no AWS

# 3. Schema
python manage.py migrate
python manage.py createsuperuser        # or: POST /api/seed-data

# 4. Serve
python manage.py runserver 5000
```

Trigger a run:

```bash
curl -X POST localhost:5000/api/runs \
  -H 'x-task-api-key: <ROTO_APP_TASK_API_KEY>' \
  -H 'content-type: application/json' \
  -d '{"rung": "s3a", "seed": 1, "steps": 200, "dataset_version": "v003"}'
```

With `RUN_EXECUTOR=local` that runs the same management command as a subprocess on this
machine, logging to `$SCRATCH_DIR/local_runs.log`. With `RUN_EXECUTOR=ecs` it launches a GPU
task. **The command is identical either way** — which is the only reason a local smoke test
says anything about whether the deployed one will work.

Without the API at all:

```bash
python manage.py train_run --create --rung s3a --seed 1 --steps 200
```

### Running the suites

```bash
pytest                          # the science: 108 tests, no database, no Django
python manage.py test roto_app  # the service: 22 tests, against the compose Postgres
```

**Verified end to end on 2026-09-15, with `STORAGE_ROOT=abc`** — the whole flow, no AWS:

1. `POST /api/runs` → row created, subprocess launched, 202-ish response with the run id
2. dataset synced `abc/datasets/v003` → scratch, S3A trained on the GPU (60 steps), run
   directory synced back to `abc/runs/local/aman/s3a_seed1/`
3. `GET /api/runs/<id>` reported `IN_PROGRESS/TRAINING` then `COMPLETED`, with every stage
   timed and the cost numbers on the row
4. `manage.py score_run --rung s3a --seeds 1` **refused** it as a local run; `--allow-local`
   tabled it with `environment local` in the header and wrote the table beside the run
5. `manage.py build_dataset` read the archive in place and published to `abc/datasets/`

What is still unexercised is the **S3 backing** of that same code path and `ecs:RunTask` — both
need the bucket and the cross-account read.

## Deployment

Devops owns the pipeline. What this repo provides:

| | |
|---|---|
| `Dockerfile` | CUDA 13.0 base, torch pinned, no `CMD` — every container gets an explicit command |
| `.dockerignore` | keeps the research tree out. `data/` alone is 11 GB for 50 shots |
| `.circleci/config.yml` | lint, service tests, `makemigrations --check`, build, deploy — copied from the sibling service's shape |
| `ci/build.py`, `ci/deploy.py` | the house build-and-push and GitOps deploy scripts, unmodified |
| `.env.example` | the full config contract |

The web service runs `gunicorn roto_app.wsgi:application --bind 0.0.0.0:5000 --timeout 120`.
Training tasks get a command override and no web server.

**The science suite is not run in CI, deliberately.** It needs numpy, opencv and a CUDA torch
wheel; installing those to run it would add minutes to every build. The service suite needs
none of them, which is a property of the import boundary rather than an accident. If the
science suite should run in CI, it wants its own job on an image that already has the stack.

## Storage layout

```
<source>/                            nobody writes — the archive: .sfx, EXRs, plates
<storage>/datasets/<version>/        build_dataset writes
<storage>/runs/staging/<run>/        staging runs
<storage>/runs/local/<user>/<run>/   smoke tests
```

Only `build_dataset` reads the archive, and it needs no GPU. Training reads the derived dataset
and nothing else — which is why **the GPU task never has to hold client footage.**

### Two roots, one layout

`STORAGE_ROOT` and `SOURCE_ROOT` decide what `<storage>` and `<source>` are:

| | local (today) | staging |
|---|---|---|
| `STORAGE_ROOT` | `abc` — a gitignored folder | `s3://<bucket>` |
| `SOURCE_ROOT` | `data/spline_dataset_08_25_26` | `s3://production-citadel/<prefix>` |

**The layout inside is identical either way**, which is the entire point: local is staging with a
different root, not a second design, so the code that will run on ECS is the code that runs here.
Switching back once the bucket exists is one environment variable.

`roto_app/helpers/storage.py` is the only module that knows which backing is in play. Its folder
mode is an *incremental* copy — size and mtime, the same fields `aws s3 sync` compares — so a
dataset already on disk costs a stat per file rather than a re-copy, and neither mode ever
deletes from the destination.

One asymmetry, deliberate: when `SOURCE_ROOT` is a local directory the build **reads it in
place** rather than copying it into scratch. It is 11 GB and the build only ever reads it.

## The two fences

Once local and staging write to the same bucket, a two-shot probe becomes quotable by accident.
Two independent mechanisms stop that:

1. **The path.** Local runs land under `runs/local/<user>/`, never beside a staging run.
2. **The stamp.** Every checkpoint carries `provenance.environment` and a fingerprint of the
   dataset's manifest. `roto.v2.score.frozen_table` refuses to table a run stamped `local`, so
   the rule holds however the table is built — this service, `scripts/score_v2.py`, or a
   notebook. `--allow-local` exists to look anyway, and says so in the header.

A run from before stamping existed reports as `unstamped` and still tables. Refusing those would
invalidate every published number, which is the opposite of the guarantee.

## What is not in here yet

Named so they are decisions rather than omissions. All are from
[aws-gpu-migration.md](../aws-gpu-migration.md) §5.

- **§5.1 lazy alpha loading.** `ElementData.alphas` materialises every frame of every layer into
  RAM. ~75 GB at 1,000 shots. Required before the archive grows; harmless at 50.
- **§5.2 mid-run checkpointing and resume.** `torch.save` runs once, after the loop. A task that
  is stopped or reclaimed loses everything — so `POST /runs/<id>/stop` currently discards a run
  rather than pausing it, and a retry starts from zero.
- **§5.3 the 256-slot cap.** Hard-fails on an element wider than `n_slots`, correctly. There are
  known layers with 9,000+ shapes. The missing piece is the policy for what to do when it fires.
- **§5.5 source checksums in the build manifest.** The manifest records paths, not hashes. Fine
  while we owned the folder; not fine with a bucket another team can change.

## The rungs

`roto.v2.rungs` holds the ladder, S0 through S3A, one named configuration per charter S4 stage.

**A run on a larger archive starts at S3A, not at S0.** The rungs below it each answered one
question about one change, against a fixed dataset, and more data does not reopen them — you
cannot learn what a loss reweight cost geometry by adding shots. They are kept so a published
number can be traced to the configuration that produced it. `DEFAULT_RUN` is `s3a` and the API
defaults to it.
