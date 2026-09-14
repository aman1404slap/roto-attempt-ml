# Moving training off the laptop: AWS GPU migration

**Status: proposed, not started. Written 2026-09-14.**

The laptop GPU is going away — power limits now, a Mac shortly, and no CUDA on the other side.
At the same time the target dataset is moving from 10 shots to a number two orders of magnitude
larger. This note says what we need, what has to change in the code before a cloud run can be
trusted, and what the two scales imply.

**Written the way the design notes are: measured first.** Every number in §1 comes off
`runs/v2/*/train_log.json`, which has recorded the cost of every run since S0 precisely so this
decision would not have to be made on taste. Sources in the appendix.

The infrastructure request is a separate, forwardable document:
**[devops-requirements.md](devops-requirements.md)**.

---

## 1. Two scales, and they need opposite answers

### What we have been running

| | measured |
|---|---|
| model | **3.0M parameters**, fp32, no mixed precision |
| peak GPU memory | **307 MB** of 8,151 MB available — **3.8%** |
| wall clock per run | **22–31 min** (40k steps, 22–30 steps/s) |
| training data on disk | **19 MB** (10 shots, 21 layers) |
| checkpoint out | 12 MB per seed |

On that workload we would be asking for the smallest GPU available and turning it off between
sessions. **That argument dies the moment the dataset grows**, and it should, because the
constraint moves from "we barely use the card" to "the data no longer fits in memory".

### What 1,000 shots implies

Projected from the measured per-layer costs — 2.1 layers per shot, ~36 MB of materialised alpha
per layer, 141 element-frames per shot:

| shots | layers | dataset on disk | **alphas held in RAM** | epochs at the current 40k-step schedule |
|---|---|---|---|---|
| 10 (today) | 21 | 19 MB | 0.7 GB | 170 |
| 50 (everything we hold) | 105 | 95 MB | 3.7 GB | 34 |
| 200 | 420 | 380 MB | 15 GB | 8.5 |
| **1,000** | **2,100** | **1.9 GB** | **75 GB** | **1.7** |

**Two things break, and neither is fixed by a bigger instance.**

1. **Memory.** `ElementData.alphas` materialises every frame of every layer into RAM up front
   and keeps it — deliberately, because the read is too slow to pay per step
   ([traindata.py:176](src/roto/v2/traindata.py#L176)). At 1,000 shots that is ~75 GB of host
   RAM for the alphas alone. This needs a real lazy loader, and that is a code change (§5.1).
2. **The training schedule.** 40k steps at batch 6 is 240k samples. Today that is 170 passes
   over the data; at 1,000 shots it is **1.7**. The schedule was never sized for this, so
   "40k steps" stops meaning what it means now and every gate reading in the charter is
   calibrated against it.

**This is where GPU hours start to matter for real** — not because the model got bigger, but
because the schedule has to. That is the honest version of the sizing argument and it is a much
stronger one than "our laptop died".

### The recommendation, stated once

> **Build the infrastructure for 1,000 shots. Run the next experiment at ~50.**

These do not conflict. Everything in §3 and §5 — the API, the queue, the bucket layout, lazy
loading — is the same work either way, and lazy loading is *required* at 1,000 and *harmless* at
50.

The reason to run the experiment at 50 first is that **we do not yet know whether more data
fixes S3.** That is the open question the whole stage hangs on
([s3-attempts.md](luthra-understands/s3-attempts.md)): the model generalises perfectly within a
shot (held-frame gap 0.0027) and not at all across shots (0.2031), on eight training shots. Going
from 8 to ~40 training shots is a 5× increase and answers the question. If the answer is no, we
will have learned it without having first built a dataset 100× the size, and if the answer is
yes, we scale knowing it pays.

Charter §2 says one interface change per re-baseline, for the same reason: moving the hardware,
the data volume and the execution model in a single step measures none of them.

### ⚠️ Open question that gates the number

**Our archive is 50 shots.** `data/spline_dataset_08_25_26/` holds 50, of which 4 are excluded
outright, 3 deferred for size, 10 are Tier 1 and 5 are Tier 2.

**Does `s3://production-citadel` hold ~1,000 shots, or is the 50 we have the whole delivery?**
If prod holds the larger population then §1's projections are the plan; if not, "1,000 shots" is
an acquisition question before it is an infrastructure one. This needs answering before the
instance is sized.

## 2. Test MPS on the Mac first — it may change what local means

An M-series Mac can plausibly run a 3M-parameter, 307 MB job at usable speed, which is what
§3 needs *local* to be: a smoke test that the code runs on a GPU, not a result.

`--device mps` runs today without a code change:
[train.py:54](src/roto/v2/train.py#L54) passes the flag straight to `torch.device`. The risk is
operator coverage — MPS still has gaps and the loss stack is where they would bite.

**Do it in parallel with the paperwork, not instead of it.** The external lead times in §8 start
whether or not MPS works.

## 3. The target architecture

Two environments, one source of truth, one writable bucket.

```
  s3://production-citadel                        account A, READ-ONLY
      raw .sfx + EXRs + plates
            │
            │ cross-account read (both environments read this)
            ├──────────────────────────────┐
            ▼                              ▼
  ┌─────────────────────┐        ┌─────────────────────────────┐
  │  LOCAL (Mac)        │        │  STAGING (EC2 + GPU)        │
  │  a few shots        │        │  the real runs              │
  │  "does it run?"     │        │  two seeds, full schedule   │
  │                     │        │  triggered over an API      │
  └─────────┬───────────┘        └──────────────┬──────────────┘
            │ writes                            │ writes
            ▼                                   ▼
  s3://<staging-bucket>/runs/local/…   s3://<staging-bucket>/runs/staging/…
                          account B, READ-WRITE — to be created
```

### 3.1 The two environments

**Local is a smoke test and nothing else.** A handful of shots, a short schedule, one seed. Its
only question is *does this code run on a GPU without crashing*. It never produces a number that
gets quoted.

**Staging is where every result comes from.** Full schedule, two seeds, the real dataset,
triggered over an API rather than a shell on someone's machine. It is also the POC target — so
the thing we demo and the thing we measure are the same environment, which is the point of
routing local through staging's bucket too.

**Both read prod. Both write staging.** Local writes under a different prefix. That keeps one
storage layer and one set of credentials, and it means a local run is a staging run with a
smaller argument list rather than a different code path.

### 3.2 Bucket layout

| path | who writes | what |
|---|---|---|
| `s3://production-citadel/<prefix>/` | **nobody — read only** | the archive: `.sfx`, EXRs, plates |
| `s3://<staging>/datasets/<version>/` | build step | derived training data (19 MB → ~1.9 GB) |
| `s3://<staging>/runs/staging/<run>/` | staging | real runs: checkpoints, logs, scores |
| `s3://<staging>/runs/local/<user>/<run>/` | local | smoke tests |

**The rule that makes this safe: nothing under `runs/local/` may ever be quoted as a result**,
and that should be enforced in code rather than remembered (§5.4).

### 3.3 API-triggered training

A 30-minute training job **cannot run inside a web request**, so the API is necessarily two
pieces: something that accepts a job and something that runs it. `POST /runs` enqueues and
returns an id; `GET /runs/<id>` reports status; the worker runs the existing
[train_v2.py](scripts/train_v2.py) as a subprocess and syncs results to staging.

**On using the sibling Django service's shape for this:** the concern is that Django's ORM,
migrations and admin are a lot of machinery for "enqueue a subprocess and report its status", and
this service will never have a schema worth that. Against that, matching an existing in-house
service means the deployment story, auth and conventions are already solved and reviewed. **If
the sibling service is the house standard, follow it** — the overhead is real but small and
one-time, and consistency across services is worth more than saving a few hundred lines here.
Django plus Celery is a coherent answer to the blocking-job problem, which is the part that
actually matters.

**What the API must carry, whatever the framework:** the run name, the dataset version, the seed,
the step count, and the environment tag. Those five are what make a run reproducible, and they
belong in the request rather than in the worker's defaults.

## 4. Two accounts, and the three things that silently block one

Source data is in **`s3://production-citadel`, a different account, read-only.** Everything we
derive goes to a **new staging bucket in our account**.

**A read-only, immutable source is better for our method than a local folder**, because it cannot
drift underneath us. It is only better if a rebuild can be *proved* to match — which is §5.5.

### The blockers, in order of how often they kill a cross-account setup

**KMS.** If prod objects are encrypted with a customer-managed key, an S3 bucket policy alone is
**not enough** — the instance role also needs `kms:Decrypt` granted in the **key policy in the
prod account**, a separate resource often owned by a separate team. It fails as a bare
`AccessDenied` with nothing pointing at KMS.

**Two accounts, two approval paths.** The bucket policy on `production-citadel` is changed by
whoever owns that account: different team, different ticket, different lead time. It is
independent of the GPU quota, so both start on day one.

**Private subnet plus VPC endpoint.** With no internet egress, S3 goes through a gateway endpoint
whose policy must explicitly allow the **prod** bucket. Defaults commonly scope to the local
account and silently exclude it.

All three are questions in [devops-requirements.md](devops-requirements.md).

## 5. Code changes, ranked

### 5.1 Lazy alpha loading — blocking at any scale above ~200 shots

[traindata.py:176](src/roto/v2/traindata.py#L176) materialises every alpha frame of every layer
into RAM and keeps it. The docstring says what it costs — ~475 MB for 13 layers — and why it was
done that way: the per-step read was too slow.

At 1,000 shots that is ~75 GB. The fix is a `DataLoader` with worker processes and on-demand
reads, which is also what makes multi-worker prefetch possible and is likely to *improve* the
22–30 steps/s throughput noted in §1.

**This is the single change that "1,000 shots" actually requires.** It is worth doing at 50 shots
so it is debugged before it is load-bearing.

### 5.2 Mid-run checkpointing and resume — blocking

`torch.save` runs **once, after the training loop**, at
[train.py:939](src/roto/v2/train.py#L939). A 31-minute run that dies at minute 30 loses
everything — and at 1,000 shots the runs get much longer than 31 minutes.

On-demand that is an annoyance. **On spot it is a certainty**, and spot is roughly a 70% saving.
It is also what makes an API-triggered job safe to retry.

Needed: `state_dict` + optimizer + scheduler + step counter + RNG state written every N steps, and
a `--resume` that picks up from the last one. Perhaps 40 lines, plus a test that kills a run
mid-way and confirms resume lands in the same place.

### 5.3 The 256-slot cap will hard-fail on a larger archive

[train.py](src/roto/v2/train.py) raises `ValueError` if any element has more shapes than
`n_slots`, on purpose — *"a slot bank that cannot hold the widest element silently drops its
tail"*. Tier 1 tops out at 247 shapes, which is why 256 was chosen.

We already know of layers with **9,008 and 9,077 shapes** in the deferred set. On a 1,000-shot
archive those are not exceptions, they are a population. Before any large build we need a stated
policy: raise the cap, exclude wide layers with the exclusion recorded, or split them. **The
current behaviour — a hard crash — is the right default and should stay**; what is missing is the
decision about what to do when it fires.

### 5.4 Stamp the environment on every run, and refuse to table a local one

Everything in this project rests on not confusing weather for a result. Once local and staging
write to the same bucket, a two-shot smoke test number becomes quotable by accident.

Add `environment: local|staging` and the dataset fingerprint to `train_log.json`, and have the
reporting path refuse to put a `local` run in a results table. Cheap, and it makes the guarantee
structural rather than remembered.

### 5.5 Record source checksums in the build manifest

[datasets/v003/manifest.json](datasets/v003/manifest.json) records the source as a **relative
local path** (`data/spline_dataset_08_25_26`) plus per-element `.sfx` paths. **No hashes.**

That was fine while we owned the folder. It is not fine when the source is a bucket in another
account that another team can change. Add a hash per source `.sfx`, and ideally per matte input.

**This fits how the pipeline already thinks.** [ingest.py:11](src/roto/v2/ingest.py#L11) records
that *every mtime in the delivery is identical*, so final-file selection is already
content-derived rather than filesystem-derived. Hashing the inputs is the same principle one
level up — and it is what makes Gate 1 (§7) a real test rather than an impression.

### 5.6 Pin the environment

[pyproject.toml](pyproject.toml) declares `numpy` and `opencv-python` and **does not declare
torch at all** — it is installed ad hoc. We are on Python 3.12.12 and torch 2.12.0+cu130.

If the instance silently gets a different torch, every number shifts and nothing will say why.
Pin torch with its CUDA build, or add a lockfile. Required before the API can build an image.

### 5.7 Teach `pick_device` about MPS

[train.py:54](src/roto/v2/train.py#L54) and
[reconstruct.py:427](src/roto/v2/reconstruct.py#L427) both fall back `cuda → cpu`. On the Mac
that means scoring, reconstruction and every figure script silently run on CPU — and under §3
that is most of what local does.

### 5.8 Cost reporting is CUDA-only

[train.py:943](src/roto/v2/train.py#L943) and [:958](src/roto/v2/train.py#L958) record peak memory
and device name only when `device.type == 'cuda'`. Harmless, except that this log is exactly what
§1 is built from — losing it on other hardware costs us the next version of this decision.

### 5.9 Keep boto3 out of the training path

Sync to local disk, then run the existing code unchanged. The code reads plain local paths
([build.py:174](src/roto/v2/build.py#L174)) and that is the right design — an S3-native data layer
would be a rewrite in exchange for nothing. The worker does:

```bash
aws s3 sync s3://production-citadel/<prefix> ./data/...      # read-only, prod account
PYTHONPATH=src python -m roto.v2.dataset ...                  # unchanged
aws s3 sync ./datasets/<version> s3://<staging>/datasets/<version>
aws s3 sync ./runs/v2 s3://<staging>/runs/<environment>/
```

### 5.10 Two seeds: decide about concurrency explicitly

Every result is two seeds. At 307 MB each they would run concurrently with room to spare, halving
wall clock — but [s2-attempts.md](luthra-understands/s2-attempts.md) already flags that concurrent
work pollutes timing: S2A's two seeds took 25 and 30 minutes for identical work because scoring
was running alongside. Either run sequentially, or run together and record that the minutes are
wall time and not training speed. Doing it silently is the only wrong answer.

## 6. Where the build runs

Training needs **only the derived dataset**. The raw archive is required *only* to build it, and
the build is CPU-only. So the GPU instance never has to hold client footage — provided the build
runs somewhere else.

**This overturns an earlier recommendation, recorded rather than quietly dropped.** The first
version of this note proposed building locally so client footage never left our machine. With
`production-citadel` as the source of truth that is no longer the clean option: **the build has
to run wherever it can read prod.** Three placements work — the GPU instance between runs, a
separate cheap CPU instance, or a local machine granted cross-account read — and the choice is
mostly about who is permitted to hold client footage, not about engineering.

At 1,000 shots the build is itself a substantial job and probably wants to be its own queued task
rather than a step inside the training job.

## 7. Two gates before any new science

Both cheap, both before the dataset grows.

### Gate 1 — rebuild `v003` from `production-citadel` and confirm it is byte-identical to the local `datasets/v003`

If it matches, the whole path is proven end to end: cross-account read works, KMS works, the build
is deterministic across machines, and **every number already published stays comparable.**

If it does not match, we have found a real problem for the price of one CPU run — and far better
here than three attempts into a 1,000-shot build. §5.5 is what makes this a test rather than an
impression.

### Gate 2 — re-run `s2c` or `s3a` on the new GPU, same dataset, two seeds

Confirm it lands within the old **0.0022** seed spread, and measure the new spread.

**This is not optional bookkeeping.** The entire method rests on *a move smaller than the seed
spread is weather, not a result*. That 0.0022 was measured on the RTX 5050. Different hardware
means different floating-point ordering, so on a new machine **we do not know what our noise floor
is** until we measure it. One hour, and it protects every number downstream.

**Only then does the dataset grow.**

## 8. Order of operations

Three items have external lead times and start on day one. Everything on the right proceeds
without waiting.

| start immediately, in parallel | ours to do meanwhile |
|---|---|
| G-family vCPU quota request | lazy alpha loading (§5.1) |
| `production-citadel` bucket policy, other account | checkpoint/resume (§5.2) |
| KMS key policy question | manifest checksums (§5.5) |
| staging bucket creation | pin the environment (§5.6) |
| MPS test on the Mac (§2) | environment stamping (§5.4) |

## 9. Open questions

1. **Does `s3://production-citadel` hold ~1,000 shots?** (§1) Our archive is 50. This gates the
   instance sizing and the schedule design.
2. **Is our local `data/` a copy of prod, and is prod now authoritative?** Changes how Gate 1's
   result should be read — a mismatch could be a bug or could be a stale local copy.
3. **What is the policy for layers wider than the slot cap?** (§5.3) Raise, exclude, or split.
   Needed before any large build.
4. **Where does the build run?** (§6) An access-policy question more than an engineering one.
5. **Does OpenCV decode our EXRs on macOS?** The tracker already records a shot whose EXRs would
   not decode in a particular OpenCV build, so this is a known-real hazard. The
   `OPENCV_IO_ENABLE_OPENEXR` env var is already handled
   ([exr.py:32](src/roto/exr.py#L32)); the wheel's build options are what vary.
6. **Sequential or concurrent seeds** (§5.10) — decide and record.

---

## Appendix — where the numbers come from

`runs/v2/<run>_seed1/train_log.json`, written by [train.py](src/roto/v2/train.py) at the end of
every run:

| run | params | steps/s | wall | peak GPU |
|---|---|---|---|---|
| `s0` | 2.98M | 29.9 | 22 min | 286.1 MB |
| `s2c` | 3.07M | 28.0 | 24 min | 295.7 MB |
| `s3a` | 3.03M | 21.7 | 31 min | 307.3 MB |

Measured directly:

| | |
|---|---|
| `datasets/v003` on disk | 19 MB (1,433 alpha PNGs at 256 px, plus tensors and IR) |
| `data/spline_dataset_08_25_26` | **50 shots**, 11 GB |
| one run directory | 12 MB |
| GPU total memory | 8,151 MB (RTX 5050 Laptop) |
| current stack | Python 3.12.12, torch 2.12.0+cu130 |
| declared dependencies | `numpy>=1.24`, `opencv-python>=4.8` — **torch absent** |
| seed spread, S2 | 0.0022 on-screen soft IoU |
| alphas held in RAM | ~475 MB for 13 layers ≈ 36 MB per layer |

Projections in §1 scale those per-layer figures at 2.1 layers and 141 element-frames per shot,
both taken from Tier 1 (10 shots, 21 layers, 1,412 element-frames).
