# Infrastructure request — GPU training environment

**From:** roto ML team · **Date:** 2026-09-14 · **Status:** request, nothing provisioned

We train a computer-vision model that reconstructs rotoscoping shapes from mattes. Training has
run on a laptop GPU to date. That machine is going away, and the dataset is about to grow by up
to two orders of magnitude, so we need a hosted GPU environment.

This document is the infrastructure ask. It is self-contained — no knowledge of the model is
needed to act on it.

---

## 1. What the workload is

A single-process **PyTorch** training job. One GPU, **no distributed training, no multi-node**.

It is triggered over an internal API rather than run from a shell: a request enqueues a job, a
worker executes it, results are written to S3. A job is long-running and cannot complete inside a
web request.

| | today (10 shots) | target (up to 1,000 shots) |
|---|---|---|
| GPU memory used | **307 MB** | ~1–2 GB (model is unchanged; batch may grow) |
| host RAM | ~1 GB | **~75 GB if unchanged — we are fixing this in code** (see §6) |
| model size | 3.0M parameters, fp32 | same |
| training data | 19 MB | ~2 GB |
| output per run | 12 MB | ~12 MB |
| run duration | 22–31 min | hours — schedule must lengthen with the data |
| frequency | a few runs/week, in bursts | same pattern, longer runs |

**The GPU is not our constraint and we are not asking for a large one.** At 307 MB we use under
4% of an 8 GB card. What grows with the dataset is host RAM, storage throughput and run
*duration* — not GPU memory.

## 2. The main question for you

**Would you rather give us an EC2 instance we start and stop, or run this as on-demand
containers (ECS/EKS + a Celery worker on a GPU task)?**

Both work for us. We have a mild preference for **EC2 first** and would like your view.

| | EC2 instance | ECS/EKS + Celery GPU task |
|---|---|---|
| fits our burst pattern | yes, if we can start/stop it | yes, better — scales to zero |
| ops overhead for you | lower to stand up | higher to stand up, lower to run |
| suits long jobs (hours) | yes | yes, with the task timeout raised |
| our code changes | none | containerise + queue wiring |
| cost when idle | zero if stopped, but relies on us stopping it | genuinely zero |

**Our reasoning for EC2 first:** it lets us validate the environment — cross-account S3 access,
GPU drivers, reproducibility — in days rather than weeks, and the containerisation work is not
wasted afterwards. **If ECS is the house standard, say so now** and we will build for it from the
start rather than migrate later. We would rather do the work once.

If EC2: **`g6.xlarge` (L4, 24 GB)** or **`g5.xlarge` (A10G, 24 GB)**. Smallest current-generation
NVIDIA option. Not a p-series, not multi-GPU.

If ECS/EKS: the same GPU class, with a task timeout that accommodates multi-hour jobs and the
ability to run two tasks concurrently (we run every experiment twice with different random seeds,
and the two runs are independent).

## 3. Storage and data access

Our source data is in **`s3://production-citadel`, in a different AWS account. We only ever need
to read it.** Everything we derive is written to a new bucket in our account.

**Requested:**

1. **A new S3 bucket in our (staging) account**, read/write for our workload. Small: datasets are
   19 MB today, ~2 GB at target; each training run writes ~12 MB.
2. **Cross-account read on `s3://production-citadel`** — `s3:GetObject` and `s3:ListBucket`,
   scoped to the prefix we need. Read only, nothing else.
3. **An IAM role attached to the instance/task** carrying both. **No long-lived access keys
   anywhere.**

Our developer machines need the same two permissions, so we can smoke-test code locally against
the same data. Local runs write to a separate prefix in the same staging bucket.

## 4. Questions we need answered

Ordered by how likely each is to block us.

1. **What is the current G-family vCPU quota in the target region?** Fresh accounts frequently
   have this at **zero**, and raising it is a support request that can take several days. If one
   thing silently delays this by a week, it is this. Please check before anything else.

2. **Are objects in `production-citadel` encrypted with a customer-managed KMS key?** If so, our
   role also needs `kms:Decrypt` granted **in the key policy in that account** — a separate
   resource from the bucket policy, often owned by a separate team. This is the most common cause
   of cross-account S3 failure and it surfaces as a generic `AccessDenied` with nothing pointing
   at KMS.

3. **Who owns the account holding `production-citadel`, and what is the lead time for a bucket
   policy change there?** We would like that request started immediately, in parallel with the
   quota request — they are independent.

4. **Which region is `production-citadel` in?** We would prefer to run in the same region.
   Cross-region transfer of the archive costs money and time on every rebuild.

5. **Is Requester Pays enabled on that bucket?** It changes our CLI calls and who is billed.

6. **Is the bucket versioned, and are its objects immutable?** This matters to us more than
   usual — our results are reproducible only if a rebuild reads the same bytes. If versioning is
   on, we will record object version IDs alongside our outputs.

7. **Will our compute sit in a private subnet?** If so we need an **S3 gateway VPC endpoint whose
   policy allows `production-citadel`**, not just our own bucket. Default endpoint policies
   commonly scope to the local account and would silently block us.

8. **Is spot acceptable?** We are adding checkpoint/resume so an interruption costs minutes rather
   than a whole run. Spot would cut the bill substantially and we are happy to take it.

9. **Is the root/EBS volume encrypted?** Client footage is cached on local disk while we build
   derived datasets.

10. **Is there a house standard for internal APIs and job queues we should follow?** We are
    planning a small Django + Celery service, modelled on an existing internal service. Tell us
    if there is a preferred pattern, base image, or deployment pipeline.

11. **Retention and lifecycle on our staging bucket** — anything we should design around?

12. **Access method** — we would prefer **SSM Session Manager** over SSH, so there is no key
    management and no open port 22. Confirm that works for you.

## 5. What we need for image builds

If we containerise (required for ECS, useful either way):

- A **container registry** (ECR) we can push to.
- Either a **Deep Learning base image**, or confirmation that we build our own CUDA image. A
  ready-made DLAMI/DL container saves us roughly a day of driver work.
- Our stack, pinned: **Python 3.12, PyTorch 2.12 (CUDA 13.0), OpenCV with OpenEXR support.** The
  OpenEXR support is not optional — our source mattes are `.exr` and some OpenCV builds ship
  without the codec.

## 6. What we are changing on our side

So the sizing above is not taken as a fixed requirement:

- **Lazy data loading.** Our loader currently holds every frame in RAM. Unchanged, that is ~75 GB
  at target scale. We are replacing it with on-demand reads, which should bring host RAM back to
  a few GB regardless of dataset size. **Please do not size the host RAM for 75 GB** — assume
  32 GB is ample and we will confirm.
- **Checkpoint/resume**, so an interrupted job resumes instead of restarting. This is what makes
  spot safe for us.
- **A pinned dependency set**, so the image is reproducible.

## 7. Cost visibility

Please attach a **cost tag and a budget alarm** from day one. We expect this to be small — a few
GPU-hours a week and a couple of GB of storage — and we would like that to be visible rather than
assumed.

---

**Summary of what we are asking for:**

| | |
|---|---|
| compute | 1 × GPU, smallest current-gen NVIDIA (`g6.xlarge`/`g5.xlarge`), or the ECS equivalent |
| storage | 1 new S3 bucket in our account |
| access | cross-account **read** on `s3://production-citadel` via instance role |
| registry | ECR repository, if we containerise |
| access method | SSM Session Manager |
| governance | cost tag + budget alarm |

**The two long-lead items to start today: the G-family vCPU quota check, and the bucket policy
request on the account owning `production-citadel`.** Everything else can follow.
