# Infrastructure request — GPU training on ECS

**From:** Aman · **Date:** 2026-09-14

We train a PyTorch model. Single process, one GPU, no distributed training. Jobs are launched
on demand and run for hours. ECS is the agreed target.

---

## Compute

- **ECS cluster with GPU capacity.** GPU is not available on Fargate, so this needs an EC2
  capacity provider.
- **Instance class:** `g6.xlarge` or `g5.xlarge`. One GPU per task.
- **Scale to zero when idle.** Usage is bursty — a few jobs a week.
- **Two concurrent tasks.** Every experiment runs twice with different seeds.
- **Task timeout:** must allow multi-hour jobs.
- **ECR repository** to push our image to.
- **CloudWatch log group** for task logs.

## Storage

- **Our own S3 bucket** in our account, read/write. We own it, we handle its cost.
- **Cross-account read on `s3://production-citadel`** — `s3:GetObject` and `s3:ListBucket`,
  scoped to the prefix we need. Read only.

## IAM

- **ECS task role** with the two S3 permissions above. No long-lived access keys.
- **`kms:Decrypt`** on the prod key, if `production-citadel` is SSE-KMS encrypted.
- **Task execution role** for ECR pull and CloudWatch write.

## Our API service

We expose an internal API to trigger, monitor and stop training jobs. Its role needs:

- `ecs:RunTask`, `ecs:DescribeTasks`, `ecs:StopTask`
- `iam:PassRole` for the task and execution roles
- `logs:GetLogEvents` on the task log group

## Networking

- If tasks run in a private subnet: the **S3 gateway VPC endpoint policy must allow
  `production-citadel`**, not only our own bucket.
- Same region as `production-citadel`, if possible.

## Image

- Base image with CUDA, or a Deep Learning container.
- Our stack: **Python 3.12, PyTorch 2.12 (CUDA 13.0), OpenCV with OpenEXR support.**

---

## Questions

1. **What is the current G-family vCPU quota in the target region?** Often zero on new accounts
   and slow to raise.
2. **Is `s3://production-citadel` SSE-KMS encrypted, and who owns that account?** We need the
   bucket policy change and possibly a KMS key policy change there.
3. **Which region is `production-citadel` in?**

**Please start the quota check and the bucket policy request now** — those are the long-lead
items.
