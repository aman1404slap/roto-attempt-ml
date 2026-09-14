# Spline Roto — v2 Charter (the do-it-once plan)

Consolidates everything proven in v1→v1.2, Jon's holistic fundamental, the labels rule, and the
zero-margin framework into one plan designed to need few iterations: interfaces frozen first,
every decision paired with its test, and decision rules written before results arrive.

## 0. Where we verifiably are

Parsing, IR, rendering, and scoring are exact and test-pinned (60+ tests). Reconstruction from
clean alpha: ~0.97 soft IoU across 13 layers, point error ~0.9 px, with the model's OWN predicted
motion (cost of removing that training wheel: 0.0049). Keys at 0.76× artist, key F1 0.408
(record) via the constrained picker. Noise floor measured (±0.007 at 12k steps, ±0.002 at 40k);
every claim above is raw-log verified. Known debts: dataset alphas are one renderer law stale and
carry two measured-wrong stroke conventions; shape structure is still given as input; nothing
written has been opened in Silhouette.

## 1. Four laws (violations are bugs, not choices)

**L1 — Labels only in the loss.** The model's input is the silhouette (and later, plate) — never
artist dots, keys, or identities. Current given-structure (shape count, point counts, lifespans,
query identity) is a declared training wheel with a removal schedule (§4); anything label-shaped
entering the input outside that schedule fails review. Hyperparameters tuned against labels
(picker tolerance) are tuned on held-out data only.

**L2 — Holistic judging.** Two artists validly draw the same matte with 20 or 40 shapes; the
archive holds *a* correct answer, not *the* answer. Therefore: correctness = the rendered result
(decomposition-independent); the artist file supplies *style* — shape-count range, key sparsity,
point economy — as soft statistical targets. Exact dot-matching is legitimate only while
structure is teacher-forced; each de-teacher-forcing stage (§4) shifts loss weight from
dot-match → curve/render + style statistics. No stage may punish a valid alternative breakdown
for not being the archived one.

**L3 — Zero margin where we control it.** Bucket A (conversions, alignment, scoring, rendering,
interpolation) must be exactly zero, each item pinned by a ledger test that fails loudly.
Bucket B (model error) is minimized and reported worst-case: per-layer min, p95, and
frames-below-threshold — never averages alone.

**L4 — Statistics before conclusions.** Two seeds per quoted number; deltas under the measured
noise floor are weather; rank on frames-below-0.90; always score the exact shipping checkpoint.

## 2. Freeze the interfaces (week 0 — this is what kills iteration churn)

Frozen after review, versioned thereafter: (a) IR schema (projective transforms, per-key interp,
lifespans); (b) dataset format `v002` (below); (c) metric definitions (Euclidean everywhere,
soft IoU, constrained key F1); (d) split definitions; (e) the exactness ledger as a CI gate.
Any change to a frozen interface bumps a version and triggers exactly one re-baseline — never
silent drift.

## 3. Dataset v002 — one rebuild, everything at once

Re-render under the clamped-CR law (restores the 1.000000 ceiling); flip the two measured
conventions (no zero-width fill; hairline stroke gain with 1 px floor; duplicate endpoints);
projective transforms throughout; splits defined at build time: every-7th-frame holdout AND two
fully held-out layers (one small, one dense). Ledger green on v002 before any training. One
re-baseline (`v002_control`, 2 seeds) anchors all history. Nothing trains on v001 after this.

## 4. Model plan — staged removal of training wheels, loss evolving in lockstep

| stage | model invents | still given | loss emphasis | gate to advance |
|---|---|---|---|---|
| S0 (now) | geometry + motion + key timing (picker) | full structure | dot L1 + curve + temporal + probe-affine | v002 gates (§5) |
| S1 | + key timing head (learned, artist keys supervise the LOSS only; picker biased by predicted keyness) | structure | + key-time BCE | key F1 ≥ 0.50 at keys ∈ [0.75,1.3]× |
| S2 | + lifespans & point counts | shape count, identity | dot-match relaxed: curve/render primary + style stats (point economy) | render within 0.01 of S1 |
| S3 | + full breakdown (dynamic queries; canonical ordering — by group, centroid — before any Hungarian matching) | nothing | render + chamfer + style statistics (shape count, key sparsity, point density ranges from archive) | render ≥ 0.95 on held-out layers |

Each stage is one rung, two seeds, judged against frozen gates — results trigger the
pre-written decision rules, not re-planning. Jitter work continues only if it blocks a gate
(delta-prediction design note exists; code only on demonstrated need).

## 5. Acceptance gates (unchanged from v1.3 plan; now the standing definition of done per stage)

Soft IoU ≥ 0.97 mean and ≥ 0.90 every layer; frames < 0.90 ≤ 1% per layer; point p95 ≤ 3 px;
keys ∈ [0.75, 1.3]× with stage-appropriate key F1; held-out-frame gap ≤ 0.002; predicted motion
end-to-end; scoring ceiling exactly 1.0. Plus the deliverable gate: `sfx/write.py` implemented,
read(write(IR)) bit-exact in the ledger, and ONE file opened, rendered identically, and edited
in a real Silhouette seat (license: standing ask). Later stages add: held-out-layer, then
held-out-shot numbers reported alongside — generalization is claimed only from those.

## 6. Data & infra rules

Elements from the .sfx layer tree (gold = EXR-matched, silver = tree-defined, tagged in meta);
pairing QC (the sh0230 detector) gates every ingested shot; archive subset ingest and the move
to company-approved GPU infrastructure happen together — governance first (client data leaves
personal machines), parallel seeds second; one A10G/L4-class instance per run, same code, spot
pricing. Tier-2 path per the standing thesis answer: corrupt clean alphas to imitate Slapshot
error, grade against clean artist answers, swap in real Slapshot mattes when plates arrive.

## 7. Explicitly out of scope until S3 gates pass

Motion-blur passes; single-frame paint-stroke hair; multi-GPU; differentiable rasterization;
any new metric. Scope additions require a gate failure that names them.

## 8. The sentence over the whole plan

Freeze the ground once, grade the picture not the handwriting, keep labels in the loss, take the
training wheels off one at a time with the finish line written before the race — and prove it in
Silhouette, not just in our own mirror.
