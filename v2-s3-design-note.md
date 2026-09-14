# S3 — dynamic queries: design note

**Status: REVIEWED AND ACCEPTED, 2026-09-14.** All three questions in §8 were answered *yes*:

1. **The gate amendment in §3 is accepted.** S3 advances on render **over the traced-silhouette
   baseline**, with style statistics as hard constraints. Raw render quoted beside it.
2. **Charter §7 gives way to plan §6 on differentiable rasterization.** The low-res composite
   raster is in scope for S3, as §5.1 recommended.
3. **Four held-out layers is accepted as the gate population, with the sample flagged.** S3 does
   *not* wait for the Tier 2 expansion — that stays queued behind S3 per charter §2 (one
   interface change per re-baseline). Trained-element render is reported beside the held-out
   number, and a pass on four layers alone is reported as unproven rather than as the gate.
   This is the §7 stop condition, promoted to a standing rule.

**The operating rule for this stage, stated by the charter owner:** *try to make the plan work
with tweaks; if it does not work, record it and keep the record.* That is what S1 bought (a
closed stage and a corrected gate for one afternoon) and what S2 bought (a passed stage and two
withdrawn predictions). A stage that fails and is written down honestly is a result.

Per [v2_implementation_plan.md](v2_implementation_plan.md) §6 a design
note is *required* before any S3 code, and per charter §4 S3 may not start before the S2 gate —
which [S2 passed](v2-tracker.md) on 2026-09-14. Governed by [v2_charter.md](v2_charter.md) §4
(the ladder), L1 (labels only in the loss), L2 (grade the picture, not the handwriting), L3
(worst case, every layer) and L4 (two seeds).

**Written the way S2's was:** measurements first, gate priced before the design commits to it.
`scripts/exp_s3_baselines.py` → `runs/v2/s3_baselines.json`.

---

## 1. What S3 is

The last thing on the cheat sheet comes off. S0 was handed the whole structure; S1 added key
timing; S2 took away lifespans and point counts. S3 takes away **identity** — the per
`(element, shape)` query row — and with it the shape *count*. The model is handed a cutout and
must invent the breakdown.

| | today (S2) | at S3 |
|---|---|---|
| shape identity | given: one learned embedding row per `(element, shape)` | **invented** — queries produced from the encoder |
| shape count | given | **invented**, up to a fixed maximum |
| lifespan, point count | predicted (S2) | predicted |

## 2. The number that frames everything else

S2 reconstructs at **0.9652** on-screen soft IoU. The same checkpoints, with the query rows
freshly initialised — which is exactly what S3 must replace them with — reconstruct at
**0.0377**.

> **The query table is carrying ~96% of the result. S3 is not the next rung on the ladder; it
> is the rest of the problem.**

That gap is the honest measure `net.RotoNetV2` has claimed since v1 and the thing S3 exists to
close. Every number below should be read against it.

## 3. The gate is clearable without any decomposition

Charter §4 sets S3's gate at **soft IoU ≥ 0.95 on held-out layers**. Soft IoU is
*decomposition-blind by construction* — that is charter L2 working as intended when it stops us
punishing a valid alternative breakdown, and a hole when it is the only thing in the gate.

So: take the artist's own alpha, find its outline with `cv2.findContours`, resample to a
plausible control-point count, rebuild it as real IR shapes and render it back through the
project's own renderer. No model, no decomposition, no understanding — the answer traced off the
question.

| baseline | shapes used | trained | worst element | **held-out layers** |
|---|---|---|---|---|
| `trace_1` @16 | 1.0 | 0.7778 | 0.3838 | 0.8268 |
| `trace_1` @64 | 1.0 | 0.8611 | 0.4304 | 0.9143 |
| `trace_k` @32 | 1.2 | 0.9076 | 0.7901 | 0.9449 |
| **`trace_k` @64** | **1.2** | 0.9368 | 0.8312 | **0.9697** |

**On the population the gate actually reads, a traced silhouette clears it.** `trace_k@64`
scores **0.9697 against a 0.95 bar**, using **1.2 shapes** where the artist used between 1 and
247. It is not roto. It cannot be edited the way an artist needs. It would pass.

Two honest qualifications, because they matter:

- On **trained** elements the same baseline reads 0.9368, below the bar. The gate falls on the
  held-out population specifically.
- **The held-out-layer population is four elements** (`ts_021182__Lady_and_Man_matte`,
  `ts_021182__Screen_matte`, `ts_021182__Sub`, `ts_021555__Red`), and three of them are simple.
  A gate read on four layers is a thin gate regardless of what it says, and that is a second
  problem with it.

### Proposed amendment, for review

> S3 advances on **render over the traced-silhouette baseline**, with the raw figure quoted
> beside it, **and** on style statistics as *hard* constraints rather than soft targets:
> shape count within the archive's range for a comparable layer, points per shape within it,
> key economy unchanged at [0.75, 1.3]×. Held-out layers stay the population, with held-out
> **shots** reported beside them because four layers is a thin sample.

This is the same correction S1's gate needed and for the same reason — a metric that a trivial
answer saturates is not measuring the thing the stage is for. The difference is that S1's
amendment was proposed after the gate had already been cleared by a dial; this one is proposed
before any S3 code exists.

**What the style numbers should be set from** (measured, trained elements):

| | range | median |
|---|---|---|
| shapes per element | 1 – 247 | 16 |
| points per shape | 4 – 68 | 18 |
| keys per live frame | 0.021 – 1.184 | 0.474 |

## 4. Burial: a question I expected to answer the other way

Before measuring I argued that if the median shape were nearly invisible inside the union, then
plan §6's chamfer-to-artist-curves would be scoring handwriting the picture cannot show, and
render-plus-style would be the only defensible loss. **That is not what the data says.**

Over 5,178 sampled live `(frame, shape)` cells — not just lifespan boundaries:

| | p5 | p25 | **median** | p75 | p95 |
|---|---|---|---|---|---|
| fraction of a shape's own ink nothing else draws | 0.000 | 0.118 | **0.296** | 0.513 | 0.892 |

Only **16.5%** of live cells are buried below 0.05, against **61.4%** of lifespan *boundaries*
in the S2 measurement. Both numbers are true and they answer different questions: a shape is
usually substantially visible while it is on screen, and usually invisible at the moment it
arrives or leaves.

**So chamfer keeps its place in plan §6's loss.** The median shape contributes ~30% of its own
ink uniquely, which is signal. Recorded because it reverses the position I was going to argue.

## 5. Deviations from plan §6, proposed

**5.1 — plan §6's loss list conflicts with charter §7, and the conflict needs resolving before
the build.** Charter §7 puts *differentiable rasterization* explicitly out of scope "until S3
gates pass". Plan §6's loss list opens with "composite render (union of per-shape polylines
rasterized at low res)" — which is a differentiable rasteriser. One of the two documents has to
give. **Recommendation:** treat the low-res composite raster as in scope for S3, because §7's
list is about *scope creep beyond the ladder* and this is the ladder's own stage; but it is a
charter question, not ours.

**5.2 — a fixed maximum shape count of 256 for Tier 1.** v003 runs 1–247 shapes per element.
256 covers it with nothing to spare, and the deferred shots (9,008 shapes on one layer) do not
fit any fixed maximum — which is the tracker's standing reason they re-enter at S3 rather than
before. The maximum is a recorded interface number, not a hyperparameter to tune.

**5.3 — canonical ordering is measured against Hungarian, not assumed better.** Plan §6 makes
canonical ordering (transform group, then centroid y,x at a reference frame) the first attempt
and Hungarian "only as a fallback ablation". That ordering is a guess about artist convention,
and this archive has already produced several such guesses that did not survive. Both are
cheap; run both.

**5.4 — the encoder-only floor (0.0377) is the rung-zero number, and the first rung should be
allowed to be terrible.** S1 and S2 both started from a working system and changed one thing.
S3 starts from something that does not work at all, so the first rung's job is to move 0.0377
somewhere — not to approach 0.95.

## 6. Proposed plan of work

Rungs, two seeds each, ~25 min a seed, reported with a card in the attempt log as S2's were.

1. **S3a — queries from the encoder, everything else S2.** Learned pooling over encoder tokens
   → a fixed bank of queries; canonical ordering; losses unchanged from S2. Answers: how far
   does 0.0377 move when the query table is *derived* rather than looked up?
2. **S3b — the loss moves to render + chamfer + style.** Only if S3a shows signal.
3. **S3c — Hungarian matching as the measured alternative to canonical ordering.**

Stop after S3a if it does not move: the remaining rungs are refinements on a mechanism that
would not exist.

## 7. What would make us stop

Written before the runs, as S1's and S2's were.

- **S3a does not move the encoder-only floor beyond the seed spread.** Report it and stop; the
  query-derivation mechanism is wrong and rungs b and c refine nothing.
- **Render climbs while shape count collapses toward the traced baseline's 1.2.** That is the
  model finding the hole in the gate rather than doing the task. The style constraints exist to
  catch it, and if they are the only thing failing, that is the result.
- **Held-out-layer render passes while trained-element render does not.** Four layers is a thin
  sample; treat a pass there alone as unproven rather than as the gate.
- **Two seeds disagree beyond the S2 spread (0.0019).** S3 is a much larger change than S2, so
  a wide spread is a real possibility and means more seeds before anything is quoted.

## 8. Questions for the reviewer

1. **Is the gate amendment in §3 accepted?** A traced silhouette using 1.2 shapes scores 0.9697
   against a 0.95 bar on the population the gate reads. Without the amendment, S3 can pass
   without producing roto.
2. **Charter §7 vs plan §6 on differentiable rasterization (§5.1).** Which document gives?
3. **Is four held-out layers enough to gate on at all**, or should the S3 gate read held-out
   *shots* (the same four elements, differently framed) or wait for the Tier 2 expansion the
   tracker has queued behind S3?

---

### Appendix — the numbers this note rests on

`python scripts/exp_s3_baselines.py --dataset datasets/v003` → `runs/v2/s3_baselines.json`,
141 s, no GPU. 17 trained elements, 4 held.

| | value |
|---|---|
| S2 render, on-screen, trained | **0.9652** |
| same checkpoints, query rows freshly initialised | **0.0377** |
| `trace_k@64`, held-out layers | **0.9697** (gate: 0.95) |
| `trace_k@64`, trained elements | 0.9368 |
| `trace_k@64` shapes used | **1.2** (artist: 1 – 247) |
| burial, median over 5,178 live cells | 0.296 |
| burial, fraction below 0.05 | 16.5% |
| burial at lifespan boundaries (S2 measurement) | 61.4% |
| shapes per element | 1 – 247, median 16 |
| points per shape | 4 – 68, median 18 |
| keys per live frame | 0.021 – 1.184, median 0.474 |
