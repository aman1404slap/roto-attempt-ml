# S2 — lifespans and point counts: design note

**Status:** for review. Per [v2_implementation_plan.md](v2_implementation_plan.md) §5 and the
precedent S1 set, no S2 code is written until this is reviewed. Governed by
[v2_charter.md](v2_charter.md) §4 (the stage ladder), L1 (labels only in the loss), L2 (judge
the picture, not the handwriting), L3 (worst case, every element) and L4 (two seeds).

**Written after S1 closed.** S1 was stopped by a number that could have been measured before
the run: a constant function passed its acceptance gate. This note therefore spends its first
half on measurements taken *before* any S2 code exists, and its gate proposals are consequences
of those measurements rather than judgement calls. Everything below is reproducible with
`scripts/exp_s2_baselines.py`; the raw output is `runs/v2/s2_baselines.json`.

---

## 1. What S2 is

Two quantities move from the model's **input** to its **output**:

| | today (S0/S1) | at S2 |
|---|---|---|
| **lifespan** — which frames a shape is on screen for | given, as `ElementData.live` `(F, S)`; masks every loss term and decides each shape's key-search span | predicted, per `(shape, frame)` |
| **point count** — how many control points a shape has | given, as `desc[:, 0] = P / Pmax`, added straight into the query | predicted, per shape |

Still given: **shape count and identity**. Still teacher-forced: the layer transform track, unless
a run asks for `motion='predicted'`.

Charter §4's gate is **render within 0.01 of S1**, and plan §5 adds a loss shift — the curve term
becomes primary, dot-L1 falls to 0.25×, plus a point-economy style statistic.

## 2. What we measured before designing anything

`scripts/exp_s2_baselines.py` re-renders the **artist's own shapes** under a deliberately wrong
structure and scores them against the stored alpha with the dataset's own conventions. The model
is not in the picture, so every number below is a property of the *task*, not of any checkpoint.
The `exact` row reads 1.000000 (residue 2.5e-8, the uint16 quantiser the ledger already accounts
for), so the variants have a proven zero.

### 2.1 The gate is well posed. Unlike S1's, no trivial rule passes it

| trivial answer | on-screen soft IoU | cost | × the gate |
|---|---|---|---|
| every shape alive on every frame | 0.902947 | **0.0971** | 9.7× |
| alive iff *any* shape of the element is | 0.902947 | **0.0971** | 9.7× |
| one point dropped from every shape | 0.917425 | **0.0826** | 8.3× |
| a fifth of every shape's points dropped | 0.885710 | **0.1143** | 11.4× |

The second row is not a coincidence and it closes off a search: on the frames that count, "alive
iff the element is on screen" **is** "always alive", because those are the frames where something
is on screen. There is no cheaper trivial lifespan rule hiding behind the one we tested.

So S2's render gate is sensitive to both quantities by roughly an order of magnitude. **This is
the check S1's gate failed, and S2's passes.**

### 2.2 The gate is also very tight, and we can say exactly how tight

A lifespan head is wrong on some fraction of `(frame, shape)` cells. Flipping that fraction of
cells at random, one direction at a time, prices the accuracy it has to reach:

| lifespan cell accuracy | wrong by drawing (FP) | wrong by omitting (FN) |
|---|---|---|
| 99% | 0.0067 | **0.0215** |
| 97% | 0.0149 | **0.0621** |
| 90% | 0.0323 | **0.2367** |

The 0.01 budget is gone somewhere between 99% and 97% for intrusions, and **already gone at 99%
for omissions**. Reading the low-rate slope off: about 0.0033 soft IoU per 1% of cells drawn
wrongly, and about 0.022 per 1% omitted.

**And the realistic mistake is worse than the random one.** Random flips are one-frame flicker.
A head that has correctly learned *which* shapes come and go, and is a frame early or late on
*when*, produces something quite different: every lifespan boundary displaced by one. There are
**509 on/off transitions** across the trained elements — 1.27% of all cells — and moving all of
them costs:

| every boundary off by one frame | on-screen soft IoU | cost | worst element |
|---|---|---|---|
| one frame late off / early on (pure FP) | 0.989764 | **0.0102** | 0.9466 |
| one frame early off / late on (pure FN) | 0.962646 | **0.0374** | 0.7677 |

> **A lifespan head that gets every shape right and every boundary off by one frame has spent
> S2's entire render budget, or 3.7× it.**

That single line is the most consequential thing in this note. It says S2's decode is not
"threshold the logit and move on" — boundary *placement* is the work, and the hysteresis has to
be chosen on held-out frames the way the operating point was at Step 2c.

### 2.3 The two mistakes are not symmetric: omitting costs 3.2× drawing

At 1% of cells, omission costs 0.0215 against intrusion's 0.0067 — **3.2×**. At 10% the ratio is
**7.3×**. The mechanism is already half-written in the codebase: [train.py:305](src/roto/v2/train.py#L305)
notes that *a dead shape's control points are meaningless — the artist left them wherever they
last were*. Wherever they last were is usually still inside the union the live shapes are drawing,
so an extra shape often costs nearly nothing. An omitted live shape leaves a hole nothing else
fills.

**Consequence for the design:** the decode's two thresholds are asymmetric by measurement, not by
taste. Turning on should be cheap and turning off expensive.

### 2.4 Point count is memorisable, so accuracy is not its metric

The query is `shape_bank(shape_ids) + desc(desc)` — **one learned row per `(element, shape)`**,
constant across frames ([net.py:173](src/roto/v2/net.py#L173)). A point count is also constant
across frames. So the row can carry it exactly, and the picture need never be consulted. Removing
`n_points` from `desc` does not change this: the gradient of a point-count loss will simply put
it back into the embedding.

What the trivial predictors that *cannot* memorise score, over the 675 trained shapes:

| predictor | accuracy | mean abs error |
|---|---|---|
| global mode (P = 16) | 0.077 | 7.04 |
| per-element mode | 0.164 | 6.95 |
| **the per-(element, shape) query row** | **1.000** | **0** |

48 distinct counts over a 4–68 range, median 18, and the commonest value covers 7.7% of shapes —
so the distribution is genuinely flat, and a head reporting 0.95 accuracy has demonstrated
memorisation rather than perception.

The one honest reading available today is the **frozen-element column**: `untrained_queries`
already allocates fresh rows for elements the run never saw ([reconstruct.py:296](src/roto/v2/reconstruct.py#L296)),
so on those, a point count has to come from the encoder alone. It is a floor and not a claim —
the encoder alone currently reconstructs at 0.0631 soft IoU.

> **Proposal: at S2, point-count accuracy is reported and never gated. What is gated is the
> render.** The honest test of point-count skill needs S3's dynamic queries, and saying so now is
> cheaper than discovering it from a 0.99 accuracy that means nothing.

### 2.5 The damage concentrates by element, and the union is why

Charter L3's per-element worst-case reading is not a formality here; a frame-weighted mean hides
all of the following.

**Lifespan.** Four of the seventeen trained elements have every shape alive on every frame —
`always_alive` is *exactly right* on them and costs zero. The whole 0.0971 is carried by six:

| element | live rate | `always_alive` on-screen |
|---|---|---|
| `ts_021150__RED` | 0.093 | 0.3273 |
| `ts_021351__Alpha` | 0.286 | 0.5909 |
| `ts_021658__Layer_4` | 0.209 | 0.6280 |
| `ts_021351__Green` | 0.543 | 0.8303 |
| `ts_021351__Red` | 0.676 | 0.8515 |
| `ts_020028__char` | 0.216 | 0.8678 |

**Point count.** `drop_1` costs between 0.0004 and 0.053 on fifteen elements, and 0.4995 / 0.4192
on the two whose smallest shape has four points — a 4-point closed B-spline dropping to three is
near-degenerate. But `ts_020028__plant1` has 4-point shapes too and still reads 0.9472, because
78 overlapping shapes hide one broken one.

So both quantities are cheap to get wrong on a dense element and ruinous on a sparse one, in
proportion to how much the union has to hide. That is a statement about the metric as much as the
task, and it is the reason the per-element floor is the number S2 is read on.

### 2.6 Addendum, measured after §2.2: most lifespan boundaries are invisible

`scripts/exp_s2_visibility.py`. For every one of the 546 interior lifespan boundaries, render
the element's union with the shape and without it. Since the union is a per-pixel max, the
difference is exactly the area only that shape covers, and dividing by the shape's own ink gives
the fraction of it that nothing else is already drawing.

| visibility at a lifespan boundary | share of boundaries |
|---|---|
| p5 / p25 / **median** / p75 / p95 | 0.000 / 0.000 / **0.011** / 0.197 / 0.909 |
| **buried** (< 0.05 — switching it off changes almost no pixel) | **61.4%** |
| mostly hidden (< 0.20) | 75.3% |
| plainly visible (> 0.80) | 7.1% |

Per element it splits hard: `ts_021243__Red` (139 boundaries) and `ts_021351__Green` (207) sit at
a median of 0.000 and 0.013 — essentially every boundary invisible — while `ts_021150__RED` reads
0.844 and `ts_021351__Alpha` 0.968.

**This corrects §2.2 rather than confirming it.** That section flipped cells *uniformly at
random* and concluded the head needs ~99% cell accuracy. But a real head's mistakes will not be
uniform — they will concentrate exactly on the boundaries it cannot see, which are the same
boundaries that cost nothing to get wrong. A buried shape is undetectable **and free**: if
switching it off changes no pixel, drawing it anyway also changes no pixel.

So the two readings pull in opposite directions and both are true:

* **the ceiling is real** — no head reading a union alpha can place 61% of these boundaries, so
  a high cell accuracy is not available and should not be a gate;
* **the render may not care** — the uniform-flip estimate in §2.2 is therefore an *upper* bound
  on the cost, and possibly a loose one.

**The consequence for how S2b is read.** Cell accuracy is the wrong headline for this head: it
counts a buried cell that costs nothing the same as a visible one that costs everything. The
render is the honest judge, which is what charter L2 said all along. So the stop condition in
§7 — "accuracy below 99% means the gate cannot be met" — is **withdrawn as a prediction**; it was
derived from the uniform assumption this section overturns. The gate itself is unchanged, and
the render still decides.

## 3. What exists, and what has to be built

S1 was *enable, tune, measure* — nothing structural. **S2 is not.** This is the first v2 stage
that writes new heads.

| piece | where | state |
|---|---|---|
| per-`(shape, frame)` logit head, shaped exactly like the alive head needs | [net.py:153](src/roto/v2/net.py#L153) `key_logit` | exists for key timing; **the alive head is a second instance of it** |
| class-balanced BCE with per-element positive weighting, measured not tuned | [train.py:392](src/roto/v2/train.py#L392) `key_term`, `key_positive_rates` | exists for key timing; **reusable verbatim** |
| per-shape classifier over allowed point counts | — | **new** |
| lifespan consumed at decode (hysteresis) | [reconstruct.py:553](src/roto/v2/reconstruct.py#L553) reads `el.live` directly | **new** |
| point count consumed at decode | [reconstruct.py:554](src/roto/v2/reconstruct.py#L554) reads `shape.n_points` | **new** |
| `n_points` removed from the query input | [traindata.py:342](src/roto/v2/traindata.py#L342), [net.py:140](src/roto/v2/net.py#L140) | **new** |
| loss reweight (curve primary, dot-L1 0.25×) | [train.py:411](src/roto/v2/train.py#L411) `losses` | weights exist; the values change |

The two heads are cheap because the key-timing head already proved the shape of the work: a
per-frame logit off the query token, a class-balanced BCE restricted to the cells that mean
something, and a positive weight measured from the dataset rather than assumed. S1's `pos_weight`
lesson applies directly — plan §4's assumed rate was 4.5× wrong, and we will measure this one too.

## 4. Design decisions, and four deviations from plan §5

**4.1 — the loss reweight is its own rung, measured before the heads.** Charter §2 allows one
interface change per re-baseline; S2 as written changes four things at once, and if it fails
nothing says which. There is also a substantive reason to doubt the reweight *at S2*: shape count
and identity are still given, so index-wise correspondence is still exact and dot-L1 is still
legitimate — the reweight is preparation for S3's weakened correspondence, and at S2 it can only
cost geometry. So it runs first, alone, and if it costs more than the seed spread we keep S0's
weights for the head rungs and log the deviation.

**4.2 — the decode hysteresis is asymmetric, and the asymmetry is measured.** Plan §5 says
"hysteresis-threshold at decode" without a direction. §2.3 gives it one: a false negative costs
3.2× a false positive, so `t_on` sits low and `t_off` lower — a shape is easy to switch on and
hard to switch off. Both thresholds, plus a minimum run length, are swept on **held-out frames
only**, which is the rule that decided the key-value refit at Step 2c.

**4.3 — the point-economy style statistic is weighted, not uniform.** Plan §5 proposes
`|predicted count − archive count|` as a soft L1. That prices ±1 the same at P = 68 and at P = 4,
and §2.5 shows the render does not: invisible on the first, near-fatal on the second. Proposal:
scale the term by `1/P`, or floor it at P ≥ 4. The uniform form is kept as the measured control
rather than skipped.

**4.4 — every existing key metric keeps reading the artist's live span.** Key F1, key economy and
the tolerance range are all computed over each shape's live frames today. If they silently switch
to the *predicted* span, the S0 → S1 → S2 key columns stop being comparable and a lifespan
regression will read as a key-timing result. They stay on the artist's span; the predicted span is
reported beside them as its own column.

## 5. Proposed gate

Charter §4 says "render within 0.01 of S1". Two things have to be pinned before that is a
measurement:

- **which soft IoU.** Step 2c established the on-screen reading as the honest one, and lifespan
  is precisely the quantity that decides what gets drawn on an off-screen frame — scoring S2 on
  all-frames soft IoU would hand a lifespan mistake a free 1.0 exactly where it was made.
  **On-screen, trained elements, two seeds.** S1 reads 0.9712, so the gate is **≥ 0.9612**.
- **beside it, per charter L3:** worst element on-screen ≥ S1's 0.8644 − 0.01, and
  frames-below-0.90 not worse than S1's.

Reported and **not** gated: lifespan cell accuracy split into FP and FN rates, point-count
accuracy on trained and on frozen elements, and the key columns unchanged.

## 6. Plan of work

Three rungs, two seeds each, ~22 min a seed on the current GPU — **about 2.2 hours in total**, no
new data and no bigger instance.

1. **S2a — loss reweight alone.** Curve primary, dot-L1 0.25×, structure still given. Answers one
   question: what does plan §5's reweight cost geometry before any head exists?
2. **S2b — lifespan head.** Alive logit, class-balanced BCE on all `(frame, shape)` cells, decode
   through the hysteresis. Thresholds swept on held-out frames only. This is the rung the render
   gate is actually about.
3. **S2c — point-count head.** Classifier, `n_points` out of `desc`, decode by argmax. Reported
   against the frozen-element column per §2.4.

Each rung is scored on the frozen table, both seeds, and compared against S1's columns.

## 7. What would make us stop

Written before the runs, as S1's were, and for the same reason: there are always more settings to
try, and with enough attempts one of them will flatter us.

- **S2a costs more than the seed spread (0.0022 on-screen).** Keep S0's weights for S2b/S2c, log
  the deviation, continue. Do not tune the weights.
- ~~**Lifespan cell accuracy below 99%, or the FN rate above ~0.5%.**~~ **Withdrawn — see §2.6.**
  It was derived from flipping cells uniformly at random, and 61.4% of this dataset's lifespan
  boundaries turn out to be invisible in the union alpha, so a real head's errors concentrate on
  exactly the cells that cost nothing. Cell accuracy is therefore not a stop condition and not a
  gate; it is reported beside the render. **What survives from it:** run the threshold sweep
  **once**, on held-out frames, take its winner, and report. Do not sweep again to force a pass.
- **Point-count accuracy is high on trained elements and at the global-mode baseline (0.077) on
  frozen ones.** Memorisation confirmed; report it and claim nothing about point-count skill.
- **Geometry regresses beyond the seed spread.** The new terms are taking capacity from the point
  head; lower their weights once, then stop.
- **The held-frame gap widens beyond S0/S1's 0.0102.** S2 is fitting structure to frames it
  trained on.

## 8. Questions for the reviewer

1. **Is 0.01 still the right gate, now that off-by-one-frame-everywhere is measured at 0.0102?**
   The gate may be tighter than the task allows — it leaves a lifespan head essentially no margin
   for boundary error. Three options: keep it and expect a reported failure (the S1 outcome, which
   was still worth having); widen it to a measured quantity such as 2× the off-by-one cost; or
   gate lifespan on its own accuracy with render as a guard rail. **This is the main thing this
   review should decide.**
2. **Should point count be attempted at S2 at all?** Plan §5 says yes; §2.4 says the result will
   be unreadable while identity is given. Deferring it to S3 would make S2 purely about lifespan
   and make both stages cleaner. The counter-argument is that the head and its decode still have
   to be built and debugged somewhere.
3. **Does S1's gate amendment — grade over the best trivial baseline, not raw — apply here?** It
   was left open for S2/S3. §2.1 says the render gate already nets out the trivial answers by
   construction, so our reading is that it does not need to be applied again. Confirm.

---

### Appendix — the numbers this note rests on

Measured on `datasets/v003`, 17 trained elements, 675 shapes, 40,098 `(frame, shape)` cells.
`python scripts/exp_s2_baselines.py --dataset datasets/v003 --noise` → `runs/v2/s2_baselines.json`.

| | value |
|---|---|
| live cells | 25,137 / 40,098 (**0.627**) |
| live rate per element | 0.093 – 1.000 |
| shapes alive on every frame | 241 / 675 (**36%**) |
| elements where every shape is always alive | **4 / 17** (and 2 of the 4 held out) |
| on/off transitions | **509** (1.27% of cells) |
| "always alive" as a classifier | accuracy 0.627, F1 0.771 |
| "alive iff the element is" | accuracy 0.700 — and *identical* to always-alive on-screen |
| point counts | 675 shapes, **48 distinct**, range 4–68, median 18 |
| global-mode point count | P = 16, accuracy **0.077**, MAE 7.04 |
| per-element-mode point count | accuracy **0.164**, MAE 6.95 |
| artist's own shapes, re-rendered (`exact`) | **1.000000** (residue 2.5e-8, the uint16 quantiser) |
| render cost — always alive | **0.0971** |
| render cost — drop one point per shape | **0.0826** |
| render cost — drop a fifth of the points | **0.1143** |
| render cost — 1% of cells drawn wrongly | 0.0067 |
| render cost — 1% of cells omitted | **0.0215** |
| render cost — every boundary one frame late | **0.0102** |
| render cost — every boundary one frame early | **0.0374** |
| S1's on-screen soft IoU, the gate's reference | 0.9712 (spread 0.0022) |
