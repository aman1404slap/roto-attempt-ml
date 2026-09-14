# S1 — key-timing head: design note

**Status:** for review. Per [v2_implementation_plan.md](v2_implementation_plan.md) §4, no code
is written until this is reviewed. Governed by [v2_charter.md](v2_charter.md) §4 (the stage
ladder), L1 (labels only in the loss), L3 (every layer, worst case) and L4 (two seeds).

**Written after Step 2c**, which changed what this note has to argue. See
[v2-tracker.md](v2-tracker.md) → Step 2c.

![S1 design note at a glance](steps/s1-design-visual.jpg)

---

## 1. What S1 is

Today the keyframe stage is a dynamic-programming search: it spends a key wherever linear
interpolation would otherwise drift past a fixed tolerance. It is good at *economy* and
indifferent to *structure* — it has no notion that artists key turnarounds, contacts, and
changes of direction rather than evenly-spaced frames.

S1 adds a learned per-`(shape, frame)` keyness probability and lets it reshape that search
locally. The artist's keys supervise the **loss only**; they never enter the model's input, and
the DP still chooses the keys. That is charter L1, and it is why the head biases a tolerance
rather than emitting keys directly: over-keying is this project's measured signature failure, and
a bias on a minimising objective cannot cause it, because every extra knot still costs.

## 2. The gate has to change before the work starts

Charter §4 sets S1's gate at **key F1 ≥ 0.50 with keys ∈ [0.75, 1.3]×**. Two independent results
say that gate does not test what S1 is for.

**First — a dial already passes it.** Step 2c changed three settings on the DP (savgol filter,
0.2 px tolerance) and reached **key F1 0.7532 at 1.119× keys**. Both halves, cleared, with no head
and no training. Against a same-count random baseline, real timing skill moved only 0.0507 →
0.0720: **+0.313 of the +0.335 raw gain was density, not timing.**

**Second — a constant function passes it.** Measured on `datasets/v003`, firing on *every live
frame* scores **0.5804 unweighted key F1** across the 17 trained elements — above the 0.50 gate
before anything is learned. (Cell-weighted it is 0.4344; the unweighted per-layer reading is the
one charter L3 grades on.) The cause is key density: it ranges **47.9×** across these elements,
from `ts_020028__stick` at 0.019 to `ts_021150__RED` at 0.906. v1's probe found exactly this
trap one level up — a head that learned each layer's base rate and nothing inside any of them.

**Proposed amendment, for review:**

> S1 advances on **key F1 over the better of two trivial baselines** — firing on every live
> frame, and firing on a random subset of the same size — reported per layer and worst-case per
> charter L3, with the raw key F1 quoted beside it. The key-economy band [0.75, 1.3]× is
> unchanged. **Threshold: ≥ 0.15 over baseline**, against the 0.072 the tuned picker achieves
> today with no head at all.

0.15 is a proposal, not a derivation. It is roughly double what the picker alone reaches, which
is the smallest gain that would justify a learned component. **This number is the main thing this
review should push back on.**

## 3. What already exists

S1 is mostly *enable, tune and measure honestly* rather than build. Ported at Step 2 and inert:

| piece | where | state |
|---|---|---|
| head: query token → MLP → per-`(shape, frame)` logit | [net.py:130](src/roto/v2/net.py#L130) `key_head`, `key_logit` | built, off |
| BCE term, per-element positive weighting | [train.py:114](src/roto/v2/train.py#L114) `key_weight`, `key_balance`, `key_pos_weight` | built, weight 0 |
| picker integration, two-sided | [reconstruct.py:138](src/roto/v2/reconstruct.py#L138) `key_bias`, `key_slack` | built, both 0 |
| the head scored alone, and over both trivial baselines | `head_key_f1`, `baseline_key_f1_all_live`, `baseline_key_f1_random`, `head_key_f1_over_best_baseline` | built, zero without a head |

**New code required:** none structural. The work is a training run with `key_weight > 0`, a small
sweep of `(key_bias, key_slack)` on held-out frames only, and a report that reads the
over-baseline column instead of the raw one.

## 4. Three deviations from plan §4's reference design

**4.1 — `pos_weight` is measured, not 13.** Plan §4 gives `pos_weight ≈ (1-r)/r` with r ≈ 0.07.
On `datasets/v003` the live-cell key rate is **0.318**, so the correct value is **2.14**. Using
13 would over-weight positives by 6× and push the head toward firing everywhere — the exact
failure mode §2 says the gate cannot detect. `key_pos_weight=0` already means *measure it from
the dataset*; that is what we will use, and the measured value goes in the run record.

**4.2 — per-element balancing is mandatory, not optional.** At a 47.9× spread, a single global
positive weight is exploitable: a head that predicts each layer's base rate and nothing else
scores well. `key_balance='per_element'` is the default and stays.

**4.3 — the picker bias is two-sided, against the plan's one-sided form.** Plan §4 proposes
`tol_f = tol_base · (1 − a·p_key)` — tightening only. Tightening can only *add* keys, and after
Step 2c we sit at **1.119× against a 1.3× ceiling**: there is almost no room to add. Precision is
also the binding constraint (this project's measured failure is over-keying at precision
0.21–0.44 against recall 0.75–1.00). So we use `key_bias` **and** `key_slack` together, which
lets the head *move* keys without changing how many there are. The one-sided form is measured as
a control, not skipped.

## 5. Plan of work

1. One training run, `key_weight` at the value where the key term is comparable in magnitude to
   the point term, everything else identical to the S0 anchor. **One interface change, one
   re-baseline** (charter §2).
2. Sweep `(key_bias, key_slack)` on **held-out frames only** (plan §4, and the rule that decided
   the refit question at Step 2c).
3. Two seeds on the winner (charter L4). Report the head alone, the pipeline, and both trivial
   baselines, per layer.
4. Compare against the **no-head** column at the same operating point: 0.7532 raw / 0.0720 over
   random / 1.119× keys.

Estimated cost: ~22 min per seed on the current laptop GPU, as S0. **No new data and no bigger
instance** — those come after S3 (see tracker, Step 3).

## 6. What would make us stop

- The head's over-baseline figure does not beat the picker's 0.072 → the head is learning base
  rates. **Report it and move to S2 rather than tuning until it passes.**
- Key economy leaves [0.75, 1.3]× and cannot be brought back by `key_slack` → the bias form is
  wrong; fall back to the one-sided control.
- Soft IoU or point error regresses beyond the seed spread → the key term is stealing capacity
  from geometry; lower `key_weight` once, then stop.
- The held-frame gap widens beyond 0.0102 → S1 is fitting key timing to frames it trained on.

## 7. Questions for the reviewer

1. **Is ≥ 0.15 over baseline the right bar?** It is a judgement call, not a measurement.
2. **Is amending a charter gate acceptable here?** Charter §7 reopens a frozen decision only on a
   gate failure that names it. This is the inverse — a gate passing for the wrong reason. We
   propose the design note review is the right venue; if not, this needs an explicit charter
   amendment before S1 starts.
3. **Should the raw key F1 stay in the gate at all**, or only as reported context? Keeping both
   means a head could fail on raw while passing over-baseline, and we have not decided which way
   that should resolve.

---

### Appendix — the numbers this note rests on

Measured on `datasets/v003`, 17 trained elements, checkpoints `runs/v2/s0_seed{1,2}`.

| | value |
|---|---|
| live `(shape, frame)` cells | 34,661 |
| artist keys on live cells | 10,574 |
| key rate, cell-weighted | 0.3184 |
| key rate spread across elements | 0.0189 – 0.9062 (**47.9×**) |
| `pos_weight = (1−r)/r` at that rate | **2.14** |
| "fire on every live frame" key F1, unweighted | **0.5804** |
| "fire on every live frame" key F1, cell-weighted | 0.4344 |
| picker today: raw key F1 / over random / keys | 0.7532 / **0.0720** / 1.119× |
| seed spread on over-random | 0.0005 |
