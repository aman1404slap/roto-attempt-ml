# v2 — tracker

Single source of truth for where v2 is and what happens next. Governed by
[v2_charter.md](v2_charter.md) (the rules) and
[v2_implementation_plan.md](v2_implementation_plan.md) (the build order).

**How to use this file.** Read "Now" to see the current step. Each step carries its goal, the
charter section it answers to, and the gate that says it is done. When something changes mid-flight
— a gate moves, an assumption breaks, a shot drops out — write it in the Deviation log rather than
editing history. Work can stop and restart at any step boundary without losing the thread.

---

## Now

| | |
|---|---|
| **Step** | **S3a run, 2 seeds. Mechanism works; does not pass. The failure names the data.** |
| **Status** | Steps 1, 2a, 2b, 2c complete. S1 closed, does not pass. **S2 complete, all three rungs pass.** S3's gate priced before the design, per the S1 and S2 lesson. |
| **Next action** | **Decision needed.** S3a's stop condition did not fire (floor moved 0.0377 → 0.2031), so S3b/S3c are formally open — but the failure is generalisation *across shots*, which neither rung addresses. Recommendation: bring the **Tier 2 expansion forward** and re-run S3a on it, per charter §7 (a gate failure that names a scope decision reopens it) |
| **Open question that blocks** | **yes — one.** Whether to spend ~2 h on S3b/S3c as planned, or reorder the Tier 2 expansion ahead of them. Evidence for reordering: held-*frame* gap 0.0027 (perfect within a shot) against 0.2031 on an unseen shot — the signature of too few shots, not of a wrong loss. Earlier answers, all **yes**, 2026-09-14: gate amendment accepted (render over the traced baseline + style statistics as hard constraints); differentiable low-res raster is in scope for S3, charter §7 giving way to plan §6; four held-out layers accepted as the population, with the thin sample flagged and Tier 2 staying queued behind S3 |
| **Last updated** | 2026-09-14 |

**S0 anchor, at the shipping operating point.** 2 seeds × 40k steps, 21.7 min each. Trained
elements: soft IoU **0.9768** all-frames / **0.9717 on-screen** (spread 0.0022), worst element
**0.8644** on-screen, point error **0.830 px** mean / 2.478 px p95, key ratio **1.119×**, key F1
0.7532, held-frame gap 0.0102. **Passes soft IoU mean, both point-error gates and key economy;
misses worst-element, frames-below-0.90 and the held-frame gap.** Full write-ups in
`steps/step2-S0-faq.md` (the anchor) and `steps/step2c-faq.md` (the operating point and its
control).

**S1 result.** 2 seeds × 40k steps, 22.5 min each. Key F1 over the same-count random baseline:
**0.0720 (no head) → 0.0914 (head consulted)**, spread 0.0032 — real, and roughly a quarter of
the ≥0.15 bar the design note proposed. The head **scored alone is negative against its own best
trivial baseline** (−0.021, −0.012 on the two seeds): it is worse than "fire on every live frame".
Geometry untouched (on-screen soft IoU 0.9717 → 0.9712, inside the 0.0022 spread); key economy
improved 1.119× → 1.044×. Winning bias/slack 0.25/0.5, selected on held-out frames by a rule
written before the results. **The design note's first stop condition fired; we report and move to
S2 rather than tuning until it passes.** Write-ups: `steps/`, and plain-English in
`luthra-understands/`.

**S3a write-ups:** `luthra-understands/s3.md` (what S3 is) and `luthra-understands/s3-attempts.md`
(the diary, two cards per attempt -- trained layers and unseen shots say different things here).

**S3a result. 2 seeds × 40k, 1 h 2 min. The mechanism works and it does not pass.** The
per-element query table is replaced by **one bank of 256 slots shared across every element**,
addressed by canonical order (transform group, then centroid). On **trained** elements: on-screen
soft IoU **0.9385** (spread 0.0004) against S2C's 0.9652 — so removing identity costs 0.0267 on
data it has seen. Point error 1.239 px. **The held-frame gap collapses 0.0108 → 0.0027**:
removing the memorisation capacity removed the memorising, which is the cleanest confirmation the
query table was what it was always claimed to be. On the **four held-out shots**, now scored with
the real model for the first time rather than with stand-in rows: **0.2031** (spread 0.0101),
against the **0.9697** a hand-traced silhouette scores there. It fails the amended gate by a wide
margin. Two style constraints also fail: key ratio 1.296× (1.345× before the decode sweep) and
point count 4.0% exact on unseen shots, below the 7.7% trivial baseline.

**What the failure names.** Held-out *frames* of a known shot: gap 0.0027, essentially perfect.
An unseen *shot*: 0.2031. It generalises within a shot and not across shots, which is the
signature of **too few shots** — eight trained — rather than of a wrong loss or a wrong matching
rule. Plan §6's remaining rungs (S3b's loss shift, S3c's Hungarian matching) address neither.

**A reporting bug S3a found, and how.** Scoring first read 0.7998 on-screen and 16.3 px while the
training log read 1.09 px. With shared slots, *withheld from training* and *scored with fresh
query rows* stop being the same fact, and the table was still splitting on the second — four
untrained elements were being averaged into the headline. Caught only because two independent
measurements disagreed; pinned by a regression test.

**S3 pre-design measurement. The gate does not survive it.** Charter §4 sets S3's gate at soft
IoU ≥ 0.95 on held-out layers, and soft IoU is decomposition-blind by construction. Tracing the
artist's own alpha with `cv2.findContours`, resampling to 64 points and rebuilding it as real IR
shapes scores **0.9697 on held-out layers** — above the bar — using **1.2 shapes** where the
artist used 1 to 247. On trained elements the same trace reads 0.9368. **The amendment proposed:
grade S3 on render *over the traced baseline*, with style statistics (shape count, points per
shape, key economy) as hard constraints rather than soft targets.** A second measurement reversed
a position the note was going to argue: over 5,178 sampled live cells the median shape contributes
**0.296** of its own ink uniquely and only 16.5% are buried — so shapes are substantially visible
while on screen, and plan §6's chamfer term keeps its place. (At lifespan *boundaries* the figure
is 61.4% buried; different question, both true.) `scripts/exp_s3_baselines.py` →
`runs/v2/s3_baselines.json`; note in [v2-s3-design-note.md](v2-s3-design-note.md).

**The number that frames S3.** S2 reconstructs at 0.9652 with its query table and **0.0377** with
the query rows freshly initialised. The table carries ~96% of the result, and replacing it is what
S3 is. S3 is not the next rung; it is the rest of the problem.

**S2 result. Three rungs, 2 seeds each, 2 h 30 min of GPU, zero retries. The gate passes.**
On-screen soft IoU **0.9713 (S1) → 0.9652 (S2C)**, a drop of 0.0061 against a 0.01 budget, with
**lifespans and point counts both predicted rather than given**. Lifespan cell accuracy **0.9941**
(FP 0.0040, FN 0.0019), **0.9828 on held-out frames** against a 0.627 always-alive baseline, and
transitions at **1.015x** the artist's. Point count is **1.000 on trained elements and 0.050 on
held-out shots** against a 0.077 global-mode baseline -- memorisation, exactly as the design note
predicted, reported and gated on nothing. Rung table: `runs/v2/s2{a,b,c}_table.txt`. Write-ups in
`luthra-understands/s2.md` (what it is), `luthra-understands/s2-attempts.md` (every attempt, its
cost and its picture) and `steps/s2-faq.md`.

**Two S2 predictions were wrong and are corrected in the design note rather than deleted.**
(1) We predicted the lifespan head would fail, from a uniform-error calculation; 61.4% of this
dataset's lifespan boundaries are invisible in the union alpha, so a real head's errors land
exactly where they are free. The stop condition derived from it is struck through and marked
withdrawn. (2) We expected the decode thresholds to be most of whether S2b passed; all 16 swept
cells landed within 0.003 of each other.

**S2 pre-design measurement.** The S1 lesson applied before the run: price the gate against the
things that require no skill. The artist's own shapes, re-rendered with the structure deliberately
broken, on the 17 trained elements. **The gate is sound** — every trivial answer costs 8–11× it:
always-alive **0.0971**, drop one point per shape **0.0826**, drop a fifth of the points
**0.1143**. And on-screen, "alive iff the element is on screen" *is* always-alive, so there is no
cheaper trivial rule left to find. **The gate is also very tight**: a lifespan head at 99% cell
accuracy costs 0.0067 if it draws wrongly and **0.0215** if it omits, and moving every one of the
509 on/off boundaries by a single frame costs **0.0102** late / **0.0374** early. Omission costs
3.2× intrusion, because a dead shape's points sit where the artist left them — usually inside the
union — while an omitted live shape leaves a hole. Point count is **memorisable**: the query row is
per `(element, shape)` and a point count never varies with the frame, so accuracy there is 1.000 by
construction against 0.077 for the global mode. Measured by `scripts/exp_s2_baselines.py` →
`runs/v2/s2_baselines.json`; write-ups in [v2-s2-design-note.md](v2-s2-design-note.md) and
`steps/s2-design-faq.md`.

**Step 1 result.** `datasets/v003`: 10 shots, 21 elements, 875 shapes, 1,412 element-frames;
17 trainable, 4 held (2 held-out shots), 198 frames held. Ledger **7/7 green**, with the named
gate — the artist's own shapes re-rendered against the stored alpha — at **exactly 1.000000000,
max abs pixel 0.00e+00**. Our renderer reproduces these artists' delivered mattes at
**0.94–0.996** on 8 of the 10 gold pairings. Build with `python -m roto.v2 dataset datasets/v003`;
check with `python -m roto.v2 ledger datasets/v003` (exits non-zero on RED).

---

## What v2 is, in one paragraph

An artist drew editable curves; those curves were rendered into a black-and-white cutout; we train
a model to look at the cutout and recover the curves. It is hard because a layer's cutout is the
**union** of many overlapping shapes — looking at the filled silhouette you cannot see where one
shape ends and the next begins, because the union destroyed that information. v2's plan is
therefore not "get a better number"; it is to remove, one at a time, the structural hints the model
is currently handed, changing what the loss rewards in lockstep (charter §4).

---

## Decisions already made

These four were open going in. All four are answered by the charter, not by us. **Frozen** — per
charter §7, reopening one requires a gate failure that names it.

| # | Question | Decision | Authority |
|---|---|---|---|
| D1 | Train on delivered mattes, or our own render? | **Our own render from the `.sfx`.** Delivered EXRs are a referee and a tag, never the input. | charter §3, §6; plan §1 |
| D2 | `hard/` or `mb/` mattes? | **Neither is the input.** Motion blur is out of scope until S3 gates pass. | charter §7 |
| D3 | Is the plate (`.mov`) an input? | **No.** All 50 `.mov` files stay unused in v2. Route to real-world robustness is corrupting clean alphas, not the plate. | charter §1 (L1), §4, §6 |
| D4 | Which layers train? | **All of them**, tagged `gold` (EXR-matched) or `silver` (tree-defined). Nothing excluded for being small — §5 grades every layer individually. | charter §6 |

**Why D1 is load-bearing.** If we train on the artist's delivered matte, the model is asked to
reproduce pixels our renderer cannot draw — their motion blur, edge softness, filters. A *perfect*
answer would then score below 1.0, and we could never separate "the model is wrong" from "our
renderer is wrong". Rendering the target ourselves makes input and answer agree exactly, so a
perfect answer scores exactly `1.000000` and every point below it is the model's. This is charter
L3 (zero margin) and it is what the Step 1 gate measures.

---

## The data: `data/spline_dataset_08_25_26`

Surveyed 2026-09-11. 50 shots, 11 GB.

| | |
|---|---|
| `.sfx` files | 483 — **30 named, 453 Silhouette autosaves** |
| parse cleanly with `roto.sfx.read` | 470 / 483 |
| layers (`.sfx` roots) | 145 |
| shapes | 27,120 |
| frames | 7,592 |
| shapes per layer | min 1, median 17, p90 247, **max 9,008** |
| dialects | 47× `5/plain`, 3× `2020/zlib`, **2× `6/plain` (new)** |

### Facts that will bite if forgotten

- **One `.sfx` per shot.** The other ~9 per shot are autosaves (`backup.sfx`, `backup.1..9.sfx`,
  `project.1..9.sfx`, and `autosave.sfx` in `ts_020769` / `ts_021097`).
- **Picking the final one cannot use mtime or filename.** All mtimes are identical (`2026-09-01
  10:34` — the copy flattened them), there is **no timestamp inside the `.sfx`**, and filenames lie
  (`ts_020028`'s long descriptive name is a 77 KB early save against `project.sfx` at 747 KB).
- **`shots._find_sfx` is wrong for this delivery.** Its rule is largest-file-wins, which lands on a
  mid-session autosave in roughly 30 of 50 shots (`ts_021262`: `backup.4.sfx` 7.7 MB vs
  `project.sfx` 5.1 MB, where the artist later deleted shapes).
  **Rule to implement:** `project.sfx` first (exists in 47/50, parses in 46) → else score candidates against the
  delivered EXR channels with `exr.best_channel` → else max shape count.
- **Each EXR packs three different mattes in RGB**, not a replicated alpha
  ([exr.py:39](src/roto/exr.py#L39)). `ts_020169`'s B channel is empty while the others carry content.
- **`.mov` plates are anamorphically desqueezed** (2× on `ts_019698`, 1.3× on `ts_020028`) and
  `metadata.json` wrongly claims `PixelAspectRatio: 1.0`. Irrelevant while D3 holds; a trap if it
  is ever reopened.
- **Dialect `6/plain` is unknown** to `DIALECT_CONTAINER` in [write.py:73](src/roto/sfx/write.py#L73).
  It defaults to plain, which is correct — but the round-trip ledger covers four dialects and this
  is a fifth.

### Excluded outright (4)

| shot | reason |
|---|---|
| `ts_021734` | final `.sfx` unreadable — a shape changes point count across keyframes, which the IR does not model. Dropped per charter §2/§7 rather than extending a frozen interface. |
| `ts_021030` | zero EXRs — no referee, no gold tag |
| `ts_019120` | 1 shape total, no EXRs, no `project.sfx` |
| `ts_019663` | no `project.sfx`; EXRs will not decode in this OpenCV build |

### Deferred (3) — real data, wrong size to start

`ts_021191` (9,077 shapes), `ts_021733` (one layer of 9,008), `ts_020966` (8K × 393 frames).
These are where the scaling problems live. They would dominate a first run.

### Tier 1 — the starting subset

**10 shots, 21 layers, 875 shapes, 1,412 element-frames.** (674 is the sum of *shot* durations;
a 5-layer shot contributes its 94 frames five times.) Every one has a readable `project.sfx` and
delivered `hard` mattes matching frame-for-frame. Chosen to span 7 → 247 shapes per layer so a
failure says *where* it breaks, not just that it broke.

| shot | frames | layers | shapes per layer |
|---|---|---|---|
| `ts_021658` | 58 | 1 | 7 |
| `ts_020036` | 104 | 1 | 13 |
| `ts_021555` | 71 | 1 | 20 |
| `ts_020355` | 63 | 2 | 20, 5 |
| `ts_021150` | 64 | 1 | 27 |
| `ts_020876` | 71 | 2 | 39, 10 |
| `ts_020028` | 94 | 5 | 78, 19, 15, 9, 1 |
| `ts_021243` | 56 | 1 | 139 |
| `ts_021182` | 51 | 3 | 175, 4, 1 |
| `ts_021351` | 42 | 4 | 247, 27, 16, 3 |

**Held-out shots:** `ts_021555` and `ts_021182`, withheld entirely, so there is a generalisation
number from day one.

### Tier 2 — expand after Step 2 is clean

`ts_019698` (643), `ts_020530` (1,235), `ts_020880` (563), `ts_020191` (436), `ts_021994` (415).

---

## Step 1 — build the ground. No training.

**Goal.** Answer one question: **does our renderer reproduce these artists' mattes?** It was
verified on 6 archive shots; on 50 new ones from different artists it is unproven, and it is the
assumption every later step rests on.

**Doc reference.** charter §3 (dataset rebuild), §2 (freeze the interfaces), plan §1.

**Work**

- [ ] Implement the `.sfx` pick rule (`project.sfx` → `best_channel` → max shapes) and fix
      `shots._find_sfx`, whose largest-file rule is wrong here
- [ ] Ingest layout adapter: `<shot>/splines/silhouette/*.sfx`, `<shot>/mattes/hard/*/`, RGB = three
      separate mattes
- [ ] Pairing QC over Tier 1 — tag each element `gold` / `silver` (charter §6)
- [ ] Build `datasets/v003` from Tier 1 under the charter §3 render law
- [ ] Record splits at build time: every-7th-frame holdout, plus held-out shots `ts_021555`,
      `ts_021182`
- [ ] Run `scripts/ledger.py datasets/v003`

**Gate.** Ledger all green, **including `artist-self-score == 1.000000` on every layer, every
frame** (plan §1).

**If the gate fails**, stop and report. A miss means our render law and these artists' conventions
disagree, and no training number would mean anything until that is resolved. This is the only step
whose outcome is genuinely unknown, which is why it is first and why it commits us to nothing.

**Result: PASSED.** 7/7 green.

| row | measured |
|---|---|
| artist shapes render back to the stored alpha | **1.000000000**, max abs pixel 0.00e+00 |
| stored alpha survives its own uint16 round trip | 0.000e+00 |
| render conventions travel with the data | `measured` set, 21 elements agree |
| every declared frame is on disk | complete |
| the dataset declares what a run may train on | 17 trained, 4 held, 198 frames held |
| `read(write(doc))` is bit-exact | 10/10 |
| pairing QC gates every ingested shot | 10/10 pass; 10 gold, 11 silver, 0.6590–0.9956 |

The question Step 1 existed to answer is answered: **our renderer and these artists' renderer
agree.** Where the pairing is clean, agreement with the delivered EXRs is 0.94–0.996, and the
artist's own shapes reproduce the stored alpha bit-exactly.

---

## Step 2 — one baseline on the subset

Split into three sub-steps so the work can stop between them. **2c is explicitly not happening
now**: the instruction is a baseline and a path forward, not a tuned number.

### 2a — v2 owns its model  ✅ done

**Doc reference.** charter §4 stage S0.

v2's network, data path, losses, trainer, reconstruction and scorer live in `roto/v2/`, ported
from `roto.model` with attribution and importing nothing from it. The boundary test still passes,
so v1 stays deletable.

Two modules were reclassified as **shared** rather than v1, because they are exactness machinery
rather than model decisions:

- `roto.program` — IR ↔ dense arrays; its round trip is a ledger row
- **`roto.model.geometry` → `roto.geometry`** — closed-form local↔crop math, pinned by
  `tests/test_geometry.py`. Filing it under `model/` made a shared guarantee look like a v1
  asset. Moved rather than copied, so there is one implementation and one test.

**S0's configuration is v1.1's `HEADLINE`, unchanged** — 3-frame aligned window, self-attention
among shape queries, curve 0.5, temporal 1.0, sqrt element weighting, 8-DOF projective transform
head at 0.25. Deliberately not retuned: Step 2 produces an *anchor*, and an anchor measured with
a term nobody has measured before anchors nothing. `sample_weight` stays `sqrt` even though
v1.3's gates named it the next rung — moving the anchor and the rung together leaves neither
readable.

Two v1-comparability options were dropped rather than carried, both with nothing left to be
comparable to: `affine_space='doc'` (the lossy 6-number target) and `use_split=False`.

### 2b — the baseline table  ⏳ running

**Doc reference.** charter §5 (gates), L4 (two seeds), plan §2 (the table's frozen format).

- [x] trainer verified end to end: point error 82.7 → 17.9 px in 1,500 steps
- [x] scorer verified end to end on a throwaway checkpoint
- [x] `s0_seed1`, `s0_seed2` at 40k steps each — 21.7 min per seed, 462 MB, GPU ~50%
- [x] frozen table, three blocks: trained / held-out elements / held-out shots
- [x] `steps/step2-S0-faq.md`

**Gate.** None. This step *is* the anchor. Per charter L4, quote nothing from a single seed and
treat any delta below the two-seed spread (0.0008) as weather.

### What 2b found, beyond the number

**A hypothesis tested and corrected.** After the first per-element look I said the keyframe stage
was the bottleneck. Sweeping the tolerance 1.0 → 0.2 px says that is only half right: key economy
moves 0.34× → **0.73×** and key F1 0.418 → **0.635** (above anything recorded in this project),
but soft IoU moves only **+0.003**, and the three worst elements barely move (0.840 → 0.845). The
keyframe stage is the bottleneck for *key economy*, not for soft IoU.

**And 2c corrected that a second time.** The sentence above is true of the *tolerance*, which is
the dial that was swept, and false of the stage. Changing the smoothing filter as well moved
on-screen soft IoU 0.9483 → 0.9717 — the single largest gain in v2 so far, from a stage that does
not learn. The correct statement is narrower than either version: **tolerance alone buys key
economy; the filter buys the picture.**

**18% of scored frames are free marks.** 215 of 1,188 frames have almost nothing on screen; the
artist drew nothing, the model drew nothing, and `soft_iou` returns exactly 1.0. That is
defensible per frame and misleading in aggregate — it inflates elements in proportion to how
often their layer is absent, and it makes the per-element floor and the frames-below-0.90 gate
non-comparable across elements.

| | all frames | on-screen only | gap |
|---|---|---|---|
| seed 1 | 0.9572 | 0.9478 | +0.0094 |
| seed 2 | 0.9581 | 0.9488 | +0.0093 |

The gap is **twelve times the seed spread**, so it is a real effect, not noise. Per element it
reaches 0.846 → 0.660 (`ts_021351__Alpha`, 23 of 42 frames absent). This is the same
`metrics.soft_iou` empty-versus-empty convention that Step 1 caught in the pairing referee —
correct for scoring a reconstruction, misleading when averaged over frames where there is nothing
to reconstruct.

**Three distinct causes sit under the aggregate**, which is why one fix was never going to close
it: the free frames above; two elements with genuinely poor geometry (`ts_020876__Red` 2.24 px,
`ts_021150__RED` 1.41 px against a 0.83 px mean); and small layers where the metric is merciless —
`ts_021351__Green` has the **best point error in the run** (0.43 px) and still scores 0.908.

### 2c — the operating point, adopted  ✅ complete

**Doc reference.** charter §5 (gates), L4 (rank on frames-below-0.90), plan §4 (tune on held-out
frames only). Write-up and figure: `steps/step2c-faq.md`, `steps/step2c-visual.jpg`.

- [x] on-screen-only soft IoU reported beside the headline (reporting change, no retraining)
- [x] operating point swept, 20 cells, both seeds on the winner
- [x] **savgol + `tol_px` 0.2 + no refit wired as `RebuildConfig`'s defaults**, with the
      measurement written into the docstring
- [x] anchor re-scored at the new default; `runs/v2/s0_table.txt` and `s0_summary.json` regenerated
- [x] the **same-count random control** run on the key-F1 gain — the reason S1 did not start here
- [x] `steps/step2c-faq.md` + `steps/step2c-visual.jpg`
- [x] full suite green, 182 passed

**What it bought.** Gates 2 of 7 → **4 of 7**. On-screen soft IoU 0.9483 → **0.9717**; worst
element 0.6594 → 0.8644; frames below 0.90 149 → 39; key economy 0.341× → **1.119×**, inside its
band for the first time. Every one of the 17 trained elements improved. Point error is unchanged
to three decimals, which is correct — the model is untouched.

**What it cost, stated plainly.** The held-frame gap widened 0.0028 → **0.0102**. No operating
point measured, v1's default included, meets the ≤ 0.002 gate. Taken because six of seven gates
improve or hold and the seventh was already failing; logged below.

**The key-value refit was measured and refused.** It wins charter L4's ranking metric (frames
below 0.90: 49 → 28) and loses plan §4's generalisation check (held gap 0.0102 → 0.0260). Two
governing documents, opposite answers; the held-out evidence decided it.

**The control that changes S1.** Key F1 rose 0.4185 → 0.7532. A same-count random baseline rose
0.3681 → 0.6812. Timing skill over chance: **0.0507 → 0.0720** (seed spread 0.0005). The raw gain
is 93% density. Recorded as a deviation, and carried into the S1 design note as a gate amendment.

**Still on the table, unchanged, one flag each:**

- `sample_weight`, the rung v1.3's failing gates named
- predicted-motion scoring (`motion='predicted'`) — the charter's de-teacher-forced number

## Step 3 — expand, then remove training wheels

**Goal.** Only once Steps 1 and 2 are clean: widen the data, then start charter §4's ladder.

**Doc reference.** charter §4 (S1 → S2 → S3), §5 (per-stage gates), plan §4–6.

**Work, in order**

- [x] S1 design note written — [v2-s1-design-note.md](v2-s1-design-note.md), with
      `steps/s1-design-faq.md` and `steps/s1-design-visual.jpg`
- [x] S1 design note's three questions — **answered by the run rather than in the abstract**:
      S1 misses both the old gate reading and the proposed one, so the amendment did not have to
      be settled to reach a verdict. It remains open for S2/S3, where key timing is still graded
- [x] **S1 — key-timing head. Run, 2 seeds. DOES NOT PASS** — see the deviation log
- [x] **S2 design note written** — [v2-s2-design-note.md](v2-s2-design-note.md), with
      `steps/s2-design-faq.md`. Gate priced *before* the design, not after: sound, and tighter
      than the task may allow (§8 Q1 is the one open decision)
- [x] **S2 — lifespans and point counts predicted. PASSES.** Three rungs, two seeds each:
      **S2a** loss reweight alone (free: −0.0018 render against a 0.0022 spread), **S2b**
      lifespan head (0.9672, gate ≥ 0.9613), **S2c** point-count head (0.9652). 2 h 30 min,
      zero retries. `scripts/train_v2.py`, `scripts/score_v2.py`,
      `scripts/sweep_s2_lifespan.py`, `scripts/fig_s2_attempt.py`
- [x] **S3 design note written** — [v2-s3-design-note.md](v2-s3-design-note.md). Gate priced
      before the design and **it falls to a traced silhouette**; three questions escalated
- [x] **S3a — queries from the encoder. Run, 2 seeds. Works; does not pass.** 0.9385 trained,
      **0.2031 on unseen shots** against a 0.9697 traced baseline. Held-frame gap 0.0027
- [ ] **Decision: S3b/S3c as planned, or Tier 2 first?** S3a's failure is cross-shot
      generalisation, which neither remaining rung addresses
- [ ] S3b — loss moves to render + chamfer + style (plan §6)
- [ ] S3c — Hungarian as the measured alternative to canonical ordering
- [ ] Expand to Tier 2 and re-anchor (charter §2: one interface change, one re-baseline)

**Reordered 2026-09-11 — the S-series runs on Tier 1, and the data widens after it.** The list
above originally put the Tier 2 expansion first. Two reasons it moved to last: charter §2 allows
one interface change per re-baseline, and changing the model and the dataset in the same move
means neither is measured; and the infra decision depends on it — the S0 run used 462 MB of
8,151 MB at ~50% GPU utilisation, so nothing needs a bigger instance until the data grows.
Logged below.

**Gates.** Per stage, from charter §4: S1 → key F1 ≥ 0.50 at keys ∈ [0.75, 1.3]×; S2 → render
within 0.01 of S1; S3 → soft IoU ≥ 0.95 on held-out layers.

**S1's gate needs amending before S1 starts.** Step 2c cleared both halves of it — 0.753 at
1.119× — with a dial, no head. The amendment to put in the design note: grade S1 on **key F1
over the same-count random baseline**, raw figure reported beside it, key-economy band unchanged.
Per charter §7 a frozen decision reopens only when a gate failure names it; this is the inverse —
a gate *passing* for the wrong reason — so it goes through the design note review rather than
being changed unilaterally.

**What the S1 note found, in one line.** S1 is almost entirely *already built* — head, BCE term,
per-layer weighting, two-sided picker bias and the over-baseline metrics were all ported at Step 2
and sit switched off. The work is one training run, one sweep on held-out frames, two seeds. It
also names three deviations from plan §4's reference design: `pos_weight` is **2.14** measured on
v003, not the plan's ~13 from an assumed 7% key rate (ours is 31.8%); per-element balancing is
mandatory at a 47.9× density spread; and the picker bias is two-sided, because tightening alone
can only add keys and we sit at 1.119× against a 1.3× ceiling.

**Status:** S1 design note written and awaiting review. No S1 code written.

---

## Deviation log

Anything that departs from the charter or the plan, with its reason. Append; do not rewrite.

| date | what changed | why |
|---|---|---|
| 2026-09-11 | Dataset is `v003` from `data/spline_dataset_08_25_26`, not plan §1's `v002` from the 6-shot archive | New 50-shot delivery supersedes the archive subset as the POC ground |
| 2026-09-11 | Holdout is **shots**, not plan §1's named holdout *layers* | 50 shots make a held-out-shot split available immediately; charter §5 says generalisation is claimed only from held-out layers, then shots |
| 2026-09-11 | 4 shots excluded, 3 deferred, 10 of 50 used at Step 1 | Charter §6 requires pairing QC to gate every ingested shot; the exclusions fail it, and the deferrals are a size decision, not a quality one |
| 2026-09-11 | v2 lives in `src/roto/v2/`, importing only the frozen core and three mechanical modules; `tests/test_v2_boundary.py` enforces it | Requested clean separation. v1 is now deletable — the test fails if an import re-couples them |
| 2026-09-11 | `roto/dataset/__init__.py` re-exports lazily (PEP 562) | Eager imports made `import roto.dataset.crop` pull in `roto.shots` via `build`, coupling v2 to a v1 module. Public API unchanged |
| 2026-09-11 | Pairing QC fails a shot **only** on an *ambiguous* collision, not on orphans or unmatched elements | Gating on orphans dropped 4 of Tier 1's 10 over facts about the delivery, not the roto. See "What Step 1 found" |
| 2026-09-11 | The scoring-ceiling row compares **through the uint16 quantiser** | Raw float vs stored uint16 can never reach 1.0; the residue is the quantiser, measured at exactly half a step. Quantising both keeps the requirement at exactly 1.0 instead of loosening the tolerance |
| 2026-09-11 | `roto.model.geometry` → `roto.geometry`; `roto.program` and `roto.keys` reclassified as shared | Coordinate math and the IR↔array representation are exactness machinery, not model decisions. Filing them under `model/` made shared guarantees look like v1 assets |
| 2026-09-11 | v2 owns its model (`roto/v2/net, traindata, losses, smoothing, train, reconstruct, report, score`) | Charter §4 adds a head at every stage, so v2's network diverges from v1's immediately. Ported with attribution; boundary test still green |
| 2026-09-11 | S0 keeps v1.1's `HEADLINE` configuration unchanged | Step 2 produces an anchor; an anchor measured with a term nobody has measured before anchors nothing |
| 2026-09-11 | **Soft IoU will be reported on-screen-only beside the headline** | 18% of frames are empty-vs-empty free 1.0s, worth +0.0094 — twelve times the seed spread — and distributed unevenly across elements |
| 2026-09-11 | v2's `RebuildConfig` defaults move off v1's: `tol_px` 1.0 → **0.2**, `smooth_kind` boxcar → **savgol** | Measured on both S0 seeds: gates 2/7 → 4/7, key economy into its band, every element improved. v1 numbers are unaffected — v1 reads its own `roto.model.reconstruct` |
| 2026-09-11 | Held-frame gap gate (≤ 0.002) knowingly missed by more, 0.0028 → **0.0102** | No operating point measured meets it, the old default included. Taken because six of seven gates improve or hold; flagged rather than hidden, and it is a question for S1/S2 to answer, not a dial |
| 2026-09-11 | Key-value refit **measured and refused**, though it wins the charter's own ranking metric | charter L4 (rank on frames < 0.90) picks it; plan §4 (tune on held-out frames) rejects it. Held-out evidence wins: a key value fitted to the predicted track is fitted to its noise |
| 2026-09-11 | **S1's gate is passed by a dial, so an amendment goes into its design note**: grade on key F1 *over the same-count random baseline* | 2c reached 0.753 F1 at 1.119× keys with no learned head. The same-count random baseline rose 0.3681 → 0.6812 over the same change, so 93% of the raw gain is density. The gate as written does not test what S1 is for |
| 2026-09-11 | Step 3 reordered: **S1 → S2 → S3, then the Tier 2 expansion** (was: expansion first) | charter §2 allows one interface change per re-baseline; moving the model and the dataset together measures neither. Also keeps the GPU decision downstream of evidence — S0 used 462 MB of 8,151 MB |
| 2026-09-11 | **"Fire on every live frame" scores 0.5804 key F1 on the 17 trained elements** — above S1's 0.50 gate, from a constant function | Key density spans 47.9× on v003 (1.9% – 90.6%), far worse than the 13× v1 recorded. Second independent demonstration that the raw key-F1 gate does not test key timing |
| 2026-09-11 | S1 `pos_weight` will be **2.14 measured**, not plan §4's ≈13 | Plan §4 derives it from an assumed key rate of ~0.07; v003's measured live-cell key rate is **0.318**. Using 13 would over-weight positives 6× and drive the head toward firing everywhere |
| 2026-09-11 | S1 picker bias is **two-sided** (`key_bias` + `key_slack`), against plan §4's one-sided `tol·(1−a·p)` | Tightening can only *add* keys, and Step 2c leaves us at 1.119× against a 1.3× ceiling. Loosening is what lets the head move a key without adding one. The one-sided form is kept as a measured control |
| 2026-09-11 | **S1 does not pass and is closed without further tuning** | Over-random key F1 0.0720 → 0.0914 against a proposed bar of 0.15, and the head alone scores *below* its best trivial baseline on both seeds. The design note's stop condition ("report it and move to S2, do not tune until it passes") was written before the run and is honoured |
| 2026-09-11 | The two-sided bias is **vindicated as a mechanism** even though S1 fails | One-sided (`slack=0`) pushes keys to 1.18–1.39× and *lowers* over-random; adding slack recovers 1.04× and gives the best over-random in the sweep. Plan §4's one-sided form would have failed the economy band as well |
| 2026-09-12 | **S2's gate is priced before the design, and it holds** — no trivial structural answer comes within 8× of it | The S1 failure was a gate nobody had priced. Always-alive costs 0.0971, drop-one-point 0.0826, drop-a-fifth 0.1143, against a 0.01 budget |
| 2026-09-12 | **The 0.01 gate is flagged as possibly tighter than the task allows**, and the decision is escalated rather than taken | Every lifespan boundary off by one frame costs **0.0102** — the whole budget — so a working head could fail on a margin nobody had measured when the gate was written. Changing a gate after seeing the difficulty is exactly what needs a reviewer, not a local call |
| 2026-09-12 | **Point-count accuracy will be reported and never gated at S2** | The query row is per `(element, shape)` and a point count does not vary with the frame, so the row carries it exactly: 1.000 by construction against 0.077 for the global mode and 0.164 for the per-element mode. Removing `n_points` from `desc` does not help — the loss puts it back. The honest test needs S3's dynamic queries |
| 2026-09-12 | **Decode hysteresis is asymmetric by measurement**, against plan §5's undirected "hysteresis-threshold" | Omission costs 3.2× intrusion at 1% of cells and 7.3× at 10%, because a dead shape's control points sit where the artist left them — usually inside the union — while an omitted live shape leaves a hole nothing fills |
| 2026-09-12 | **Plan §5's loss reweight runs as its own rung before the heads** | Charter §2 allows one interface change per re-baseline and S2 as written changes four things at once. At S2 identity is still given, so correspondence is exact and dot-L1 is still legitimate — the reweight is preparation for S3 and can only cost geometry now. Measured, not assumed |
| 2026-09-12 | **Plan §5's `\|P̂ − P\|` style statistic will be weighted by `1/P`** | The uniform form prices ±1 the same at P = 68 and P = 4; the render does not. `drop_1` costs 0.0004–0.053 on fifteen elements and 0.4995 / 0.4192 on the two whose smallest shape has four points, where a 4-point closed B-spline drops to a near-degenerate 3. Uniform kept as the measured control |
| 2026-09-12 | **Key F1, key economy and the tolerance range keep reading the artist's live span at S2** | If they switched to the predicted span, the S0 → S1 → S2 key columns would stop being comparable and a lifespan regression would read as a key-timing result. The predicted span is reported as its own column |
| 2026-09-14 | **§2.2's "the head needs 99% cell accuracy" is withdrawn as a prediction** | It flipped cells uniformly at random. Measured: **61.4%** of this dataset's 546 lifespan boundaries are invisible in the union alpha (`exp_s2_visibility.py`), so a real head's errors concentrate on exactly the cells that cost nothing. `ts_020036__Car` misses 3.1% of its live cells -- the worst in the set -- and its render *improved*. The gate is unchanged; the derived stop condition is struck through in the note rather than removed |
| 2026-09-14 | **The lifespan decode's thresholds turned out not to matter** | The note argued boundary placement was the work and the hysteresis would be most of the outcome. All 16 swept cells landed in 0.9509–0.9540 on held frames. Recorded because it was a written prediction, not a passing remark |
| 2026-09-14 | **S2C swept its own operating point rather than inheriting S2B's** | The model changed, so reusing the thresholds would assume rather than measure -- and it picked a different cell (`on` 0.7 → 0.5), which justifies the five minutes |
| 2026-09-14 | **Point-count memorisation confirmed and reported, gated on nothing** | 1.000 on trained elements, **0.050 on held-out shots** against a 0.077 global-mode baseline: below the trivial predictor once the memory row is gone. Caveat recorded with it -- an unseen shot has no query row for *anything*, so its 0.04 soft IoU makes this a floor, not a clean isolation |
| 2026-09-14 | **Every training attempt is logged with its cost and a picture**, one diary per stage (`luthra-understands/s2-attempts.md`, `s3-attempts.md`) | Requested, and the right discipline: a record that keeps only the attempts that worked cannot justify the ones that did not. Retry count is a headline field; both stages read **0** so far. Split per stage because the two are read against different baselines -- S2 against "did the picture stay where it was", S3 against "does a hand-traced outline beat it" |
| 2026-09-14 | **With shared slots, a held-out shot is scored by the real model** — the first true generalisation number in the project | Through S2 an unseen element had no query rows, so it was scored with freshly initialised ones and reported as an encoder floor, never a claim. A shared bank applies to any element. `score_run` now splits the table on *what the run trained on* rather than on *how it was scored*; those were the same fact until S3 and are not any more |
| 2026-09-14 | **A reporting bug put four untrained elements in the headline block**, reading 0.9385 as 0.7998 and 1.24 px as 16.3 px | Introduced by the change above and caught only because the training log disagreed with the scorer. Fixed, and pinned by `test_an_untrained_element_stays_out_of_the_headline_even_when_properly_scored` |
| 2026-09-14 | **S3a's failure names the dataset, not the loss** | Held-out *frames* gap 0.0027 against 0.2031 on an unseen *shot*: perfect within a shot, near-total failure across shots, on eight trained shots. Charter §7 reopens a scope decision on a gate failure that names it, and this one names the Tier 2 expansion the tracker had queued behind S3 |
| 2026-09-14 | **S3 note accepted; all three escalated questions answered** | Gate amended to render *over the traced baseline* with style statistics as hard constraints. Charter §7 gives way to plan §6: the low-res composite raster is in scope for S3. Four held-out layers accepted as the gate population, sample flagged, Tier 2 stays queued behind S3. Standing rule restated by the charter owner: **try the plan with tweaks; if it does not work, record it and keep the record** |
| 2026-09-14 | **S3's gate is clearable without decomposition, and an amendment is proposed before any S3 code** | `trace_k@64` — the artist's own alpha traced with 1.2 shapes — scores **0.9697** on held-out layers against charter §4's 0.95. Soft IoU is decomposition-blind by construction (charter L2), so render alone cannot be the whole gate at the stage where the breakdown *is* the task. Proposed: render over the traced baseline, plus style statistics as hard constraints |
| 2026-09-14 | **Chamfer keeps its place in plan §6's loss** — a position reversed by measurement | The note was going to argue that burial made curve-matching meaningless. Over 5,178 sampled live cells the median shape contributes 0.296 of its own ink uniquely and only 16.5% are buried, against 61.4% at lifespan boundaries. Shapes are visible while on screen and invisible at their edges; different questions |
| 2026-09-14 | **Charter §7 and plan §6 conflict on differentiable rasterization**; escalated rather than resolved locally | §7 puts it out of scope "until S3 gates pass"; §6's loss list opens with a low-res composite raster, which is one. Standing rule: ambiguity goes back to the charter owner, not resolved by local improvisation |
| 2026-09-14 | The held-out-layer population is **four elements**, which is thin to gate on | Flagged in the S3 note §8 Q3. Options: gate on held-out shots instead, or wait for the Tier 2 expansion already queued behind S3 |
| 2026-09-14 | Wall-clock per run is **not** comparable across S2 | Scoring and sweeps ran concurrently with training on the same machine; S2A's two identical seeds read 1477s and 1827s. The minutes in the attempt log are honest wall time for justifying the work, not a measurement of training speed |

---

## What Step 1 found

Five things the delivery does that nothing in the plan anticipated. Each is now covered by a
test, because each was silent.

1. **Two frame-number separators.** `MON_..._V01.1030.exr` and `inn020_..._V01_1001.exr`.
   `roto.exr._FRAME_RE` matches only the dot form — and does not fail on the other, it returns
   *no frames*, so the directory is skipped as empty and the shot is graded silver for "no
   delivered mattes" while its mattes sit on disk. It had removed both held-out shots from the
   referee. v2 discovers mattes itself.

2. **`soft_iou` returns 1.0 when both images are empty.** Correct for scoring a reconstruction
   — predicting nothing where there is nothing *is* right — and exactly wrong for a referee: an
   element live on a third of its track collects a free 1.0 on every frame where neither it nor
   the channel draws anything. It scored `ts_020028__char` at **0.8778 against a channel with no
   content at all**. v2 counts a frame as evidence only when both sides draw something, which
   moved that pairing to 0.6590 and dissolved a false collision.

3. **A 1×1-pixel delivered matte** (`ts_021658`). Rejected before it is read — at any reduced
   scale it also makes `cv2.resize` raise.

4. **A delivered matte at a different resolution *and aspect* from its document** (`ts_020355`,
   958×1435 against 2882×2006). `score_against_channel` would crop both to the smaller shape and
   return a confident, meaningless number for two misaligned pictures.

5. **Orphaned channels are normal here.** The delivery carries outputs that are not one `.sfx`
   layer — combined passes, utility mattes, elements from another project. `ts_021555`'s layer
   holds a full-opacity `Square` that no delivered channel shows. This is what `silver` is for;
   failing a shot over it would have dropped 4 of 10.

---

## Open questions

Carry these until answered; do not resolve by local improvisation (plan, standing rules).

- **Silhouette seat** — charter §5's deliverable gate needs one `.sfx` opened, rendered identically,
  and edited in a real seat. Standing external dependency, unchanged.
- **When does Tier 3 come in?** `ts_021191` / `ts_021733` at ~9,000 shapes are deferred, not
  dismissed. They should re-enter when S3's dynamic queries exist, since that is the design that
  makes them tractable.
