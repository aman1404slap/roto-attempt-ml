# roto-v2 POC plan

**Spine adopted from `data/Spline_Roto_Handoff_and_POC_Plan.md` (Lokesh handoff, 29 Aug 2026).**
That document is authoritative for phase order and scope. This file exists to (a) restate the
plan in the terms we execute against, and (b) record where our independent measurements go
beyond or disagree with it — see §4.

Last updated: 2026-08-31 · Status: **Phase 0 done** — 9 packets emitted, 68 tests green. Phase 1 next.
Prior version: `POC.md.bak` in scratchpad. Detailed measurements: [FINDINGS.md](FINDINGS.md).

---

## 1. Goal (unchanged)

Element in, spline program out. Input = one delivered matte channel's alpha sequence
(+ RGB later). Output = a native, artist-style B-spline program with sparse keys, layer
transform factorization, and lifespans. The model never selects the object; the alpha *is*
the selection.

Tier 1: input = alpha we render ourselves from the artist splines, target = the artist's
program. Tier 2 (post-POC): swap input for real Slapshot alpha, target unchanged.

## 2. Phases (from the handoff, verbatim intent)

| Phase | What | Gate |
|---|---|---|
| **0 — Training packets** ✅ | `emit_packets.py`: per element render clean alpha (512-ish crop, keep crop transform), slice IR to that element, emit `{alpha/, target_ir.json, meta.json}` + derived tensors (per-key point arrays, key-time lists, lifespan intervals, per-frame composed layer matrices). Skip sh0230 and MB passes. | ~a dozen elements emitted, each reloadable and re-renderable to the same alpha |
| **1 — Overfit sanity** ✅ | Tiny net memorizes 2–3 elements. | train loss → ~0 and the artist program reproduced on seen data. If this fails the representation is broken, not the model |
| **2 — Teacher-forced animation model** *(the POC)* | Given the artist breakdown at a reference frame, predict per element: layer transform track, per-shape key times, point positions at keys, opacity lifespans. Video encoder over the alpha crop + shape-token decoder. Losses: point L1/L2 at keys, key-time as per-frame binary, lifespan CE, transform regression, **render-consistency at sampled non-key frames**. | beats a baseline on non-key-frame fidelity at ≤ artist-comparable key count |
| **3 — remove training wheels** (post-POC) | Model generates its own breakdown; then Tier 2. Needs the bigger archive, not these 6 shots. | — |

**Evaluation protocol — frozen now:** per element, never averaged: soft-IoU of rendered
prediction vs clean alpha reported **separately for key and non-key frames**; key-count ratio
vs artist; key-time agreement within ±2 frames; point-count parity; frame-to-frame point
velocity smoothness.

**Parallel infra track** (independent of the phases): soft-edge rendering (keep AA values),
feather/blur + motion blur, all-frames verification of the remaining 5 shots, and
project↔matte pairing QC as an archive-ingest gate.

## 3. Answer to the open question: MB / feather / single-frame hair

Boss's question was whether the renderer's missing motion-blur passes, feather, and
single-frame paint-stroke hair are needed for the POC. **All three are out as POC targets,
in as IR fields and infra-track renderer work.** Reasons, in order of force:

1. **Motion blur and feather are render settings, not spline structure.** The model's product
   is the editable program; blur and feather are applied downstream by whoever renders it. A
   blurred target would teach the model to bend geometry to compensate for a shutter — a
   regression, not a feature. The studio itself renders `mb/blue_nmb` no-blur variants, so
   sharp targets are the native ones. Keep `motionBlur` / `feather` as passthrough IR fields
   so the written program is faithful; keep them out of every loss.
2. **The handoff already excludes them** — Phase 0 says "skip flagged pairings (sh0230) and MB
   passes". So this is a confirmation, not a change.
3. **Single-frame hair strokes must be excluded or they poison the key-sparsity loss.** They
   are 51% of shape count but have exactly one key by construction (`hold` opacity 0→100→0
   gating one frame). Mixed into a key-time loss they teach *key every frame* — precisely the
   failure this project exists to prevent. They are also concentrated in sh0230, which the
   handoff already excludes for a pairing mismatch. Filtering to persistent shapes leaves
   **3419 shapes** across the drop — an ample Tier 1 target.
4. **Soft-edge AA is a different thing and it IS needed** — keeping fractional coverage instead
   of thresholding is a small change that raises headline IoU to true accuracy. Infra track,
   do it during Phase 0.

Hair as a target is a **separate stage with its own model and metric**, post-POC. Deferred, not
dropped.

## 4. Where we diverge from the handoff

Independently confirmed and consistent: both dialects, coordinate convention, row-vector
transform composition, B-spline-native targets, opacity-as-lifespan, motion factorization,
nfl_0200 economy (~11 pts / ~14 keys), layer→channel union discovery, and the finding that the
residual error is edge AA rather than geometry. No conflicts on any of that.

Four deltas, all additive:

- **4.1 sRGB-encoded alpha in FAM_0060 and MAT_0130 — not in the handoff.** Their 256 stored
  values match `sRGB_to_linear(k/255)` to float16 precision: 50% coverage is stored as 0.214.
  Any soft-edge loss taken against those two shots without inverting it is systematically
  wrong. Handled in `src/roto/matte/exr.py` (auto-detected). **Port this into the toolkit's
  `load_matte_channels` before Phase 0**, or the emitted packets inherit the bug.
- **4.2 `.sfx` writer is not in the handoff's phase list at all.** The handoff's plan is
  correct that Tier 1 trains without it — the target is `target_ir.json`. But the *product* is
  a file an artist opens in Silhouette, and we have never verified that anything we write
  loads. Keep it on the **infra track running alongside Phase 0–1**, not as a blocking gate.
  Its own gate: our renderer produces identical output from original vs round-tripped file,
  and Silhouette opens the result with the right shapes and hierarchy. Needs one seat (§5).
- **4.3 A classical baseline (contours + planar track + greedy key dropping) is worth ~1 week
  during Phase 1.** The handoff's Phase 2 gate has nothing to beat. We measured the layer
  transform carrying 57–95% of shape motion (95.8% on nfl_0200/green), so a classical fit may
  land close to artist-acceptable. Either outcome is decision-grade before we build a network,
  and it is a shippable fallback if the ML slips. Proposing this as an addition, boss's call.
- **4.4 Persistent-vs-ephemeral must be a first-class field in `target_ir.json`,** not a filter
  applied at training time. Same for fill-vs-stroke. §3 item 3 is the reason.

Both numbers that looked like contradictions are now reconciled — see §5 *Reconciled*.
Neither was a disagreement about the data: one was a stale constant, the other three
different layer→channel pairings being compared as if they were the same measurement.

## 5. Codebase: one, not two

Two implementations now exist. `data/roto_toolkit.py` (682 lines, single file) and `src/roto/`
(1741 lines, packaged, 29 tests green, plus exact B-spline→Bézier, sRGB decoding, and
error-decomposition-by-distance-to-boundary that the toolkit lacks).

**Done, 2026-08-31.** `src/roto/` is the codebase; the toolkit's CLI surface and
`roto_ir.json` field names are the contract. `data/roto_toolkit.py` is left exactly as
delivered — it is the reference, not a live dependency.

| # | merge item | where |
|---|---|---|
| 1 | CLI, same names/flags/defaults as the toolkit | `src/roto/cli.py`, `python -m roto` |
| 2 | greedy union layer→channel matcher, `soft_iou`, `dice` | `src/roto/eval/match.py`, `eval/metrics.py` |
| 3 | toolkit-schema `roto_ir.json`, both directions | `src/roto/sfx/json_ir.py` |
| 4 | sRGB decode + soft-edge AA in the shared path | `matte/exr.py`, `render/raster.py` |
| + | shot discovery across both vendor layouts | `src/roto/data/shots.py` |
| + | TRS transforms (the toolkit had them, we did not) | `sfx/read.py`, `render/raster.py` |
| + | persistent/ephemeral + fill/stroke as IR fields (§4.4) | `ir.py: shape_class` |

Commands are unchanged, so anything scripted against the toolkit still works:

```
PY=/home/aman/.pyenv/shims/python3          # the python3 first on PATH has no numpy
PYTHONPATH=src $PY -m roto inventory  data/extracted/test_data
PYTHONPATH=src $PY -m roto parse      data/extracted/test_data/<shot>
PYTHONPATH=src $PY -m roto verify     data/extracted/test_data/<shot> --scale 0.35 --stride 5 --viz-dir viz
PYTHONPATH=src $PY -m roto verify-all data/extracted/test_data --report report.csv
```

What changed underneath: per-key interpolation instead of a global `INTERP_MODE`, sRGB matte
decoding, anti-aliased edges on by default (`supersample=2`), `soft_iou` reported next to
thresholded IoU, and each channel's worst frame labelled edge-only or STRUCTURAL.

### Reconciled

- **`STROKE_WIDTH_GAIN`** — the handoff doc was right, the toolkit code was stale. 0.0625 is
  now a named constant in `render/raster.py`. The toolkit's 1.0 is what made its stroke
  renders hairline, and is most of why its `nfl_0080` MB number was 0.08.
- **`nfl_0080` MB, all three figures explained.** The matcher finds `mb 1/red → matte01/R` at
  **0.984** — matte01 is a *non-MB* pass and reproduces fine. The actual MB pass is
  `matte02/R`, which scores **0.200** because motion blur is not modelled. Our old 0.883 was
  a hand-picked pairing (`MB 2 → matte01/R`) that the matcher beats. Nothing was wrong with
  the renderer; the pairings were wrong. This is the empirical case for §3.
- **TRS is identity on every layer of all six shots** — asserted in a test now, so it cannot
  explain any IoU gap, and a future shot that does use it cannot fail silently.

## 6. Asks

| # | ask | blocks |
|---|---|---|
| 1 | **One Silhouette seat** | §4.2 writer verification, and the deliverable |
| 2 | Sign-off that MB / feather / hair are out as POC targets (§3) | Phase 0 scoping |
| 3 | RGB plates for a subset | RGB branch, Tier 2 — not Tier 1 |
| 4 | **Authoritative layer→channel mapping** (render-node settings), at least for `TVC_sh0260` | 7 of 23 channels; §7 |
| 5 | Newer sh0230 save, or a manifest linking matte renders to project versions | archive-scale ingest QC |
| 6 | Archive-wide format census | IR design at scale — 6 files gave 4 dialects |

## 7. Phase 0 results

`PYTHONPATH=src python -m roto packets packets --tier verified` → **9 elements, 1210 frames,
1310 shapes, 10 957 path keys, 23 MB.** Layout per element, as the handoff specifies:

```
packets/<element_id>/
  alpha/<sfx_frame>.png   16-bit clean alpha rendered from the artist splines (Tier 1 input)
  target_ir.json          the element's own IR, toolkit schema, native B-splines
  tensors.npz             key_frames, points_norm, opacity, lifespan, layer_matrix per shape
  meta.json               provenance, crop transform, per-shape index, verify scores
  preview.png             first / middle / last, for eyeballing
```

The element manifest is checked in at [src/roto/data/elements.py](src/roto/data/elements.py),
generated by [scripts/discover_elements.py](scripts/discover_elements.py) — the layer→channel
mapping is recovered knowledge, so it is reviewable in a diff rather than re-derived silently
on every run. 9 verified, 14 excluded, each exclusion carrying its reason.

| element | IoU | frames | shapes | pts/shape | keys/shape | **keys per live frame** |
|---|---|---|---|---|---|---|
| `MAT_0130/matte01/R` | 0.9980 | 55 | 16 | 10 | 5 | **1.67** ⚠ |
| `FAM_0060/matte02/B` | 0.9973 | 133 | 11 | 14 | 7 | 0.06 |
| `nfl_0200/matte01/R` | 0.9944 | 191 | 75 | 11 | 14 | 0.08 |
| `nfl_0080/matte01/R` | 0.9930 | 154 | 2 | 19 | 42 | 0.27 |
| `TVC_sh0260/matteL110/R` | 0.9898 | 77 | 10 | 14 | 18 | 0.64 |
| `FAM_0060/matte02/R` | 0.9861 | 133 | 2 | 13 | 4 | 0.03 |
| `FAM_0060/matte01/G` | 0.9858 | 133 | 147 | 11 | 5 | 0.04 |
| `nfl_0200/matte01/G` | 0.9858 | 191 | 11 | 11 | 14 | 0.17 |
| `FAM_0060/matte01/B` | 0.9843 | 133 | 1036 | 15 | 8 | 0.07 |

Median 10–19 points per shape confirms the handoff's "~11 points per shape". Eight of nine
elements key on **0.03–0.64 of their live frames** — that sparsity is the product.

### Three things Phase 0 turned up

**7.1 Every fidelity number was being measured wrong.** Comparing a downscaled render to a
`cv2.resize`-downscaled EXR carries a half-pixel convention mismatch: our renderer maps output
pixel `i` to source `i/scale`, cv2 maps it to `(i+0.5)/scale − 0.5`. On
`nfl_0200/matte01/R`: **0.9773 at scale 0.35, 0.9942 at 0.7, 0.9947 at 1.0**, with 100% of
the 731 disagreeing pixels within 1px of the boundary. The error is entirely at the edge, so
it is invisible in aggregate and reads as renderer inaccuracy.

Consequence: the first manifest, scored at 0.35, mis-tiered elements. `nfl_0200/matte01/G` —
our own named POC-ladder target — read 0.948 and was excluded; at full resolution it is
**0.9858** and verified. Discovery now matches layers at a cheap scale (only relative ranking
matters there) and scores at 1.0. **Every fidelity number in §8 and earlier is understated by
up to 0.02.** The §7 table above is the corrected one.

**7.2 MAT_0130 has the best IoU and the worst keyframe economy.** It keys **1.67 times per
live frame** — more often than every frame — against 0.03–0.64 everywhere else, 10–20× denser.
FINDINGS §5.4 called it unrepresentative WIP on shape count; this is the quantitative version.
It is the single most dangerous element to train sparsity on precisely because it scores
highest, so it should be excluded from Phase 1/2 key-timing work while staying useful for
geometry and pipeline debugging. Same argument as §3 item 3, from a different direction.

**7.3 A static crop is wrong for anything that moves.** `nfl_0200`'s person-and-chair is a
~500×380px object that crosses the whole 2880px frame, so a crop covering its union bbox came
out **larger than the frame**, leaving the element at **1% coverage** — unusable. Packets now
use a **fixed-size window that tracks the element**: the size and therefore the scale are
constant (so point distances stay comparable between frames), and only the translation varies,
recorded per frame in `meta['crop']['offsets']` — data the model is given, not motion it has
to infer. Coverage went 0.011 → 0.278. There is a regression test on minimum coverage.

Also fixed on the way: `render()` centred on the *rounded* canvas rather than the true image
centre scaled, so every `scale<1` render was off by up to half a pixel — a crop now equals the
corresponding window of the full render exactly, at every supersample level. And `render()`
zeroed a full-canvas scratch buffer per shape, a 16 MB memset × N shapes at a 512 crop with
supersample 4; compositing inside each shape's own bbox made shape-dense elements **~40×
faster** (1036 shapes: 10 s → 230 ms per frame).

### Phase 0 notes for Phase 1

- `FAM_0060/matte01/B` has **1036 shapes** in one element, against the handoff's "typically
  tens". It is either an outlier or wants splitting; do not let it dominate an overfit test.
- `nfl_0200/matte01/G` renders at **scale 4.4** (a 116px element upsampled to 512). That is
  real resolution, not interpolation — we rasterise from splines at the output size — but it
  means its geometry detail is finer than any raster the archive contains.
- Overfit candidates (Phase 1): `FAM_0060/matte02/B` (11 shapes) and
  `nfl_0200/matte01/R` (75 shapes, tracked, sparse). Both small, both sparse, both ≥0.994.

## 8. Phase 1 results — the overfit gate

`PYTHONPATH=src python -m roto overfit packets/<element>... --steps 8000`

Two gates, in order, because a failure in the second is ambiguous on its own.

### Gate 1 — representation round-trip (no model at all)

Load a packet's target tensors, decode them back to a document, render, compare to the
packet's own alpha. **soft-IoU 1.000000 on all nine elements.** The representation is lossless.

This is the single most important number produced so far. If the artist program could not
survive a trip through `ProgramTensors`, no model trained on those tensors could reproduce it,
and every later fidelity number would be measuring the representation's ceiling rather than
the model. It is now a test over all nine packets, so it cannot silently regress.

Three measured facts made it lossless, each of which would otherwise have been a silent
truncation:

| fact | consequence for the target |
|---|---|
| Transform tracks are **affine, never perspective** on every measured track — but **not** similarity (`TVC_sh0260` and `nfl_0080` carry shear) | 6 DOF per frame, not 16 and not 4 |
| 1310 shapes resolve to **64 distinct transform tracks** (`nfl_0200/R`: 75 shapes, 3 tracks) | predict per *group*, not per shape — a 20× smaller target that cannot disagree with itself about one rigid group's motion |
| **0.8% of path keys sit outside the rendered frame range** (e.g. a key at frame −1) | the key mask needs its own `key_axis`; a mask over rendered frames drops them, and a dropped key changes every frame up to the next one |
| Opacity is **strictly binary** — not one of 1310 shapes ever between 0 and 100 | a per-frame live mask is *exact*, not an approximation of a fade |

### Gate 2 — overfit 3 elements: **PASS**

8000 steps, 903k params, 29 min CPU. Thresholds were fixed before the run, not after.

| element | point err | key F1 | live F1 | transform | render soft-IoU |
|---|---|---|---|---|---|
| `FAM_0060/matte02/B` (11 shapes) | **0.000px** | **1.0000** | 0.9996 | **0.000px** | **1.0000** |
| `nfl_0200/matte01/G` (11 shapes, 4.4× crop) | **0.000px** | **1.0000** | **1.0000** | **0.000px** | **1.0000** |
| `nfl_0200/matte01/R` (75 shapes, incl. beziers) | **0.000px** | **1.0000** | 0.9989 | **0.000px** | 0.9984 |

Final loss 0.0434, from 3816 at step 0. The decoded programs render back to the alpha they were
trained on at soft-IoU ≥0.9984 — which is the requirement that actually matters, because a low
loss on padded masked tensors can hide a program that does not draw the right thing.

**So the representation and the losses can express the artist program.** That is the whole
claim Phase 1 was built to test, and it now holds end to end: pixels → tokens → heads →
discrete program → IR → render → back to the same pixels.

### The ablation — where Phase 2 actually starts

Same run with `--no-memorise`, i.e. the per-element bias tables removed so everything must
come through the encoder. This is the honest Phase 2 baseline, and the gap is the work:

| element | | point err | key F1 | key precision | key recall | render |
|---|---|---|---|---|---|---|
| `FAM_0060/matte02/B` | memorise | 0.000px | **1.0000** | 1.000 | 1.000 | **1.0000** |
| | encoder-only | 0.100px | 0.5675 | 0.396 | 1.000 | 0.9647 |
| `nfl_0200/matte01/G` | memorise | 0.000px | **1.0000** | 1.000 | 1.000 | **1.0000** |
| | encoder-only | 1.233px | 0.6154 | 0.444 | 1.000 | 0.6836 |
| `nfl_0200/matte01/R` | memorise | 0.000px | **1.0000** | 1.000 | 1.000 | **0.9984** |
| | encoder-only | 1.875px | 0.3331 | 0.214 | 0.750 | 0.7239 |

Two things to carry into Phase 2:

- **Geometry is nearly solved by the encoder already** — point error 0.10–1.88px and lifespan
  F1 ≥0.998 without any memorisation. The encoder can see where the shapes are.
- **Key timing is not, and it fails in the project's signature direction.** Precision
  0.21–0.44 against recall 0.75–1.00: the encoder-only key head finds the artist's keys and
  then fires on roughly 2.5× too many frames besides. That is *over-keying* — exactly the
  "key every frame" failure the whole project exists to avoid, showing up as the first thing
  the model does when left to itself. Keyframe economy is the hard problem, as the reference
  doc says; this is the measurement that confirms it rather than assuming it.

Render fidelity being 0.68–0.96 encoder-only while geometry is sub-2px is itself the point:
**the keys, not the shapes, are what break the render.**

### Three findings from building it

**8.1 The per-frame binary key formulation collapses under plain BCE, and hides behind a good
metric.** Only ~6% of (shape, frame) pairs are keys — that sparsity *is* the product. Trained
with unweighted BCE, the key head went to "never key" and plateaued at **exactly**
`1 − positive_rate`: 0.9440 on `FAM_0060/matte02/B` (positive rate 0.0560) and 0.9372 on
`nfl_0200/matte01/G` (0.0628). A 94% accuracy that predicts nothing. Fixed with
`pos_weight = negatives/positives` — which also handles the live mask, where positives
dominate at 96% and the *negative* class is the rare one — and the gate is now stated on **key
F1, never accuracy**. There is a test asserting the trap: an all-negative head must score
>0.9 accuracy and exactly 0.0 F1.

The handoff picks this formulation ("start with the binary formulation, it's simpler") and it
is the right starting point — but it needs class balancing to work at all, and that is not
obvious from the outside.

**8.2 Phase 0's tracking crop makes the transform track unrecoverable from pixels alone.** This
is a direct consequence of §7.3 and I had not seen it: the crop window follows the element, so
it removes exactly the gross motion the transform track encodes. Measured on
`nfl_0200/matte01/G`: the window travels **2638px** while the cropped alpha changes by a mean
absolute **0.084** between frames. A model given only the cropped alpha cannot fit the
transform at any capacity. The crop offsets are now an explicit per-frame model input, and
the transform error went to **0.000px**. Phase 0's meta called the offsets "data the model is
given, not motion it has to infer" — that turns out to be a requirement, not a convenience.

**8.3 Both binary heads and the point head share one architectural bottleneck.** `key_logits`
was `MLP([shape_token, frame_feature])` — a rank-limited outer product of one shared per-frame
vector with one per-shape vector, which cannot express an arbitrary per-shape key pattern. It
plateaued at F1 0.18–0.33 with recall 0.65–0.78 against precision 0.11–0.21, firing on ~6×
too many frames regardless of training length. The point head has the identical problem: 220
distinct key slots on one element separated only by two 64-d vectors.

Phase 1 is not an architecture test, and conflating the two is how a representation bug hides
for a week. So the tiny model carries per-element **bias tables** on the key, live, affine and
point heads under a `memorise=True` flag — a lookup table, deliberately, because an overfit
test should be able to overfit. **Phase 2 must run `--no-memorise`**, and the gap between the
two settings is the honest measure of what the encoder still has to learn.

### For Phase 2

- Predict transforms **per group** (64 tracks, not 1310 shapes) as 6-DOF affine residuals from
  identity. The affine head is initialised at identity so step 0 is already a rigid program.
- Feed the crop offsets. Not optional — see §8.2.
- Gate on **key F1**, and report precision/recall separately; accuracy is actively misleading
  at this sparsity.
- Geometric losses are in **packet pixels** (`× px_per_norm`), so one weight is meaningful
  across elements whose crop scales differ 30×. Note the flip side: a 1-pixel threshold is
  ~10× stricter in normalised terms for `nfl_0200/matte01/G` (px_per_norm 8731) than for
  `FAM_0060/matte02/B` (812).
- The encoder-only baseline is already measured (above) — beat **key F1 0.33–0.62** and
  **render 0.68–0.96**, and treat point error as close to solved.
- Attack precision, not recall. The key head already finds the artist's keys; it fires far
  too often besides. A sparsity penalty on predicted key count, or the handoff's alternative
  "regress sorted key times" formulation, both target precision directly.
- The **render-consistency loss at non-key frames** is the natural next term and is not yet
  implemented — Phase 1 uses analytic losses only and our renderer for evaluation, which is
  what the handoff specifies. That is the one Phase 2 component with no Phase 1 evidence
  behind it.

## 9. First exhaustive sweep (all 23 channels, not 7 hand-picked)

*Numbers below were measured at scale 0.35 and are understated by up to 0.02 — see §7.1. Kept
because the shape of the result stands: coverage, not fidelity, was the finding.*

`verify-all --scale 0.35 --stride 10`, aggregate **mean IoU 0.694 / soft-IoU 0.709 over 23
channels, 261 frame checks**. That number is *not* a regression — it is the first time every
delivered channel was scored rather than the seven we had hand-mapped. The spread is the point:

| shot | channels | result |
|---|---|---|
| `MAT_0130` | 1 | 0.995, edge-only |
| `nfl_0200` | 2 | 0.981 / 0.956, both edge-only |
| `FAM_0060` | 6 | 0.879–0.994; four best are 0.977–0.994 |
| `nfl_0080` | 4 | 0.984 on the non-MB pass; **0.200 on the MB pass** |
| `TVC_sh0260` | 7 | 0.264–0.980 — one top layer against three delivered mattes, matcher guessing |
| `TVC_sh0230` | 3 | **0.002** thresholded / 0.128 soft — total failure |

Reads directly onto the plan:

- **`sh0230` is unusable**, confirming the handoff's exclusion from two independent
  directions: its matte was rendered from a different save *and* it is 87% single-frame hair
  strokes (3224 of 3688 shapes ephemeral). Not a renderer bug to chase.
- **The MB pass fails exactly as predicted** and is the concrete argument for §3.
- **`sh0260` is a matching problem, not a geometry problem** — a single top layer `Layer 52`
  against three mattes gives the greedy union nothing to latch onto. Needs either the
  authoritative render-node mapping (ask #4) or manual pairing before it can be a Phase 0
  element.
- **Phase 0 elements are the ones that already reproduce**: `MAT_0130/R`, `nfl_0200/R`,
  `nfl_0200/G`, and FAM_0060's four good channels. Seven elements, all edge-only or near it.
  That is fewer than "roughly a dozen", so Phase 0's element count should be stated as **7
  verified + the rest pending a mapping**, not a dozen.
- **The STRUCTURAL flag is deliberately harsh** — it profiles only each channel's *worst*
  frame, so `FAM_0060/matte02/B` shows STRUCTURAL at 0.994 mean. Treat it as "look here",
  not "broken".

## 10. Log

- **2026-08-31** — **Phase 1 complete.** 125 tests green. New: `src/roto/train/`
  (`program.py`, `losses.py`, `model.py`, `overfit.py`), `roto overfit`. Gate 1 —
  representation round-trip — passes at **soft-IoU 1.000000 on all nine elements**, and is
  now a test, so the target representation is provably lossless before any model work. Gate 2
  reaches key F1 1.0000 and transform 0.000px on all three elements. Three findings, all from
  building rather than reasoning: unweighted BCE collapses the key head and hides at 94%
  accuracy (§8.1); Phase 0's tracking crop makes the transform unrecoverable from pixels, so
  the crop offsets are a required input (§8.2); and the key/point heads share a rank
  bottleneck that Phase 1 must not mistake for a representation failure (§8.3).
- **2026-08-31** — **Phase 0 complete.** 9 packets, 68 tests green. New:
  `src/roto/packets.py`, `src/roto/data/elements.py` (generated), `scripts/discover_elements.py`,
  `scripts/emit_packets.py`, `roto packets`. Three real bugs found by building it rather than
  by reasoning about it: the scoring-scale half-pixel mismatch (§7.1), the static-crop coverage
  collapse (§7.3), and the per-shape full-canvas memset (~40× render speedup). The manifest is
  the durable artefact here — `tests/fixtures.py` is now superseded by it and should be deleted
  once nothing imports it.
- **2026-08-31** — Codebases merged; 29 tests → **51**. New: `data/shots.py`,
  `eval/match.py`, `sfx/json_ir.py`, `cli.py`. JSON IR round-trips bit-identical on all six
  shots (rendered output equal, not just stats). Three things fell out of doing it properly:
  (a) **tracked layers are always leaves** in all six shots — named group layers
  (`core`/`face`/`body`/`CH1`) are never tracked, tracked ones are auto-named `Layer N` leaves
  owning shapes. So the artist's motion factorization is *one tracker per shape group*, not a
  hierarchy of nested transforms — that is the shape the model's transform head should
  predict, and it is now a test. (b) The matcher beats our hand-recovered pairings in two
  places (`nfl_0200/G` → `green/hair/Layer` not `green`; `nfl_0080/matte01/R` → `mb 1/red`
  at 0.984 not `MB 2` at 0.883), so `tests/fixtures.py` is now the *weaker* mapping and
  should be regenerated from the matcher. (c) We had no TRS support at all; it is identity
  everywhere, so it bought no accuracy — but it was a silent-wrong-position risk.
- **2026-08-31** — Handoff doc received and reconciled. POC.md rewritten to adopt the handoff's
  Phase 0–3 spine; our prior Stage 0/1 demoted from blocking gates to an infra track (§4.2) and
  a proposed addition (§4.3). Independent findings that survive: sRGB-encoded alpha in two shots,
  the persistent/ephemeral split, exact B-spline→Bézier. MB/feather/hair answered: out as
  targets, in as IR fields.
- **2026-08-27** — Pre-handoff work: package scaffolded, 29 tests, renderer at 0.968–0.995 IoU
  across seven elements with error confined to <3px of the boundary on five of them; opacity
  gating and sRGB decoding were each worth real accuracy. Detail in [FINDINGS.md](FINDINGS.md)
  and the scratchpad `POC.md.bak`.
