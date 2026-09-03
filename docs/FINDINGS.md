# roto-v2 — problem statement + test data inventory

Status: data reconnaissance complete. No model work started.
Source docs: `data/Spline_Rotoscoping_High_Level_Primer.docx`, `data/Spline_Rotoscoping_Technical_Detail.docx` (Aug 24 2026).

## 1. The problem, in one paragraph

Slapshot Roto already answers *"which pixels are the object?"* — it emits a raster alpha sequence.
It does not answer *"how would a professional roto artist represent this object as an editable
spline animation?"* — the shape breakdown, the control-point economy, the layer hierarchy, and
the sparse keyframe timing. That structured artist program is what makes roto stable, editable,
and deliverable, and it is what our Hotspring archive contains.

So the task is **inverse rendering with a learned artist prior**: given RGB + alpha, predict the
spline program. The inversion is one-to-many (many spline programs render the same matte), and the
archive is the supervision that picks the *artist-plausible* one. Success is not matte IoU — it is
**artist correction time**: how much work remains before a professional calls the spline file usable.

The docs stage it as: (1) canonical roto IR + deterministic renderer that round-trips, then
(2) Tier 1 train on clean artist-rendered alpha, then (3) Tier 2 retrain on real Slapshot alpha.
**Step 1 is the gate**, and that is exactly what the test data lets us attack now.

## 2. What is actually in `data/test_data.zip`

161 MB zipped / 254 MB extracted → `data/extracted/test_data/`. Six shots from four different jobs:

| Shot | Res | Frames | .sfx dialect | Layers | Shapes | Delivered mattes |
|---|---|---|---|---|---|---|
| `FAM_0060_L1_A0003C007_v001` | 2880×1620 | 133 | 2022.5 (zlib) | 159 | 1604 | matte01, matte02 (RGB each) |
| `MAT_0130_L1_C002_260809_v001` | 2880×1620 | 55 | v5 (plain) | 2 | 16 | matte01 (R only) |
| `TVC_SHOTS_sh0230_BG01_v003` | 2496×1280 | 50 | v5 (plain) | 60 | 3688 | matte_L110 only |
| `TVC_SHOTS_sh0260_BG01_v003` | 3055×1280 | 77 | v5 (plain) | 185 | 592 | matte_L100/L110/L120 |
| `nfl_0080_bg02_v001_compplate` | 2880×1978 | 154 | 2025.5 (zlib) | 95 | 983 | matte01, matte02 |
| `nfl_0200_bg01_v001_compplate` | 2880×1978 | 191 | 2020 (zlib) | 11 | 86 | matte01 |

Totals: **6969 shapes, 52 565 path keyframes, 660 frames of finished production roto, 1101 EXR frames.**

### Per shot you get exactly two things

1. **One Silhouette `.sfx` project** — the artist spline program (the supervision signal).
2. **One or more rendered matte EXR sequences** — 16-bit half, `zips` (lossless), written by
   **Nuke** (`nuke/node_hash` in the header), R/G/B carrying **three independent, freely
   overlapping** mattes per file.

   The container is lossless but **the content is not**: every channel in every shot holds only
   **256 distinct values**, i.e. the matte was 8-bit somewhere upstream. Worse, the encoding is
   inconsistent across vendors:

   | Shots | Stored values | Meaning |
   |---|---|---|
   | `TVC_sh0230`, `TVC_sh0260`, `nfl_0080`, `nfl_0200` | `k/255` | correct — 0.5 means 50% coverage |
   | `FAM_0060`, `MAT_0130` | `sRGB_to_linear(k/255)` | **an sRGB→linear colour transform was applied to an alpha channel** — 50% coverage is stored as 0.214 |

   Verified exactly: the 256 values in `FAM_0060` match `sRGB_to_linear(k/255)` to float16
   precision. An alpha matte is not a colour, so this is a pipeline bug at the vendor. Any raster
   loss taken against these EXRs without inverting the transform will have systematically wrong
   soft edges — and soft edges are precisely where hair, feather and motion blur live. Hard
   interiors (0 and 1) are unaffected, which is why the §3 IoU numbers are still valid.

### What you do NOT get — the two blocking gaps

- **No RGB plates.** Every `.sfx` points at a Windows path on someone's workstation
  (`D:/Roto_2026/...`, `E:/Hotspring/...`, `C:/footage/...`). The training recipe in both docs
  requires aligned RGB. Right now we can do matte→spline only, not RGB+alpha→spline.
- **No Nuke comp.** The `.sfx` layers were rendered *per layer* into colour-named folders
  (`render/L110/blue`, `render/green`, `mb/blue_nmb`, `hard/r1/n`) and a Nuke script packed them
  into the RGB channels. That packing recipe is not in the archive — see §4.

## 3. The `.sfx` format is fully readable — confirmed, with a working renderer

Two container variants, both plain Python, no Silhouette licence:

- 3 of 6 files: raw XML starting `<!-- Silhouette Project File -->`
- 3 of 6 files: big-endian uint32 uncompressed-length + zlib stream

Tools written: [tools/sfx_inspect.py](tools/sfx_inspect.py) (decode + inventory),
[tools/sfx_render.py](tools/sfx_render.py) (decode + rasterise), [tools/sfx_classify.py](tools/sfx_classify.py)
(shape taxonomy). Outputs: [data/sfx_inventory.txt](data/sfx_inventory.txt),
[data/sfx_shape_classes.txt](data/sfx_shape_classes.txt), [data/previews/](data/previews/).

### Confirmed conventions

- Coordinates normalised, **center origin, y-down, both axes divided by image height**.
- Layer transforms are **row-major 4×4 with translation in the last row**, so points are row
  vectors: `p' = p @ M_leaf @ … @ M_root`. Getting this backwards is what makes a naive parse
  render garbage.
- Key interpolation modes: `linear`, `hold`, `catmullrom`. Blend modes `Add`/`Subtract`.
- Point count per shape is **constant across all its keys** in all 6 files → point
  correspondence is stable and shape topology is fixed. Good news for the IR and for
  trajectory-based losses.
- Deep, semantically named layer hierarchies (up to 6 levels): `green 2 / CH1 / face / Layer 26`,
  `Layer 52 / details / ppp`, `body`, `core`, `detail`, `cp`, `op`, `shadow`, `SR_Hand`, `SL_Hand`.
  This is free weak supervision for the shape-breakdown and semantic-class heads.

### Round-trip validation (the Step-1 gate)

Rendered the parsed splines and compared to the delivered mattes at 0.5 threshold:

| Shot | Layer → channel | IoU |
|---|---|---|
| MAT_0130 | `Red Matte` → matte01/R | **0.965** |
| nfl_0200 | `blue` → matte01/R | **0.975** |
| nfl_0200 | `green` → matte01/G | **0.936** |
| FAM_0060 | `blue` → matte02/B | **0.992** |
| FAM_0060 | `green` → matte01/G | **0.977** |
| FAM_0060 | `red`+`red 1` → matte02/R | **0.985** |
| FAM_0060 | `green 2`+`r1_w_c`+`green 1`+`r1_t_c` → matte02/G | **0.978** |
| FAM_0060 | `blue 1` → matte01/B | 0.813 |
| nfl_0080 | `MB 2` → matte01/R | 0.846 |

**The archive round-trips.** A few hundred lines of Python reaches 0.94–0.99 IoU on filled-region
roto with no feather, no antialiasing and no motion blur modelled. Closing the remaining gap is
engineering (edge softness, feather, motion blur), not research.

## 4. Four things the docs do not account for

These are the real POC risks, in priority order.

### 4.1 The shapes are B-splines, not Béziers

6919 of 6969 shapes are `shape_type="Bspline"`; 26 are `Bezier` (and those are 4-point tracking
squares, not roto); 5 are `Xspline`. Both docs assume "Bézier control points and handles" and pick
DiffVG on that basis. B-spline control points do **not** lie on the curve and there are no handles.

Consequences: the IR must be B-spline-native or convert (uniform cubic B-spline → Bézier is exact
and cheap, so DiffVG still works as a *loss*), but the model's output head and every geometry loss
should predict **B-spline control points**, because that is what the artist authored and what an
artist has to edit on delivery. Predicting Béziers would mean the delivered file is not in the
artists' working representation.

### 4.2 Half the archive is per-frame disposable detail strokes, not animated shapes

| Shot | open/stroke shapes | ephemeral (alive ≤2 frames) |
|---|---|---|
| TVC_sh0230 | 2515 / 3688 (68%) | 3225 (87%) |
| nfl_0080 | 516 / 983 (52%) | 153 (16%) |
| FAM_0060 | 504 / 1604 (31%) | 0 |
| TVC_sh0260 | 8 / 592 (1%) | 165 (28%) |
| MAT_0130 | 0 / 16 | 7 (44%) |
| nfl_0200 | 0 / 86 | 0 |
| **total** | **3543 / 6969 (51%)** | **3550 / 6969 (51%)** |

Open shapes carry a `strokeWidth` property and are rendered as **strokes, not fills**. And they
carry animated `opacity` with `hold` keys `0 → 100 → 0`, gating each one to a **single frame**.

`TVC_sh0230`'s delivered matte is the clearest case: it is nothing but **hair wisps** — thousands
of thin open splines, each drawn fresh on one frame. Rendering all 2515 at once over-covers by 11×;
gating on opacity brings 2515 → 184 live strands on frame 25 and puts them on the right pixels
(0.95 recall at 3 px stroke width on frame 12; see `data/previews/rt_hair_f25.png`).

This matters a lot:
- **A filled-path renderer alone cannot reproduce ~half the archive.** The IR needs stroke shapes
  with width, and per-shape opacity animation, or those shots are unusable as targets.
- **"Keyframe economy" is meaningless for these shapes.** They have one key by construction.
  Mixing them into a keyframe-timing loss teaches the model to key every frame — the exact
  failure mode the project exists to avoid. Persistent and ephemeral shapes must be **separate
  prediction problems**, or the ephemeral ones excluded from Tier 1 entirely.
- Filtering to persistent shapes leaves **3419 shapes** — still a fine Tier 1 target, and the
  keyframe statistics on that subset are genuinely sparse and artist-like
  (median 7–16 keys per shape over 50–191 frames; ~0.1–0.9 keys per live frame).

### 4.3 Motion blur is baked into the delivered mattes

Shapes carry `motionBlur=true`, and layers are literally named `MB 2` / `mb 1`; render paths
include `mb/blue_nmb` (no-motion-blur variant). Delivered hair mattes peak at **0.76, not 1.0** —
consistent with 180° shutter smear. Any raster loss against these mattes is comparing a
sharp render to a blurred target unless motion blur is modelled or a non-MB variant is used.
The docs' "compare to the clean artist alpha" rule needs a definition of *clean* that says
whether motion blur is in or out.

### 4.4 The layer → matte-channel mapping is not in the data

It lives in a Nuke script we do not have. Recovered by brute force above: each channel is the
**union of one or more top-level Silhouette layers**, and layer names only sometimes hint at the
channel (`blue`→B, `green`→G, but FAM_0060's `green 2 + r1_w_c + green 1 + r1_t_c` → matte02/G).
Two shots are worse: `TVC_sh0230` has layers `L100` and `L110` but only the `L110` matte shipped,
and `TVC_sh0260` has a single top layer `Layer 52` against three delivered mattes.

For a manufactured-pairs pipeline this is fine — **we render the alpha ourselves from the IR and
never need the delivered EXRs as targets.** They are validation data, not training targets. Worth
being explicit about that, because it changes what we need to ask Hotspring for.

## 5. What to ask for before the POC

1. **The RGB plates for these six shots.** Without them this is a matte→spline dataset, and
   neither Tier 1 nor Tier 2 as written is runnable. Highest-value single ask.
2. **A no-motion-blur matte render**, or confirmation that MB should be modelled.
3. **Archive-wide format census** — we have 4 Silhouette dialects in 6 files (2020, 2022.5, v5,
   2025.5). Before committing to the IR we should know how many dialects and how much
   hair/stroke work the full archive holds, since §4.2 says that decides half the parsing work.
4. Confirmation of which shots are "clean" deliveries vs partial/WIP. `MAT_0130` has **16 shapes
   for 55 frames** and renders as one ellipse plus a few blobs — that is not representative
   production roto and should not be in a "representative subset".

## 6. Proposed next step (Step 1 of the docs, scoped)

Build the canonical roto IR + deterministic renderer against these six shots, with acceptance
criteria we can actually measure today:

- IR covers: B-spline / Bézier / X-spline, closed fills **and** open strokes with width, per-shape
  opacity animation, layer hierarchy with 4×4 transform keys, Add/Subtract, feather, motion blur flags.
- Renderer reproduces every delivered matte channel at **≥0.98 IoU** (currently 0.94–0.99 on
  filled regions with a deliberately naive renderer, and structurally correct but soft on hair).
- Emits per-shot JSON: shapes, class (persistent vs ephemeral, fill vs stroke), points, keys,
  hierarchy path, semantic label from the layer name.
- Ships the persistent/ephemeral split as a first-class field, so Tier 1 can train on the 3419
  persistent shapes without hair contamination.
