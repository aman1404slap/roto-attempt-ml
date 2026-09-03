# Implementation plan — clean-alpha spline roto

*Supersedes the EXR-supervised approach in [POC.md](POC.md). Written 2026-09-03.*

---

## The strategy in one line

Render clean alpha from the artist's own splines, feed it to a model, and get the splines
back. The delivered EXRs are reference only — they prove the renderer is right, and they
never enter training.

## What dropping EXR supervision buys

Three items on the ask-the-client list were EXR-side problems. They are now moot for v1:

| was blocking | why it no longer is |
|---|---|
| Missing Nuke layer→channel mapping (cost 14 of 23 deliverables) | Elements are selected from the .sfx layer tree; no channel identity needed |
| sh0230 project/matte disagreement (~20 frames) | Input is rendered from the project, so it agrees with itself by construction |
| Vendor colour transform on two shots' alpha | Only affects EXR decode, which is no longer in the training path |

`roto verify` still scores against the EXRs and that check stays — it is the evidence the
renderer is trustworthy, and it is what makes the clean alpha worth training on.

## Elements

**One top-level layer of one .sfx = one element.** Derived from the file, not from a
hand-maintained table (`roto elements`). Six shots carry **18** top-level layers.

Exclusion is by measurement, each rule naming the number it fires on:

| rule | threshold | fires on |
|---|---|---|
| `over_keyed` | keys/live-frame > 0.75 | MAT_0130/Red Matte (1.34), sh0230/L100 (1.59), sh0230/L110 (0.95) |
| `paint_strokes` | open-stroke fraction > 0.9 | sh0230/L110 (1.00), nfl_0080/MB 2 (1.00) |
| `too_few_shapes` | < 3 shapes | FAM_0060/red (2) |

The `over_keyed` threshold reads a real bimodality rather than cutting an arbitrary tail:
14 elements sit at 0.03–0.59 and four sit at 0.95–1.59, with no continuum between.

**Result: 13 target elements, 2,753 shapes, 25,127 artist keys — across 4 shots, not 6.**
sh0230 and MAT_0130 lose every one of their layers. See *Open items*.

## Repository layout

```
src/roto/
  ir.py            canonical IR — the contract every reader/writer/renderer converts through
  sfx/read.py      .sfx → IR, all four Silhouette dialects
  sfx/write.py     IR → .sfx — seam only, deliberately not implemented (see below)
  sfx/json_ir.py   IR ↔ readable JSON
  render/          deterministic rasteriser
  matte/exr.py     EXR decode                  ─┐ verification path;
  eval/            channel assignment, metrics  ┘ not in the training path
  data/            shot discovery; recovered EXR channel map (used by verify only)
  dataset/         manifest → clean alpha + spline program per element
  program/         spline program ↔ dense arrays, and back
```

```
roto inventory | parse | verify | verify-all     renderer + EXR confidence check
roto elements                                    list training elements, no EXR
roto dataset <out>                               render clean alpha + program
```

## Version management

**Git for code, directories for data, neither for the other.**

- **Code** — one repo, linear history, tagged `v1-poc` and `v2-100shots`. Never fork v2 into
  a `v2/` folder: copied code directories diverge silently and you lose the ability to diff
  what actually changed between versions.
- **Datasets** — `datasets/<name>/v001/`, immutable, never overwritten. Each carries a
  `manifest.json` with a sha256 per file **and the git SHA of the renderer that produced it**.
  Bump to `v002` on any change. That one field is what keeps a v1 number reproducible after
  the renderer moves under v2.
- **Runs** — `runs/<date>_<slug>/` with config, metrics, checkpoint. Disposable.

Git is useless for 160 MB zips and 1,200-frame PNG sequences; a hash manifest gives
reproducibility without DVC's overhead at this scale. Both generated trees are gitignored.

## Algorithm

Two heads, and the second is deliberately not a neural net.

**Geometry — DETR-style set transformer.** CNN encoder (ResNet/U-Net) over the alpha crop →
transformer decoder with learned shape queries → per-shape B-spline control points, matched
to artist shapes by Hungarian assignment. Set prediction is the right frame because shape
count is variable and shapes are unordered. PyTorch.

Sizing caveat: elements run from 4 shapes to 1,036, so a fixed query count over a whole
top-level layer is wasteful at one end and expensive at the other. Predict per **shape
group** (the leaf layer that directly holds shapes) instead — 387 groups across the archive,
median 3 shapes, p90 35. The element stays the dataset unit; the group is the model unit.

**Motion — 6-DOF affine per shape group.** Measured: layer transforms are always affine,
never perspective, and shared per group (1,310 shapes → 64 tracks on the old set). Per-shape
prediction would be ~20× redundant and would let the model disagree with itself about the
motion of one rigid object.

**Keyframes — dynamic-programming knot selection, not classification.** This is the load-
bearing opinion. Per-frame BCE over "is this frame a key" is what produced precision
0.21–0.44 at recall 0.75–1.00 — firing on ~2.5× too many frames. It is structurally doomed:
it makes an independent decision per frame when the artist's keys are a *jointly optimal*
sparse set. Reframed as curve simplification — given a dense point track, find the k knots
whose interpolation best reproduces it — a DP solves it exactly in O(T²k), with no training
and no threshold to tune. Learn the error tolerance / k, not the individual key decisions.

## Phases

| # | step | gate |
|---|---|---|
| 0 | Build all 13 elements to clean alpha + program | every element builds; manifest hashed |
| 1 | Representation round-trip | `decode(encode(p))` renders to the element's own alpha ✅ **done** |
| 2 | Geometry only, GT shape count, single frames | control points < 2 px on held-out frames |
| 3 | DP knot selection on GT tracks | key F1 vs artist keys — beat 0.33–0.62 |
| 4 | Joint, **no per-element identity** | decoded program renders back ≥ 0.99 |
| 5 | `.sfx` writer + Silhouette validation | an artist opens and edits it |

**Run phase 3 first if you want the fastest read on feasibility.** It needs no model — it can
run today against ground-truth point tracks, and it is the half of the problem that is
actually unsolved.

**Phase 4 must carry no per-element embedding, bias table or lookup.** The earlier gate hit
0.000 px point error partly via learned per-element bias tables. That is memorisation by
construction; a great score on 13 elements with it tells you nothing about 100 shots.

## Out of scope for v1, and why the seam exists

`sfx/write.py` raises `NotImplementedError` and documents what a real writer needs. **There is
no ML in writing a .sfx** — the model's job ends when it produces a `RotoDoc`, and everything
from there to a file an artist opens is deterministic serialisation. `roto.program.decode`
already returns a `RotoDoc`, so the model is already finished; adding .sfx output later
changes nothing upstream. The blocker is not code — it is that no file we write has ever been
opened in Silhouette, and validating one needs a seat.

## Known limits

**v1 measures expressiveness, not roto skill.** Input and target are exactly consistent — the
alpha is a render of the answer. No noise, no ambiguity, one right answer. That is the correct
first step, but v1's number is not a production number. In deployment the input will be a
plate or an AI matte with fuzzy, wrong edges, and that gap is deliberately deferred.

**Motion blur, feather, and single-frame paint-stroke hair** are carried through as fields but
are not prediction targets. Blur and feather are render settings, not spline structure — a
blurred target teaches the model to bend geometry to compensate for a shutter.

## Open items

1. **v1 covers 4 shots, not 6.** Excluding sh0230 (both layers) and MAT_0130 removes two
   shots entirely. Confirm that is acceptable, or relax one rule.
2. **FAM_0060/green_2 is borderline** — open-stroke fraction 0.80 against a 0.90 threshold.
   216 of its 269 shapes are open strokes. It is currently a target.
3. **Production input format** — plate or AI matte? Decides whether v2's 100 shots need
   plates delivered alongside, which is a long-lead ask.
4. **Silhouette seat** — still the only blocker with no workaround, now scoped to phase 5.
