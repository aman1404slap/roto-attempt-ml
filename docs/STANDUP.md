# Standup — Spline Roto

*Where the project stands, end of August 2026. Written for someone who knows both a roto
bay and a codebase, and wants the shape of it rather than the detail. Detail lives in
[POC.md](POC.md) and [FINDINGS.md](FINDINGS.md).*

---

## What we were handed

Six shots. Each one an artist's Silhouette project plus the matte EXRs that were delivered
from it. No plates. No Nuke script. Four different Silhouette versions across six files.

That's it. And it's enough, because of one property: **the artist file is the answer, and the
matte is the question it already answers.** If we can regenerate the question from the answer
ourselves, we have supervised training data without needing footage from anyone.

That bet is what the last two weeks tested.

## What we did

**We read the files.** Both container formats, all four Silhouette versions, no licence
needed — plain Python. 6,969 splines, 52,565 keyframes, 660 frames of finished production
roto. First surprise: they're **B-splines**, not Béziers. The reference docs assumed Bézier
and picked tooling on that basis. It matters because whatever we hand back has to be in the
curves the artist actually works in.

**We redrew them ourselves and checked against the delivered mattes.** This is the step that
either validates the whole premise or kills it. Our renderer reaches **0.9843–0.9980** overlap
on every channel we certified. On the main test element, at full res, **0.9947** — and every
single disagreeing pixel is within one pixel of the outline. Not one wrong pixel in the
interior. The shapes are right; only the anti-aliased edge differs.

Getting there meant getting three conventions exactly right, each of which renders
*plausible* garbage when wrong: coordinates normalised by frame height from the centre,
transform matrices composing up the layer chain in the right order, and shapes being switched
on and off by **opacity hold keys** rather than being added and removed. That last one was
worth three full points of accuracy and looked like a geometry bug, not a visibility bug.

**We recovered the mapping nobody sent us.** The .sfx has layers; the EXR has R/G/B. Nothing
says which made which — that lived in a Nuke script we don't have. We found it by brute-force
search, and the search beat our own hand-guesses twice. Then we checked the result into the
repo as a file, because it's recovered knowledge rather than archive data and it should be
reviewable in a diff.

Result: the six shots are really **23 deliverables**, of which **9 are trustworthy today**.
Ten are blocked on that missing mapping. Three are one shot whose matte was rendered from a
*different save* than the archived project — hair animating twenty frames past the last key
in the file we were given. That's a data QC catch, and at archive scale it needs to be an
automated ingest gate.

**We packaged the 9 into training examples.** Clean matte rendered from the artist's own
splines — deliberately not the delivered EXR, so the model isn't asked to explain a vendor's
compositing quirks alongside the artist's craft. 1,200 frames, 1,310 shapes, **10,957 artist
keyframes**, 23 MB.

**We proved the training format loses nothing.** Artist program → our arrays → back out →
draw → compare. **1.000000 overlap, all nine elements.** Exactly, not approximately. This is
the ruler-before-the-measurement step, and it's now an automated test, so it can't silently
break later.

**We put a tiny model on it.** Not to be good — to prove a gradient can travel from pixels to
a spline program and back. It reproduced three artist programs at **0.000px** point error,
**F1 1.0000** on keyframe timing, and the programs it emitted drew back to the matte at
**0.9984–1.0000**.

## What we learned

**Geometry is close to solved. Timing isn't.** With the memorisation crutch removed, so the
model has to actually look at the picture: control points land within 0.1–1.9 px and shape
lifespans are near-perfect — but keyframe **precision** is 0.21–0.44 against recall
0.75–1.00. It finds the artist's keys, then fires on about **two and a half times too many
frames besides**. That's over-keying: the exact thing that makes a roto file unusable to the
person who has to open it. Geometry within 2px while the drawn result sits at 0.68–0.96 says
it plainly — **the keys, not the shapes, break the picture.**

**Trackers live on the leaves.** Named group layers (`core`, `face`, `body`) are never
tracked; the tracked layers are the ones directly holding shapes. So motion is one tracker per
shape group: 1,310 shapes resolve to just **64** transform tracks. That's the shape the model
should predict, and we'd have got it wrong by guessing.

**Three measurements were quietly wrong before we checked them.** Two shots ran their alpha
through a colour transform, so 50% coverage is stored as 0.214. Scoring at reduced resolution
understates every number by up to 0.02, all of it on the outline, so it reads as our renderer
being sloppy — it had already mis-tiered our own named target element (0.948 → really 0.9858).
And a keyframe head trained the obvious way predicts *no keyframes at all* while scoring 94%
accuracy, because only 6% of frames carry a key.

**The best-scoring element is the worst example.** `MAT_0130` renders at 0.998 and keys 1.67
times per live frame, against 0.03–0.64 everywhere else. Smallest, cleanest, most tempting —
and it would teach precisely the wrong lesson.

## Decisions taken

**Motion blur, feather and single-frame paint-stroke hair are out as targets, in as fields we
carry faithfully.** Blur and feather are render settings, not spline structure — a blurred
target teaches the model to bend geometry to compensate for a shutter. The hair strokes have
one keyframe by construction, so mixing them in teaches key-every-frame. The handoff's own
Phase 0 already skips both, so this confirms rather than changes the plan.

## Where it goes next

Beat the measured baseline: keyframe F1 0.33–0.62, drawn result 0.68–0.96. Attack
**precision**, not recall — the keys are being found, there are simply too many of them. Then
add the render-consistency loss at non-key frames, which is the one planned component with no
evidence behind it yet.

## What we need

| | why |
|---|---|
| **One Silhouette seat** | Nothing we write has ever been opened in Silhouette. The deliverable is a file an artist edits. This is the only blocker with no way around it. |
| **The layer → channel mapping** | Render-node settings, especially for `sh0260`. Recovers up to 10 more elements — more than doubling usable data — for what is probably a five-minute lookup. |
| **A newer `sh0230` save** | Its matte and its project disagree by ~20 frames. |
| **RGB plates for a subset** | Needed for the RGB branch and Tier 2. Not blocking today. |

---

*125 tests green. Every number above is reproducible: `roto verify-all`, `roto packets`,
`roto overfit`.*
