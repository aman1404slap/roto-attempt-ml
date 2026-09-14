# What we got, and what we picked

## The two words you need

**Layer** — a thing the artist named and worked on. "The girl", "the car", "her hair".

**Shape** — one outline inside that layer. A layer might be one shape, or two hundred.

A layer is what gets delivered. Shapes are what it's made of.

---

## What arrived

**50 folders.** One per shot of footage.

Inside each: the artist's tracing work, the black-and-white cut-outs it produced, and the
original video.

## The messy part

Each folder had about **eleven copies** of the same tracing work. The software auto-saves as the
artist goes, so you get snapshots from throughout the session. Only one is the finished piece.

We had to work out which. Every easy method was wrong:

- **Dates?** All identical — copying the files flattened them.
- **Filenames?** They lie. One folder's long, official-looking filename is a 77 KB early save;
  the real one is called `project.sfx` and is ten times the size.
- **Biggest file?** Wrong in about **30 of the 50 folders** — because artists often *delete*
  shapes near the end. The finished work is frequently smaller than a halfway save.

So we wrote a rule: take the file called `project.sfx` (47 of 50 have one). If there isn't one,
check each candidate against the delivered cut-outs and take whichever matches best.

## Everything we had, once we'd picked one file per folder

| | |
|---|---|
| shots | **50** |
| layers | **145** |
| shapes | **27,120** |
| frames | **7,592** |

The range is enormous. **Shapes per shot go from 1 to 9,094.** One single layer in this delivery
has more shapes in it than everything we're currently training on, several times over.

---

## What we left out

**Four folders are broken.** One won't open at all — it has a shape that changes its number of
corners partway through, which our system doesn't handle. One has no cut-outs, so there's nothing
to check our work against. One has almost nothing in it. One won't decode.

**Three folders are enormous.** One has 9,077 shapes by itself. Another is a single layer of
9,008. Another is 8K resolution across 393 frames. If we included these, they'd swamp everything
else — any result would really just be a result about those three.

That leaves **43 usable folders**.

---

## What we picked

**10 shots. 21 layers. 875 shapes.**

| | picked | available | share |
|---|---|---|---|
| shots | 10 | 50 | 20% |
| layers | 21 | 145 | 14% |
| **shapes** | **875** | **27,120** | **3%** |

Here they are, smallest layer to largest:

| shot | frames | layers | shapes in each |
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

---

## Why so little? Four reasons.

**We picked for variety, not size.** The smallest layer has 7 shapes; the biggest has 247. That's
a 35× range. When something fails, we want to see *what kind* of thing it fails on — simple or
complicated. A bigger but narrower set would tell us less.

**Everything would be 30 times the work.** The model learns each shape individually, one slot per
shape. 875 slots is manageable. 27,120 is not — certainly not before we know the approach works
at all.

**It has to be fast.** A full training run takes 22 minutes on my laptop. That means I can try
something, be wrong, and try again the same afternoon. With everything, that becomes days per
attempt, and we'd learn far less per week.

**Every folder had to pass a check first.** For each layer we verify the curves actually
correspond to the right cut-out. This matters more than it sounds — layer names and cut-out
channels don't reliably line up across this delivery, and a mismatched pair would quietly teach
the model to draw the wrong thing while every number still looked healthy. All 10 passed.

---

## Two of the ten are hidden

**`ts_021555` and `ts_021182` are kept away from the model entirely.** It never sees them.

They're the exam. Without them we couldn't tell whether the model learned something general, or
just memorised the specific shots it was shown. With them, we get an honest answer from the very
first run rather than a claim bolted on later.

Within the other eight, every seventh frame is also held back, for the same reason at a finer
grain.

**So of 21 layers: 17 are used for learning, 4 are held back.**

---

## What we deliberately never use

**The video footage.** All 50 `.mov` files sit unused. The model only ever sees the
black-and-white cut-out — never the actual picture, faces, colour or background.

**The motion-blurred cut-outs.** Every shot ships both a clean version and a blurred one. We only
use clean. Blur is a later problem.

**The artist's delivered cut-outs, as the thing to copy.** This one surprises people. We use them
to *check* our work — to confirm our version of the picture matches the artist's, and to confirm
which layer belongs with which cut-out. But the thing the model is trained against is a cut-out
we generate ourselves from the artist's own curves.

The reason is worth understanding: if we trained against the artist's delivered file, the model
would be asked to reproduce things our software can't draw — their motion blur, their edge
softness, their filters. A *perfect* answer would then still score less than 100%, and we could
never tell "the model is wrong" apart from "our drawing tool is wrong". Generating it ourselves
means a perfect answer scores exactly 100%, so every point below that belongs to the model.

---

## In one paragraph

> 50 shots arrived, containing 483 project files — but only one per shot is the artist's finished
> work; the rest are auto-saves. Taken properly, that's 145 layers and 27,120 individual curves.
> Four shots are broken and three are far too big to start with. From the remaining 43 we picked
> **10 shots — 21 layers, 875 curves, about 3% of the total** — chosen to cover simple layers and
> complicated ones so that failures tell us something. Two of those ten are hidden from the model
> so we always have an honest test. The footage and the blurred versions go unused on purpose.
> The other 90% isn't rejected — it's waiting, and it comes in once we know the method works.
