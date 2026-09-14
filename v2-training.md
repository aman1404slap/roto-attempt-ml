# How v2 learns — the S0 training run

Companion to [v2-training.jpg](v2-training.jpg). The picture shows it; this explains it.
Governed by [v2_charter.md](v2_charter.md) §4 (the stage ladder) and §5 (the gates); progress in
[v2-tracker.md](v2-tracker.md).

Every number here is from the real run on `datasets/v003`, two seeds at 40,000 steps each.

---

## The one-line version

Show the network a silhouette and its two neighbouring frames, let it guess where every control
point goes, punish it **in pixels** for being wrong, and only then ask which of those frames
deserve to be keyframes.

---

## 1. What goes in

Three consecutive mattes, 256×256, stacked as channels. Nothing else — no plate, no colour, no
layer names.

**Why three and not one.** With a single frame the model's error is independent from frame to
frame, which shows up as jitter: a control point that wobbles by a pixel even when the shape is
still. Showing it the neighbours lets it steady its own hand, and costs one convolution's worth
of input channels.

**Why they are re-aligned first, which is the non-obvious part.** Each frame is rendered in its
*own* crop window that tracks the layer, and that window twitches — about 1 crop pixel per step,
4 px at worst. That is **larger than the jitter the window exists to remove**. Stack three
neighbours raw and you hand the network three copies of the shape displaced by more than the
quantity under test, which is why an earlier round concluded "the temporal window doesn't help"
and was measuring the wrong thing. Each neighbour is now shifted into the anchor's window
before stacking, sub-pixel and bilinear.

Panel 2 of the figure shows this directly: the left image is the three frames as stored,
composited to R/G/B, and the colour fringes *are* the disagreement. The right image is the same
three after alignment.

## 2. What else the model is handed — the training wheels

At stage S0 the model does **not** have to work out the structure. It is given:

| given | for this element |
|---|---|
| how many shapes | 139 |
| how many control points each | 4–68 |
| open or closed | 133 closed, 6 open |
| which shape is which | 139 learned identity rows |
| which shapes move together | 40 motion groups |
| which frames each shape is alive on | given (50% alive) |

This is why S0 is honestly called **reconstruction**, not roto from scratch. Charter §4 removes
these one rung at a time — key timing at S1, lifespans and point counts at S2, the whole
breakdown at S3.

**The identity rows are the thing to watch.** They are a lookup table with one row per
`(element, shape)`, and they are pure memorisation capacity. 875 rows for the ten-shot subset;
**27,120** if we used all fifty shots, with one element alone wanting 9,008. That is the
constraint that makes S3's dynamic queries load-bearing rather than a nice finish.

## 3. The network

```
3 aligned mattes ──► CNN encoder ──► 16×16 = 256 image tokens
                                            │
shape queries  (identity row + a descriptor:│how many points, closed, coords)
        │                                   │
        └──►  ×3 [ self-attend among shapes, then cross-attend to the image ]
                                │
                ┌───────────────┴───────────────┐
         control points                   transform track
      (per shape, crop space)     (8-number projective, per group)
```

Two design choices that were measured rather than chosen, and both cost something when done the
other way:

**Self-attention among shape queries comes before they look at the picture.** Without it, two
queries claim the same contour and the result is visible scribble — 0.97 soft IoU at ≤75 shapes
against 0.73–0.78 at 592 and 1036. Letting the shapes negotiate with each other first fixes it.

**The point head is a direct projection**, not a per-slot lookup. The cheap version —
`MLP([query, slot]) → 2` — is a rank-limited outer product and it stalls: 93 point slots on one
shape cannot be separated by two 192-dimension vectors.

**Coordinates are predicted in crop space** — `[0,1]` across the picture the model was given —
and converted back to the IR's native coordinates afterwards, exactly and in closed form.
Predicting the native coordinates directly stalls at ~260 px, because they are absolute document
positions whose range is up to 8× the crop the network can actually see.

## 4. What it is punished for

Four terms. Three of them are in crop pixels, and that is deliberate.

| term | what it measures | over the run |
|---|---|---|
| **point** | where each control point sits | 47.1 → **0.61 px** |
| **transform** | where the layer's own motion puts it | 35.3 → **0.62 px** |
| **curve** | the drawn curve, not just the dots | 21.8 → 0.26 |
| **temporal** | no jitter between consecutive frames | 2.6 → 0.42 |

**Why pixels matter.** If the transform term were measured on the matrix entries it would be
unitless, and its weight would be a free parameter somebody had to guess — an earlier version
used 20, picked so the term was merely audible. Measured as *where five probe points land*, a
pixel of motion error costs exactly what a pixel of point error costs, and the weight means
something.

The **curve** term exists because the point term alone grades the handwriting rather than the
picture: an artist's control polygon and ours can differ while drawing the same curve. Charter §4
promotes this term to primary at S2 and demotes dot-matching to 0.25×, for exactly that reason.

## 5. How one number teaches 2.98 million parameters

Yes, this is a neural network — **2.98M parameters**. But the pipeline is a *hybrid*, and only
part of it learns, which is worth being precise about because it is what makes a failure
diagnosable.

### What the parameters are

Three groups: the convolution filters in the encoder, the attention weight matrices in the three
decoder blocks, and the projection matrices in the two heads. Plus the identity table — 875 rows
of 192 numbers, one row per `(element, shape)`.

They start random, with one deliberate exception. The point head's output bias starts at **0.5**,
so the first guess puts every control point at the centre of the crop rather than in the
top-left corner; otherwise the first thousand steps are spent translating rather than shaping.

### The loop

```python
opt.zero_grad(set_to_none=True)
loss.backward()                                     # d(loss)/d(theta) for all 2.98M
torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
opt.step()                                          # AdamW nudges every parameter
```

Forty thousand times. Each iteration draws six frames from **one** element — one, so the shape
queries in a batch are consistent — runs them forward, and collapses everything into a single
number.

### The loss, with S0's weights substituted

```
total =       point_px                # L1 on control-point position
      + 0.25 x affine_px              # where 5 probe points land under the predicted motion
      + 0.5  x curve_px               # the drawn curve, not the control polygon
      + 1.0  x temporal_px            # frame-to-frame consistency
      + 0.25 x affine_temporal_px     # the same, for the motion track
```

The first term in full ([train.py](src/roto/v2/train.py#L412)):

```python
mask     = point_mask & live                        # real points only, and only while alive
point_px = (abs(pred - target) * mask).sum() / mask.sum() * 256
```

Three things in that line matter:

- **L1, not squared.** A squared error would let a handful of wild points dominate the gradient,
  and roto has exactly that failure mode at shape boundaries.
- **The mask.** A shape with 16 points sits in a 68-slot array. Without masking, the model would
  be graded on 52 slots that do not exist.
- **The `* 256`.** Predictions live in `[0,1]` across the crop, so this converts to **crop
  pixels** — which is what makes the weights above meaningful ratios rather than arbitrary
  scalings. `0.25 x affine_px` says *a pixel of motion error costs a quarter of what a pixel of
  point error costs*, which is a claim somebody can argue with. The old document-space weight of
  20 was not.

### Why one number can teach all of them

The whole forward pass is differentiable — convolutions, attention, the crop-space projection,
the polyline evaluation inside the curve term. So the loss is a smooth function of the
parameters, and the chain rule gives the gradient: a vector in 2.98-million-dimensional space
pointing in the direction that increases the loss fastest. Step a little the other way, and
repeat.

The rest is bookkeeping, and each piece earns its place:

- **AdamW**, lr `3e-4`, weight decay `1e-4`. It keeps a running mean and variance of each
  parameter's gradient and scales that parameter's step by them, so a parameter with
  consistently small gradients still moves and a noisy one does not thrash.
- **200-step linear warmup.** Early gradients are large and largely wrong; a full-size step on
  them can wreck the initialisation.
- **Gradient clipping at norm 1.0.** If one batch produces an enormous gradient it is rescaled —
  insurance against a single bad sample undoing thousands of steps.

The visible result is the curve in panel 6: mean control-point error **47.1 px to 0.61 px**.
Literally, the average control point started 47 pixels from where it belonged and finished 0.6
pixels away.

### The part that does not learn

Three stages have **no parameters and no training at all**:

1. **The keyframe picker** — dynamic programming. It finds the provably optimal set of keys for a
   given tolerance, so there is nothing to learn; there is an exact answer. (S1 lets the network
   *bias* it, but the DP itself stays.)
2. **The renderer** — fixed, and the thing the exactness ledger pins to zero error.
3. **Crop space to native coordinates** — closed-form algebra, exact.

That split is deliberate: the network does the part that is genuinely perceptual — *where in this
silhouette does control point 7 of shape 32 belong* — and classical algorithms do every part with
a known correct answer.

It is also why §7's result is diagnosable rather than just disappointing. Geometry is at 0.52 px,
so the learned part is working; key economy is at 0.31x, and that is the *unlearned* part being
mis-tuned. Different problems, different fixes, and one of them is a knob.

## 6. Then — and this is a separate stage — the keyframes

The network predicts **every frame**. It never decides a keyframe.

A dynamic program then walks the predicted track and spends a key only where interpolating from
the previous one would drift past a tolerance. This is what makes the output *editable* rather
than merely accurate: a model that keys all 56 frames reproduces the matte perfectly and is
useless to an artist, because they would have to delete 40 keys before they could touch it.

Two consequences worth knowing:

- **Key economy is reported beside accuracy, never after it.** The gate is 0.75–1.3× the artist's
  key count.
- **The tolerance cannot be tuned below the model's own noise.** A predicted track carries jitter
  of roughly the model's point error; asking the search to reproduce it within 0.1 px forces a
  key on nearly every frame — measured at 16.5× the artist's count.

## 7. What we got, and what we did not

One element (`ts_021243__Red`, 139 shapes), seed 1, at the **default** operating point:

| | want | got | |
|---|---|---|---|
| soft IoU | ≥ 0.97 | 0.9650 | ✗ |
| worst frame | ≥ 0.90 | 0.9305 | ✓ |
| point error | ≤ 1.2 px | **0.52 px** | ✓ |
| point p95 | ≤ 3 px | 1.43 px | ✓ |
| key economy | 0.75–1.3× | **0.31×** | ✗ |

**The geometry is essentially solved and the keyframes are not.** 0.52 px mean point error is
well inside the gate. The miss is key economy: at the default tolerance the search keys *too
little*, less than a third of what the artist spent.

That is a knob, not a retrain. The operating point — tolerance, smoothing window, filter choice,
key-value refit — has not been swept on this model. An earlier round measured a better operating
point worth +0.0035 soft IoU and +0.027 key F1 **for no retraining at all**. It is deliberately
deferred (tracker Step 2c) so that the anchor and the tuning do not move together, which would
leave neither readable.

## 8. Why two seeds

Charter L4: two seeds per quoted number, and a difference smaller than the gap between them is
weather.

The two curves in panel 6 lie almost on top of each other. That is itself the result — it means
a later change worth 0.003 would be a real difference rather than luck, and it is what lets the
rest of this project stop hedging.

## 9. What this run is, and is not

**It is the anchor.** Every later change — S1's key head, S2's lifespans, S3's dynamic queries,
a wider shot set — is measured against it.

**It is not the finish line**, and three things are deliberately not in it:

- the **operating point** is unswept (§7 above)
- `sample_weight` is at `sqrt`, the rung an earlier round's failing gates named
- motion is **teacher-forced** at the default scoring setting; the de-teacher-forced number is
  one flag away

None of those change the anchor, and each is one flag. That is the point of freezing the
configuration first.
