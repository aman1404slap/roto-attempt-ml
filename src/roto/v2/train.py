"""v2 training loop, charter S4 stage **S0**: geometry + motion, structure given.

Ported from ``roto.model.train`` at v2 Step 2. What changed is the defaults and the vocabulary;
what did not is any loss term, because Step 2 exists to produce an *anchor*, and an anchor
measured with a term nobody has measured before anchors nothing.

**The defaults are v1.1's ``final_long_v2`` / v1.3's ``HEADLINE``, unchanged**: a 3-frame
aligned window, self-attention among shape queries, the curve term at 0.5, the temporal term at
1.0, sqrt element weighting, and the transform head predicting the 8-number local-to-crop
projective map at weight 0.25. That configuration is the one v1.2 recommended building on and
the one v1.3 shipped, so a v2 number read against it differs by the **data**, which is the
comparison Step 2 is for.

Two things were dropped rather than carried, both of them v1-comparability machinery with
nothing left to be comparable to: ``affine_space='doc'`` (the lossy 6-number document-space
target, kept in v1 only so v1's own transform rows still reproduced) and ``use_split=False``
(kept in v1 for the single row that had to read against a pre-split result). v2's dataset
records its split at build time and there is no pre-split v2 result, so a run that could ignore
the split would only ever be a mistake.

``sample_weight`` stays configurable and stays at ``'sqrt'``. v1.3's gates named it as the next
rung -- it is what de-prioritises a small element, and three of that round's four failing gates
traced to one 16-shape layer -- but changing it here would mean the anchor and the rung moved
together, and neither could then be read.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..program import AFFINE_DOF, PROJ_DOF
from .losses import PolylineMaps
from .losses import curve_loss as polyline_loss
from .traindata import ElementData, canonical_order, load_dataset, query_desc
from .net import SLOTS, TABLE, RotoNetV2


DOC, CROP = 'doc', 'crop'
AFFINE_SPACES = (DOC, CROP)


RANDOM, PAIRS, RUNS = 'random', 'pairs', 'runs'
SAMPLING_MODES = (RANDOM, PAIRS, RUNS)


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(slots=True)
class TrainConfig:
    steps: int = 40000
    batch: int = 6
    lr: float = 3e-4
    dim: int = 192
    depth: int = 3
    affine_weight: float = 0.25
    """Weight on the transform term.

    With ``affine_space='doc'`` this has to be a lift: the entries are O(1) and unitless while
    point error is in pixels, so without one the term is noise next to the geometry term and
    the track never converges. 20 was picked to make it audible, not measured.

    With ``affine_space='crop'`` the term is already in crop pixels, so 1.0 means "a pixel of
    transform error costs what a pixel of point error costs" -- the same reading
    ``temporal_weight`` has. Leaving it at 20 there would price the transform above the
    geometry by 20x."""
    affine_space: str = CROP
    """``'doc'`` (v1: 6 document-space matrix entries, L1 on the entries) or ``'crop'``
    (v1.1: the 8-number local-to-crop projective map, L1 on where it puts five probe points).

    Two separate faults are fixed by the same switch, and both were measured:

    * The document-space target asks the head for translation in normalised *document* units
      from a crop that shows none of that -- the framing that stalled point regression at
      260 px until ``local_to_crop`` fixed it, repeated verbatim.
    * The 6-number form is **lossy**. Two layers carry perspective, and round-tripping the
      artist's own track through 6 numbers renders ``FAM red_1`` at 0.9100 instead of 1.000
      and displaces ``Layer_52`` by 23.8 crop px. That is a ceiling the head could not have
      beaten however well it predicted, and it is most of why v1's head read 0.756.
    """
    in_frames: int = 3
    """Consecutive alphas stacked as input channels. 1 reproduces v1."""
    align_window: bool = True
    """Warp each window neighbour into the anchor's crop window before stacking.

    Off reproduces v1.1's first ladder, where the window was tested *misaligned* and read as
    worthless. See ``data.ElementData.window``: the crop offset twitches 1.09 crop px per step
    on average, which is larger than the 0.77 px of jitter the window was meant to remove, so
    the three channels disagreed about where the shape was by more than the quantity under
    test. Only meaningful when ``in_frames > 1``."""
    self_attn: bool = True
    """Self-attention among shape queries. See ``net.SelfBlock``."""
    affine_depth: int = 0
    """Depth of the transform head's own decoder. ``0`` means ``depth``, which is v1's.

    The v1.1 handover's decision tree says that if the crop-space head still falls short,
    "give the head its own small decoder depth before giving up". It already has its own
    decoder -- ``net.RotoNet.gblocks`` is a separate cross-attention stack, sharing only the
    encoder -- so what this knob actually tests is whether the head is *capacity*-limited,
    and the alternative hypothesis is that it is schedule-limited like everything else here:
    at 40k the transform term is still falling (0.858 crop px and descending, ``final_long_v2``).
    Measured as one rung rather than argued."""
    key_weight: float = 0.0
    """Weight on the key-timing term. ``0`` leaves the head off entirely, which is v1.2.

    Unlike every other term here this one is not in crop pixels and cannot be made so -- it is
    a cross-entropy on a probability. So the weight is a real choice rather than a unit fix,
    and it is set by what the term is worth: at 1.0 the key term is comparable in magnitude to the
    point term at convergence (point error lands near 1 crop px, and a class-balanced BCE at
    this archive's 9.6% positive rate lands in the same order), which is the ratio that lets
    the head learn without the geometry giving anything up. Measured as a rung rather than argued -- see
    ``scripts/train_v13.py``."""
    key_balance: str = 'per_element'
    """Whether the key term's positive weight is one number or one per layer.

    ``'per_element'`` is the default and the fix the first probe asked for; ``'global'`` is what
    that probe ran and is kept so the comparison stays reproducible. See
    :func:`key_positive_rates` for what the probe measured and why one global weight is
    exploitable: key density ranges 13x across these layers, and learning that spread scores
    well on any metric that does not net it out."""
    key_pos_weight: float = 0.0
    """Positive-class weight for the key term. ``0`` means *measure it from the dataset*.

    The archive keys 9.6% of live shape-frames, so an unweighted BCE is minimised at
    "never a key" -- which scores 0.90 accuracy and F1 zero. This is the standard correction,
    ``(1 - p) / p``, and it is measured rather than tuned because it is a property of the
    data. Recorded on the run summary so a run can be read against the rate it was trained
    at."""
    curve_weight: float = 0.5
    """Weight on the polyline term, relative to the point term. 0.5 was the starting point;
    the point term is kept because it is what pins down a control polygon the artist can edit,
    while the curve term is what the render actually judges."""
    point_weight: float = 1.0
    """Weight on the dot-L1 term. ``1.0`` is S0 and S1, where the term is the reference unit
    every other weight is quoted against.

    Charter S4 demotes it at **S2**: as the model starts inventing structure, matching the
    artist's dots one for one stops being the right question and the curve the dots draw
    becomes it (charter L2). Plan section 5 sets the demotion at 0.25x with the curve term
    promoted to primary.

    Split out as its own weight -- rather than scaling the other three up -- so the change is
    one recorded field on the run rather than a reinterpretation of every other number, and so
    ``point_px`` stays readable in crop pixels in the log while its *contribution* changes.

    Measured before the S2 heads exist rather than bundled with them (rung ``s2a``): at S2
    shape count and identity are still given, so index-wise correspondence is still exact and
    the dot term is still legitimate. The demotion is preparation for S3's weakened
    correspondence, and at S2 it can only cost geometry."""
    temporal_weight: float = 1.0
    """Weight on the frame-to-frame consistency term."""
    alive_weight: float = 0.0
    """Weight on the lifespan term, charter S4 stage **S2**. ``0`` leaves the head off, which
    is S0 and S1.

    Like the key term this is a cross-entropy rather than a distance, so the weight is a real
    choice and not a unit fix. 1.0 puts it at the same standing as the key term, which is where
    the S2 rung starts; the design note's stop condition lowers it once if geometry regresses
    beyond the seed spread, and then stops rather than searching."""
    alive_balance: str = 'per_element'
    """``'per_element'`` or ``'global'``, exactly as ``key_balance``, and for a sharper reason.

    The live rate spans **0.093 to 1.000** across v003's trained elements. With one global
    weight the cheapest thing for the head to learn is each element's own base rate, and a per
    ``(element, shape)`` query row is precisely the capacity to do it -- the same exploit the
    key head's first probe fell into, at a wider spread."""
    alive_pos_weight: float = 0.0
    """Positive-class weight for the lifespan term. ``0`` means *measure it from the dataset*.

    ``(1 - r) / r`` at v003's 0.627 live rate is 0.60 -- below 1, because live cells are the
    *majority* here. That is the opposite of the key term's situation and worth stating: this
    head's degenerate answer is "always alive", not "never", and the balancing is what stops it
    being free. An element whose shapes are always live has ``r = 1`` and no negatives at all;
    its weight is pinned to 1.0 rather than to the ``(1-r)/r`` formula's 0, which would zero the
    positive class and delete the element from the term."""
    count_weight: float = 0.0
    """Weight on the point-count cross-entropy, charter S4 stage **S2**. ``0`` leaves the head
    off.

    Reported, never gated -- see ``v2-s2-design-note.md`` section 2.4. A point count does not
    vary with the frame and the query row does, so the row can carry it exactly and accuracy
    here measures memory rather than perception. The head is built at S2 anyway because it and
    its decode have to exist and be debugged somewhere, and S3 -- where identity goes away and
    the number becomes real -- is not the place to be debugging it."""
    count_style_weight: float = 0.0
    """Weight on plan section 5's point-economy style statistic, ``|E[P] - P| / P``.

    Charter L2 asks for point economy as a *soft statistical* target rather than a hard match,
    so this is an L1 on the head's expected count rather than a second cross-entropy. The
    ``/ P`` is deviation 4.3 of the design note and is measured, not chosen: plan section 5's
    unweighted form prices one point the same at P = 68 and at P = 4, and the render does not.
    Dropping a point costs 0.0004-0.053 soft IoU on fifteen of v003's elements and 0.42-0.50 on
    the two whose smallest shape has four points, where a 4-point closed B-spline drops to a
    near-degenerate 3."""
    n_slots: int = 0
    """Charter S4 stage **S3**: share one bank of this many shape queries across every element,
    instead of one learned row per ``(element, shape)``. ``0`` keeps the per-element table,
    which is S0 through S2.

    256 covers v003, whose largest element has 247 shapes. It is a recorded **interface**
    number rather than a hyperparameter -- changing it changes what a checkpoint means -- and
    it is also the reason the deferred 9,008-shape shots re-enter at S3 rather than before: no
    fixed maximum accommodates them, which is a fact about the design and not about the data.

    Turning this on does three things at once, and they are inseparable by construction: the
    query stops carrying identity, the *count* stops being given (a slot the alive head never
    fires on is not a shape), and the shape-type descriptor leaves the input because it is
    per-(element, shape) information. See ``net.QUERY_MODES``."""
    give_point_count: bool = True
    """Whether ``n_points`` is fed to the query as part of ``desc``. ``False`` is S2's training
    wheel coming off.

    Removing it does **not** make the count unmemorisable -- the query row can still carry it,
    and the count loss will put it there. What removal does buy is that the count is no longer
    handed over on a path that bypasses learning entirely, which is charter L1: at S2 the
    declared given structure is shape count and identity, and a point count is neither."""
    holdout_every: int = 0
    """Withhold every Nth frame from training and score it separately. 0 holds nothing.

    Ignored when the dataset records its own split, which ``datasets/v002`` does and
    ``datasets/v001`` does not -- see ``data.ElementData.split`` and ``use_split``."""
    use_split: bool = True
    """Obey the split the dataset recorded at build time. ``False`` trains on everything.

    The default is to obey it, because a split that a run can quietly ignore is not a split.
    The opt-out exists for exactly one row -- the one that has to be comparable to a
    pre-split result -- and it is a *flag on the run* rather than a second dataset, so the
    two rows differ in one recorded field and share every alpha. A run that opts out says so
    on its own checkpoint (``split_source``), so a table can never mix the two silently."""
    sampling: str = 'pairs'
    """How a step's frames are drawn: ``'random'`` (v1), ``'pairs'``, or ``'runs'``.
    See :func:`sample_indices` -- the temporal term needs ``'pairs'`` to have anything to
    act on, and ``'runs'`` is kept only because its cost is worth having on record."""
    sample_weight: str = 'sqrt'
    """``'frames'`` (v1: proportional to frame count) or ``'sqrt'`` (sqrt(frames * shapes)).

    v1's weighting let the two giant layers dominate wall-clock while the tiny ones overfit.
    ``'sqrt'`` compresses that range; which one wins is measured, not assumed."""
    log_every: int = 500
    seed: int = 0
    warmup: int = 200
    device: str | None = None


@dataclass
class TrainState:
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def sample_indices(rng: np.random.Generator, pool: np.ndarray, batch: int,
                   mode: str = RANDOM) -> np.ndarray:
    """Frame positions for one step, drawn from the *training* pool.

    Three modes, and the choice is measured rather than assumed:

    * ``random`` -- v1's: ``batch`` distinct positions, unordered. Maximum gradient
      diversity, but adjacent frames essentially never co-occur, so a temporal term would
      see nothing to penalise.
    * ``pairs`` -- ``batch // 2`` random anchors, each with the frame after it. Keeps most
      of the diversity (three independent places in the shot rather than one) *and* gives
      the temporal term real adjacent pairs. This is v1.1's default wherever that term is on.
    * ``runs`` -- one contiguous run. The obvious way to get adjacency and **measurably the
      wrong one**: six consecutive frames of a roto layer are nearly the same picture, so a
      step learns far less than from six scattered ones. Measured at 12k steps against
      ``random`` at an identical seed and configuration -- see the sampling rows of
      ``v1.1/results/ladder.md``. Kept as a configuration so that comparison stays
      reproducible, not because it is ever the right choice.

    A held-out frame is skipped rather than borrowed, so ``pairs`` occasionally proposes an
    anchor whose successor was withheld; those are excluded when the pairs are built, and
    any non-adjacent pair that survives is dropped again by ``temporal_term``.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(f'unknown sampling mode {mode!r}, want one of {SAMPLING_MODES}')
    n = min(batch, len(pool))
    if mode == RANDOM or len(pool) <= 2:
        return np.sort(rng.choice(pool, size=n, replace=False))
    if mode == RUNS:
        if len(pool) <= batch:
            return pool.copy()
        s = int(rng.integers(0, len(pool) - batch + 1))
        return pool[s:s + batch]
    starts = pool[:-1][np.diff(pool) == 1]                # anchors whose successor is kept
    if not len(starts):
        return np.sort(rng.choice(pool, size=n, replace=False))
    k = min(max(1, batch // 2), len(starts))
    chosen = rng.choice(starts, size=k, replace=False)
    return np.concatenate([[s, s + 1] for s in np.sort(chosen)])


def _batch(el: ElementData, idx: np.ndarray, shape_base: int = 0, group_base: int = 0,
           in_frames: int = 3, device: torch.device | str = 'cpu',
           align: bool = False, give_point_count: bool = True,
           n_slots: int = 0, perm: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    a = torch.from_numpy(el.window(idx, in_frames, align))
    to = lambda t: t.to(device, non_blocking=True)
    # S3: targets are reordered into the canonical order the shared slots are addressed by,
    # and the *geometry* targets stay n_shapes wide rather than being padded to n_slots --
    # padding a (F, 256, 68, 3, 2) float array would cost 392 MB on a 94-frame element for
    # rows that are masked out of every geometry term anyway. Only the alive target is padded,
    # because that is the one head with something to say about an empty slot.
    sl = perm if perm is not None else slice(None)
    b = {
        'alpha': to(a),
        'points': to(torch.from_numpy(el.points[idx][:, sl])),
        'live': to(torch.from_numpy(el.live[idx][:, sl])),
        'affine': to(torch.from_numpy(el.affine[idx])),
        'proj_crop': to(torch.from_numpy(el.proj_crop[idx])),
        'probe': to(torch.from_numpy(el.probe_local)),
        'group_live': to(torch.from_numpy(el.group_live[idx])),
        'frames': to(torch.from_numpy(el.frames[idx].astype(np.int64))),
        'key_mask': to(torch.from_numpy(el.key_mask[idx][:, sl])) if el.key_mask.size
        else to(torch.zeros((len(idx), el.n_shapes), dtype=torch.bool)),
        'shape_ids': to((torch.arange(n_slots or el.n_shapes)
                         + (0 if n_slots else shape_base))[None].expand(len(idx), -1)),
        'group_ids': to((torch.arange(el.n_groups) + group_base)[None].expand(len(idx), -1)),
        'desc': to(torch.from_numpy(query_desc(el, give_point_count)[sl])[None]
                   .expand(len(idx), -1, -1)),
        'point_mask': to(torch.from_numpy(el.point_mask[sl])[None]),
        # The point-count head's target, ``(S,)``. Not per frame: a shape's point count is
        # fixed for its whole track -- ``ir.Shape.validate`` raises if it is not -- and that
        # invariance is the whole difficulty with the head. See ``count_term``.
        'point_count': to(torch.from_numpy(el.n_points_per_shape[sl].astype(np.int64))),
    }
    if n_slots:
        # The existence signal. A slot past this element's shape count is a slot with nothing
        # in it, and "never alive" is how the model is asked to say so -- which is why S3
        # needs no separate no-object class: the S2 lifespan head already carries it.
        alive = np.zeros((len(idx), n_slots), bool)
        alive[:, :el.n_shapes] = el.live[idx][:, sl]
        b['alive_target'] = to(torch.from_numpy(alive))
    return b


def temporal_term(pred: torch.Tensor, b: dict[str, torch.Tensor],
                  out_px: float) -> torch.Tensor:
    """``|dpred - dtarget|`` over adjacent frame pairs, in crop pixels.

    Only pairs one frame apart count, and only where the shape is live at *both* frames --
    a shape blinking on carries a step change the network should not be asked to smooth.
    """
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    ok = (b['frames'][1:] - b['frames'][:-1]) == 1                      # (B-1,)
    if not bool(ok.any()):
        return pred.new_zeros(())
    live2 = b['live'][1:] & b['live'][:-1]                              # (B-1, S)
    mask = (b['point_mask'] & live2[..., None, None]).unsqueeze(-1) \
        & ok[:, None, None, None, None]
    n = mask.sum().clamp(min=1)
    d_pred = pred[1:] - pred[:-1]
    d_true = b['points'][1:] - b['points'][:-1]
    return ((d_pred - d_true).abs() * mask).sum() / n * out_px


def apply_proj(params: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    """``(..., G, 8)`` projective params x ``(G, K, 2)`` local probes -> ``(..., G, K, 2)``.

    The 8 numbers are the varying entries of a row-vector 4x4 divided through by ``m33`` --
    see ``roto.program.PROJ_ENTRIES`` -- so for a point lifted to ``[x, y, 0, 1]``::

        u = x*m00 + y*m10 + tx      w = x*m03 + y*m13 + 1
        v = x*m01 + y*m11 + ty      out = (u/w, v/w)

    Written out rather than assembled into a matrix so it stays one fused expression on the
    GPU, and so the perspective divide -- the part the 6-number form threw away -- is visible.
    """
    x, y = probe[..., 0], probe[..., 1]                             # (G, K)
    m00, m01, m03, m10, m11, m13, tx, ty = [params[..., i:i + 1] for i in range(8)]
    u = x * m00 + y * m10 + tx
    v = x * m01 + y * m11 + ty
    w = x * m03 + y * m13 + 1.0
    w = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
    return torch.stack([u / w, v / w], dim=-1)


def affine_probe_term(pred: torch.Tensor, target: torch.Tensor, probe: torch.Tensor,
                      out_px: float, group_live: torch.Tensor | None = None) -> torch.Tensor:
    """How far the predicted transform moves five probe points from where the true one puts
    them, in crop pixels. See :func:`apply_proj` and ``data.group_probes``.

    This is to the transform head what ``curveloss`` is to the point head: stop scoring the
    parameters and score what they draw. Beyond fixing the unit, it weights the eight degrees
    of freedom by how much each actually displaces the layer -- a rotation error on a large
    group costs more than the same radians on a small one, which is exactly the ordering a
    renderer would report and the opposite of what L1 on raw matrix entries says.

    ``group_live`` masks groups with no live shape at that frame, and it is not an optional
    refinement. **34% of (frame, group) cells in this archive have nothing on screen** -- 59%
    on ``Layer_52`` and 70% on ``FAM blue_2`` -- so unmasked, a third of this term asks the
    network where an invisible group is. That is unlearnable from the alpha, which shows
    nothing there, and it is also unscorable: ``reconstruct.affine_doc`` only ever moves live
    shapes, so a wrong transform on a dead group cannot change a rendered number.

    It is the same argument this module already makes for masking the point term by ``live``,
    with one difference worth stating. A dead shape's control points are *meaningless* -- the
    artist left them wherever they last were. A dead group's matrix is a real sample of a real
    dense track, so the target is not wrong; it is merely impossible to predict from the input
    and free to get wrong. Masked either way, and for the stronger of the two reasons.
    """
    d = (apply_proj(pred, probe) - apply_proj(target, probe)).abs()
    if group_live is None:
        return d.mean() * out_px
    m = group_live[..., None, None].expand_as(d)
    return (d * m).sum() / m.sum().clamp(min=1) * out_px


def affine_temporal_term(pred: torch.Tensor, target: torch.Tensor, probe: torch.Tensor,
                         frames: torch.Tensor, out_px: float,
                         group_live: torch.Tensor | None = None) -> torch.Tensor:
    """The transform head's own jitter term, in crop pixels over adjacent frame pairs.

    The review's point: motion is the one quantity that cannot be read from a single frame,
    and the transform head is the only head whose entire output *is* motion -- yet in v1.1's
    first ladder it received none of the temporal machinery. It shares the encoder, so the
    window reached it; the consistency term did not.

    Only pairs one frame apart count, and only where the group is live at *both* frames -- a
    group blinking on carries a step change nobody should be asked to smooth, exactly as in
    :func:`temporal_term`.
    """
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    ok = (frames[1:] - frames[:-1]) == 1                                # (B-1,)
    if not bool(ok.any()):
        return pred.new_zeros(())
    a, t = apply_proj(pred, probe), apply_proj(target, probe)
    d = ((a[1:] - a[:-1]) - (t[1:] - t[:-1])).abs()
    m = ok[:, None, None, None].expand_as(d)
    if group_live is not None:
        live2 = group_live[1:] & group_live[:-1]                         # (B-1, G)
        m = m & live2[:, :, None, None].expand_as(d)
    n = m.sum()
    if not bool(n):
        return pred.new_zeros(())
    return (d * m).sum() / n * out_px


def key_positive_rate(els: Sequence[ElementData]) -> float:
    """Fraction of *live* (frame, shape) cells the artist put a key on, over the dataset.

    Measured rather than assumed, and reported, because it is what sets ``key_pos_weight``
    and it is also the number that makes accuracy a useless metric for this head: at 11.3%
    a head that never fires is right 88.7% of the time. The denominator is live cells only,
    matching how the term is masked and how ``reconstruct.rebuild`` chooses keys.
    """
    keys = live = 0
    for e in els:
        if not e.key_mask.size:
            continue
        keys += int((e.key_mask & e.live).sum())
        live += int(e.live.sum())
    return keys / live if live else 0.0


def key_positive_rates(els: Sequence[ElementData]) -> dict[str, float]:
    """The same rate **per layer**, which is the number that made the first probe fail.

    Measured on ``datasets/v002``, key density ranges from **0.037** on ``FAM green`` to
    **0.474** on ``TVC Layer_52`` -- a 13x spread across layers against an archive mean of
    0.113. With one global positive weight, the cheapest way for the head to lower its loss is
    to learn each layer's *base rate* and fire at it, which needs no timing information at
    all -- and a per-(layer, shape) query embedding is exactly the capacity to do that with.

    That is what the 12k probe measured: on ``Layer_52`` (density 0.474) the head fired on 77%
    of live cells and scored 0.803 matched key F1 against **0.797 for firing on everything**,
    while on ``FAM blue`` (density 0.057) it stopped firing altogether and scored 0.017. It had
    learned the spread and nothing inside it.

    Balancing per layer removes the exploit: within a layer, positives and negatives carry
    equal total weight, so matching the base rate gains nothing and the only remaining way
    down is *when*.
    """
    out: dict[str, float] = {}
    for e in els:
        if not e.key_mask.size:
            continue
        live = int(e.live.sum())
        out[e.element_id] = (int((e.key_mask & e.live).sum()) / live) if live else 0.0
    return out


def key_term(logits: torch.Tensor, target: torch.Tensor, live: torch.Tensor,
             pos_weight: float) -> torch.Tensor:
    """Class-balanced BCE on "is this frame a key for this shape", over live cells only.

    ``pos_weight`` multiplies the positive class, which is the whole reason this term learns
    anything: keys are 9.6% of live cells, so plain BCE is minimised by a head that never
    fires. Nothing here penalises firing *often* -- that job belongs downstream, where the
    probability biases a tolerance rather than emitting a key, because over-keying is this
    project's signature failure and a head that could emit keys directly would find it.
    """
    m = live
    if not bool(m.any()):
        return logits.new_zeros(())
    w = logits.new_tensor(float(pos_weight))
    loss = F.binary_cross_entropy_with_logits(
        logits[m], target[m].to(logits.dtype), pos_weight=w, reduction='mean')
    return loss


def live_rate(els: Sequence[ElementData], n_slots: int = 0) -> float:
    """Fraction of all ``(frame, shape)`` cells on which the artist has the shape on screen.

    The lifespan head's base rate, and the mirror image of the key head's. Keys are 32% of
    *live* cells, so that head's degenerate answer is "never"; live cells are **63%** of all
    cells, so this head's degenerate answer is "always" -- and unlike the key head's, the
    degenerate answer here is a well-known trivial baseline that costs 0.0971 soft IoU. The
    denominator is every cell, because that is what the term is computed over: a head asked to
    predict which frames a shape is on screen for cannot be masked by which frames it is on
    screen for.
    """
    live = cells = 0
    for e in els:
        live += int(e.live.sum())
        # With shared slots the denominator is every *slot*, not every shape: the empty slots
        # are cells the head has to get right too, and they are the majority. On v003 this
        # moves the rate from 0.627 to about 0.04, which flips the head's degenerate answer
        # from "always alive" back to "never" -- the same trap the key head fell into, and the
        # reason the weight is measured rather than carried over from S2.
        cells += int(len(e.frames)) * (n_slots or e.n_shapes)
    return live / cells if cells else 0.0


def live_rates(els: Sequence[ElementData], n_slots: int = 0) -> dict[str, float]:
    """The same rate **per element**, which is what the balancing needs.

    On ``datasets/v003`` it spans **0.093 to 1.000** -- wider than the 13x key-density spread
    that made the key head's first probe fail, and with the same exploit available: one global
    weight makes "learn each element's base rate and fire at it" the cheapest way down, and a
    per ``(element, shape)`` query row is exactly the capacity to do it.

    Four of the seventeen trained elements sit at 1.000 -- every shape on screen on every
    frame. Those have no negative class at all, so ``(1-r)/r`` is 0 for them, which would zero
    their positive loss and delete them from the term. :func:`positive_weights` pins those to
    1.0 instead.
    """
    return {e.element_id: int(e.live.sum()) / (len(e.frames) * (n_slots or e.n_shapes))
            for e in els}


def positive_weights(rates: dict[str, float]) -> dict[str, float]:
    """``(1 - r) / r`` per element, with the degenerate rates pinned to 1.0.

    ``r = 1`` means no negatives exist, so no correction is meaningful and the formula's 0
    would silently drop the element; ``r = 0`` means no positives exist, and the formula
    diverges. Both are real on this dataset -- five elements are always live -- so both are
    handled here rather than at the one call site that happens to have hit them.
    """
    return {k: ((1.0 - v) / v if 0.0 < v < 1.0 else 1.0) for k, v in rates.items()}


def alive_term(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Class-balanced BCE on "is this shape on screen at this frame", over **every** cell.

    The one structural difference from :func:`key_term`, and the reason this is a separate
    function rather than a second call: the key term is masked to live cells, and this term
    cannot be. Masking a lifespan loss by the lifespan would leave it predicting only the
    frames where the answer is already yes.

    Nothing here knows that the two mistakes cost differently -- omitting a live shape is
    worth 3.2x drawing a dead one, measured in ``v2-s2-design-note.md`` section 2.3. That
    asymmetry is applied at **decode**, in ``reconstruct.alive_mask``, rather than here: a
    BCE skewed to one side moves the probabilities themselves, which would then be miscalibrated
    for every threshold the sweep tries. The loss learns the probability; the decode prices the
    mistake.
    """
    w = logits.new_tensor(float(pos_weight))
    return F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), pos_weight=w, reduction='mean')


def count_term(logits: torch.Tensor, target: torch.Tensor,
               style_weight: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy on the point count, plus plan section 5's point-economy style statistic.

    ``logits`` is ``(B, S, max_points + 1)`` with class index == the count; ``target`` is
    ``(S,)``. The count does not vary with the frame, so every row of the batch carries the
    same target and the term is averaged over the batch as well -- which costs nothing and
    keeps the head's gradient at the same scale as the per-frame ones.

    The style statistic is ``|E[P] - P| / P`` on the head's *expected* count, which is
    differentiable where an argmax is not. Two decisions in it:

    * **expectation, not argmax.** Charter L2 asks for point economy as a soft statistical
      target rather than a hard match, and an expectation is what makes "about the right
      number of points" expressible at all.
    * **divided by P.** Deviation 4.3 of the design note. Plan section 5's unweighted form
      prices one point the same at P = 68 and at P = 4, and the render does not: dropping a
      point costs 0.0004-0.053 soft IoU on fifteen of v003's elements and 0.42-0.50 on the two
      whose smallest shape has four points, where a 4-point closed B-spline drops to a
      near-degenerate 3.
    """
    B, S, C = logits.shape
    tgt = target[None].expand(B, -1).reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, C), tgt, reduction='mean')
    if not style_weight:
        return ce, logits.new_zeros(())
    grid = torch.arange(C, device=logits.device, dtype=logits.dtype)
    expected = (logits.softmax(-1) * grid).sum(-1)                    # (B, S)
    style = ((expected - target[None]).abs() / target.clamp(min=1)[None]).mean()
    return ce, style


def losses(pred_pts: torch.Tensor, pred_aff: torch.Tensor, b: dict[str, torch.Tensor],
           out_px: float, cfg: TrainConfig,
           maps: PolylineMaps | None = None,
           pred_key: torch.Tensor | None = None,
           key_pos_weight: float | None = None,
           pred_alive: torch.Tensor | None = None,
           alive_pos_weight: float | None = None,
           pred_count: torch.Tensor | None = None
           ) -> tuple[torch.Tensor, dict[str, float]]:
    mask = (b['point_mask'] & b['live'][..., None, None]).unsqueeze(-1)
    n = mask.sum().clamp(min=1)
    point_px = (((pred_pts - b['points']).abs() * mask).sum() / n) * out_px
    parts = {'point_px': float(point_px.detach())}

    if cfg.affine_space == CROP:
        aff = affine_probe_term(pred_aff, b['proj_crop'], b['probe'], out_px,
                                b['group_live'])
        parts['affine_px'] = float(aff.detach())
    else:
        aff = (pred_aff - b['affine']).abs().mean()
        parts['affine'] = float(aff.detach())
    total = cfg.point_weight * point_px + cfg.affine_weight * aff

    if cfg.curve_weight and maps is not None and maps.groups:
        curve_px = polyline_loss(pred_pts, b['points'], b['live'], maps, out_px)
        total = total + cfg.curve_weight * curve_px
        parts['curve_px'] = float(curve_px.detach())
    if cfg.temporal_weight:
        temp_px = temporal_term(pred_pts, b, out_px)
        total = total + cfg.temporal_weight * temp_px
        parts['temporal_px'] = float(temp_px.detach())
        # The transform head gets the same term, and only in crop space: differencing raw
        # document-space matrix entries would not be in pixels and could not share a weight.
        if cfg.affine_space == CROP:
            at_px = affine_temporal_term(pred_aff, b['proj_crop'], b['probe'],
                                          b['frames'], out_px, b['group_live'])
            total = total + cfg.temporal_weight * cfg.affine_weight * at_px
            parts['affine_temporal_px'] = float(at_px.detach())

    if cfg.key_weight and pred_key is not None:
        w = cfg.key_pos_weight if key_pos_weight is None else key_pos_weight
        # Sliced to the real shapes: a slot with nothing in it has no artist keys to find,
        # and scoring it would pay the head for staying quiet where quiet is free.
        n_real = b['live'].shape[1]
        pred_key = pred_key[:, :n_real]
        key_px = key_term(pred_key, b['key_mask'], b['live'], w)
        total = total + cfg.key_weight * key_px
        parts['key_bce'] = float(key_px.detach())
        with torch.no_grad():
            hit = (pred_key > 0) & b['key_mask'] & b['live']
            fired = (pred_key > 0) & b['live']
            want = b['key_mask'] & b['live']
            # Precision and recall at the 0.5 threshold, logged rather than optimised: they
            # are what says whether a falling BCE is the head learning or the head giving up.
            parts['key_prec'] = float(hit.sum() / fired.sum().clamp(min=1))
            parts['key_rec'] = float(hit.sum() / want.sum().clamp(min=1))

    if cfg.alive_weight and pred_alive is not None:
        w = cfg.alive_pos_weight if alive_pos_weight is None else alive_pos_weight
        tgt = b.get('alive_target')
        if tgt is None:
            tgt = b['live']
        alive_bce = alive_term(pred_alive[:, :tgt.shape[1]], tgt, w)
        total = total + cfg.alive_weight * alive_bce
        parts['alive_bce'] = float(alive_bce.detach())
        with torch.no_grad():
            fired, want = pred_alive[:, :tgt.shape[1]] > 0, tgt
            # The two error rates, over every cell, logged rather than optimised -- and split,
            # because they are not worth the same. A false negative costs 3.2x a false
            # positive at the render, so a falling BCE that is trading recall for precision is
            # going the wrong way and only these two columns would show it.
            parts['alive_fp'] = float((fired & ~want).sum() / want.numel())
            parts['alive_fn'] = float((~fired & want).sum() / want.numel())
            parts['alive_acc'] = float((fired == want).sum() / want.numel())

    if cfg.count_weight and pred_count is not None:
        pred_count = pred_count[:, :b['live'].shape[1]]
        ce, style = count_term(pred_count, b['point_count'], cfg.count_style_weight)
        total = total + cfg.count_weight * ce + cfg.count_style_weight * style
        parts['count_ce'] = float(ce.detach())
        if cfg.count_style_weight:
            parts['count_style'] = float(style.detach())
        with torch.no_grad():
            pred = pred_count.argmax(-1)
            parts['count_acc'] = float((pred == b['point_count'][None]).float().mean())

    parts['total'] = float(total.detach())
    return total, parts


def element_weights(els: Sequence[ElementData], mode: str) -> np.ndarray:
    if mode == 'sqrt':
        w = np.array([np.sqrt(len(e.frames) * e.n_shapes) for e in els], float)
    elif mode == 'frames':
        w = np.array([len(e.frames) for e in els], float)
    else:
        raise ValueError(f'unknown sample_weight {mode!r}')
    return w / w.sum()


def train(dataset_root: str | Path, out_dir: str | Path,
          cfg: TrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    if cfg.affine_space not in AFFINE_SPACES:
        raise ValueError(f'unknown affine_space {cfg.affine_space!r}, '
                         f'want one of {AFFINE_SPACES}')
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)

    # ``trained_only`` drops whatever the dataset's own split withholds from training. On
    # datasets/v001 that is nothing -- it has no split record -- so every earlier number
    # reproduces; on v002 it is the two layers named in ``splits.json``. Dropping them here
    # rather than at sampling time also drops their *query rows*, which is the point: a
    # layer whose embedding is never allocated cannot be accidentally trained by a stray
    # gradient, and the checkpoint records which layers existed.
    all_els = load_dataset(dataset_root, with_local=False)   # training never reads local
    els = [e for e in all_els if e.in_train or not cfg.use_split]
    withheld = [e.element_id for e in all_els if not e.in_train] if cfg.use_split else []
    if not cfg.use_split:
        for e in els:                       # drop the recorded frame split as well
            e.split_train = np.zeros(0, np.int64)
            e.split_held = np.zeros(0, np.int64)
    for e in els:
        _ = e.alphas                                   # materialise once, up front
    # Each layer owns a contiguous block of the query tables, so shape identity is per
    # (layer, shape) rather than per index. See net.RotoNet.
    shape_base, group_base, ns, ng = {}, {}, 0, 0
    for e in els:
        shape_base[e.element_id], group_base[e.element_id] = ns, ng
        ns += e.n_shapes
        ng += e.n_groups
    max_shapes, max_groups = ns, ng
    # S3: the query bank is the slot count, shared by every element, rather than the running
    # total of per-element rows. 675 rows becomes 256, and none of them means a named shape.
    perms = {e.element_id: canonical_order(e) for e in els} if cfg.n_slots else {}
    if cfg.n_slots:
        widest = max(e.n_shapes for e in els)
        if widest > cfg.n_slots:
            raise ValueError(f'n_slots={cfg.n_slots} but {widest} shapes in one element; a '
                             'slot bank that cannot hold the widest element silently drops '
                             'its tail')
        max_shapes = cfg.n_slots
    max_points = max(e.points.shape[2] for e in els)
    max_coords = max(e.points.shape[3] for e in els)
    splits = {e.element_id: e.split(cfg.holdout_every) for e in els}
    n_held = sum(len(h) for _, h in splits.values())
    recorded = cfg.use_split and any(len(e.split_train) for e in els)
    affine_dim = PROJ_DOF if cfg.affine_space == CROP else AFFINE_DOF
    # The key term's positive weight: measured from the data, never tuned. One global number
    # is exploitable -- see key_positive_rates -- so the default balances within each layer,
    # and both modes are kept because the first probe ran the global one.
    key_w: dict[str, float] = {}
    if cfg.key_weight:
        rate = key_positive_rate(els)
        if not cfg.key_pos_weight:
            cfg = replace(cfg, key_pos_weight=(1.0 - rate) / rate if rate else 1.0)
        if cfg.key_balance == 'per_element':
            rates = key_positive_rates(els)
            key_w = {k: ((1.0 - v) / v if v else 1.0) for k, v in rates.items()}
            lo, hi = min(key_w.values()), max(key_w.values())
            print(f'key head: the artist keys {100 * rate:.1f}% of live (frame, shape) cells '
                  f'over the archive, and {100 * min(rates.values()):.1f}%-'
                  f'{100 * max(rates.values()):.1f}% per layer -- so the positive weight is '
                  f'balanced per layer, {lo:.2f} to {hi:.2f} (measured, not tuned)')
        elif cfg.key_balance == 'global':
            print(f'key head: the artist keys {100 * rate:.1f}% of live (frame, shape) cells, '
                  f'so key_pos_weight = {cfg.key_pos_weight:.2f} globally (measured, not '
                  f'tuned). NOTE: one global weight is exploitable by learning each layer\'s '
                  f'base rate -- see key_positive_rates')
        else:
            raise ValueError(f'unknown key_balance {cfg.key_balance!r}, '
                             "want 'per_element' or 'global'")
    # The lifespan term's positive weight, on the same measured-never-tuned rule. The mirror
    # image of the key head's: live cells are the *majority* here, so the correction is below
    # 1 and the degenerate answer it guards against is "always alive" rather than "never".
    alive_w: dict[str, float] = {}
    if cfg.alive_weight:
        rate = live_rate(els, cfg.n_slots)
        if not cfg.alive_pos_weight:
            cfg = replace(cfg, alive_pos_weight=(1.0 - rate) / rate if 0 < rate < 1 else 1.0)
        rates = live_rates(els, cfg.n_slots)
        if cfg.alive_balance == 'per_element':
            alive_w = positive_weights(rates)
            lo, hi = min(alive_w.values()), max(alive_w.values())
            n_full = sum(1 for v in rates.values() if v >= 1.0)
            print(f'lifespan head: shapes are on screen on {100 * rate:.1f}% of all '
                  f'(frame, shape) cells, and {100 * min(rates.values()):.1f}%-'
                  f'{100 * max(rates.values()):.1f}% per element -- so the positive weight is '
                  f'balanced per element, {lo:.2f} to {hi:.2f} (measured, not tuned)'
                  + (f'; {n_full} element(s) are always live and are pinned to 1.0, since '
                     '(1-r)/r would be 0 and delete them from the term' if n_full else ''))
        elif cfg.alive_balance == 'global':
            print(f'lifespan head: {100 * rate:.1f}% of all cells are live, so '
                  f'alive_pos_weight = {cfg.alive_pos_weight:.2f} globally. NOTE: one global '
                  'weight is exploitable at a 0.093-1.000 live-rate spread -- see live_rates')
        else:
            raise ValueError(f'unknown alive_balance {cfg.alive_balance!r}, '
                             "want 'per_element' or 'global'")
    if cfg.count_weight:
        counts = np.concatenate([e.n_points_per_shape for e in els])
        print(f'point-count head: {len(counts)} shapes, {len(set(counts.tolist()))} distinct '
              f'counts over {counts.min()}-{counts.max()}, {max_points + 1} classes. '
              f'REPORTED, NOT GATED -- the query row is per (element, shape) and a count does '
              f'not vary with the frame, so accuracy here measures memory (see the design note)'
              + ('' if cfg.give_point_count else '; n_points is out of the query input'))
    print(f'{len(els)} elements | {sum(len(e.frames) for e in els)} frames '
          f'({n_held} held out{", from the dataset\'s own split" if recorded else ""})'
          f'{f" | {len(withheld)} element(s) withheld entirely" if withheld else ""} '
          f'| queries {max_shapes} shape / {max_groups} group  '
          f'Pmax {max_points} Cmax {max_coords} | {device} | window {cfg.in_frames}'
          f'{" aligned" if cfg.align_window and cfg.in_frames > 1 else ""} '
          f'| self-attn {cfg.self_attn} | sampling {cfg.sampling} '
          f'| transform {cfg.affine_space} ({affine_dim}-dof'
          f'{f", decoder depth {cfg.affine_depth}" if cfg.affine_depth else ""})'
          f'{f" | key head w={cfg.key_weight}" if cfg.key_weight else ""}')
    if withheld:
        print('  withheld from training: ' + ', '.join(withheld))

    net = RotoNetV2(max_shapes, max_groups, max_points, max_coords, cfg.dim, cfg.depth,
                  in_frames=cfg.in_frames, self_attn=cfg.self_attn,
                  affine_dim=affine_dim, align_window=cfg.align_window,
                  affine_depth=cfg.affine_depth or None,
                  key_head=bool(cfg.key_weight), query_mode=SLOTS if cfg.n_slots else TABLE,
                  alive_head=bool(cfg.alive_weight),
                  count_head=bool(cfg.count_weight or cfg.count_style_weight),
                  desc_dim=3 if cfg.give_point_count else 2).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    maps = {e.element_id: PolylineMaps(e.n_points_per_shape, e.closed_per_shape,
                                     e.coords_per_shape, device)
            for e in els} if cfg.curve_weight else {}
    if maps:
        print(f'  curve loss covers {sum(m.n_shapes_covered for m in maps.values())} '
              f'of {sum(e.n_shapes for e in els)} shapes '
              f'(Bezier and <3-point shapes sit out)')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup)
        * (0.5 * (1 + np.cos(np.pi * min(1.0, s / cfg.steps)))))

    weights = element_weights(els, cfg.sample_weight)
    state = TrainState()
    t0 = time.time()
    run: dict[str, list[float]] = {}

    for step in range(cfg.steps):
        ei = int(rng.choice(len(els), p=weights))
        el = els[ei]
        idx = sample_indices(rng, splits[el.element_id][0], cfg.batch, cfg.sampling)
        b = _batch(el, idx, shape_base[el.element_id], group_base[el.element_id],
                   cfg.in_frames, device, cfg.align_window, cfg.give_point_count,
                   cfg.n_slots, perms.get(el.element_id))

        pred = net(b['alpha'], b['shape_ids'], b['group_ids'], b['desc'])
        # Geometry terms read the real shapes only; the alive head keeps all the slots,
        # because an empty slot is exactly what it is there to recognise.
        pts = pred.points[:, :el.n_shapes, :el.points.shape[2], :el.points.shape[3]]
        loss, parts = losses(pts, pred.affine, b, el.out_px, cfg, maps.get(el.element_id),
                             pred.key, key_w.get(el.element_id),
                             pred.alive, alive_w.get(el.element_id), pred.count)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run.setdefault(k, []).append(v)
        state.step = step + 1

        if (step + 1) % cfg.log_every == 0:
            row = {'step': step + 1, 'lr': sched.get_last_lr()[0],
                   'elapsed_s': round(time.time() - t0, 1),
                   **{k: float(np.mean(v[-cfg.log_every:])) for k, v in run.items()}}
            state.history.append(row)
            extra = ''.join(f'  {k[:-3]} {row[k]:6.3f}' for k in ('curve_px', 'temporal_px')
                            if k in row)
            if 'key_bce' in row:
                extra += (f'  key {row["key_bce"]:5.3f} '
                          f'(P {row["key_prec"]:.2f} R {row["key_rec"]:.2f})')
            if 'alive_bce' in row:
                # FP and FN separately, never as one accuracy: they cost 1:3.2 at the render.
                extra += (f'  alive {row["alive_bce"]:5.3f} '
                          f'(acc {row["alive_acc"]:.3f} FP {row["alive_fp"]:.3f} '
                          f'FN {row["alive_fn"]:.3f})')
            if 'count_ce' in row:
                extra += f'  count {row["count_ce"]:5.3f} (acc {row["count_acc"]:.3f})'
            tr = (f'  transform {row["affine_px"]:7.3f}px' if 'affine_px' in row
                  else f'  affine {row["affine"]:.5f}')
            print(f'  step {row["step"]:>5}  point {row["point_px"]:7.3f}px'
                  f'{extra}{tr}  total {row["total"]:7.3f}  '
                  f'[{row["elapsed_s"]:.0f}s]', flush=True)

    ckpt = {
        'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
        'arch': {'max_shapes': max_shapes, 'max_groups': max_groups,
                 'max_points': max_points, 'max_coords': max_coords,
                 'dim': cfg.dim, 'depth': cfg.depth,
                 'in_frames': cfg.in_frames, 'self_attn': cfg.self_attn,
                 'affine_dim': affine_dim, 'align_window': cfg.align_window,
                 'affine_depth': net.affine_depth,
                 'key_head': net.key_head, 'alive_head': net.alive_head,
                 'count_head': net.count_head, 'desc_dim': net.desc_dim,
                 'query_mode': net.query_mode},
        'config': asdict(cfg),
        'elements': [e.element_id for e in els],
        # What this run was *not* allowed to see, on the checkpoint rather than only in the
        # dataset, so a scored run can be read without also having the dataset to hand.
        'withheld_elements': withheld,
        'dataset': str(dataset_root),
        'split_source': 'dataset' if recorded else 'holdout_every',
        'shape_base': shape_base,
        'group_base': group_base,
        'splits': {k: {'train': v[0].tolist(), 'held': v[1].tolist()}
                   for k, v in splits.items()},
        'n_params': n_params,
    }
    torch.save(ckpt, out / 'model.pt')
    wall = time.time() - t0
    # What the run cost, on the record. The handover asks local-versus-cloud as a capability
    # question; these two numbers are the whole of the local side of that answer.
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2 ** 20
               if device.type == 'cuda' else None)
    summary = {'n_params': n_params, 'steps': cfg.steps, 'layers': len(els),
               'withheld_layers': withheld, 'dataset': str(dataset_root),
               'split_source': 'dataset' if recorded else 'holdout_every',
               'key_positive_rate': key_positive_rate(els) if cfg.key_weight else None,
               'key_positive_rate_per_element': (key_positive_rates(els) if cfg.key_weight
                                               else None),
               'key_pos_weight_per_element': key_w or None,
               'live_rate': live_rate(els, cfg.n_slots) if cfg.alive_weight else None,
               'live_rate_per_element': (live_rates(els, cfg.n_slots) if cfg.alive_weight
                                        else None),
               'alive_pos_weight_per_element': alive_w or None,
               'frames': sum(len(e.frames) for e in els), 'frames_held_out': n_held,
               'device': str(device),
               'device_name': (torch.cuda.get_device_name(device)
                               if device.type == 'cuda' else None),
               'peak_gpu_mb': round(peak_mb, 1) if peak_mb is not None else None,
               'steps_per_s': round(cfg.steps / wall, 2) if wall else None,
               'config': asdict(cfg),
               'wall_clock_s': round(wall, 1),
               'final': state.history[-1] if state.history else {},
               'history': state.history}
    (out / 'train_log.json').write_text(json.dumps(summary, indent=2))
    print(f'\nsaved {out / "model.pt"}  ({n_params/1e6:.2f}M params, '
          f'{summary["wall_clock_s"]:.0f}s, {summary["steps_per_s"]:.1f} steps/s'
          f'{f", peak {peak_mb:.0f} MB" if peak_mb is not None else ""})')
    return summary
