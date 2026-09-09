"""Model output -> a real spline program -> pixels, scored against the layer's own alpha.

This is the step that decides whether the model worked, and it is deliberately the strictest
reading available. The network's per-frame geometry is converted back to local coordinates,
the keyframe selector picks which frames become keys, a ``RotoDoc`` is rebuilt from those
keys alone, and that document is *rendered* and compared to the clean alpha the network was
given.

Scoring the rendered result rather than the control points is the whole point. A low point
error can still draw the wrong picture -- a handful of points on a small shape can be badly
wrong while the mean stays flattering -- and a keyframe set that looks sparse can interpolate
into something an artist would reject. Rendering collapses geometry, keys, and lifespan into
the single question that matters: does it look like the matte.

The render must be at the **dataset's own supersample**. Scoring a prediction rendered at
supersample 2 against a target written at 4 disagrees on every anti-aliased boundary pixel,
which is precisely what soft IoU is built to notice: re-rendering the *artist's own shapes*
that way scores 0.953-0.995 instead of 1.000. Every v1 number carried that handicap.

What is teacher-forced, and therefore not claimed: the shape breakdown (how many shapes,
their point counts, their groups), each shape's lifespan, and the layer transform track. The
network supplies the geometry; ``roto.keys`` supplies the timing. ``RebuildConfig.
predicted_affine`` puts the transform head's own output on the table instead -- see
``affine_doc``.

**``RebuildConfig.motion`` is the one that drops the teacher-forcing entirely**, and it is a
different measurement from ``predicted_affine`` rather than a stronger version of it.
``predicted_affine`` isolates the head: the artist's own control points, moved by the
predicted track, so the number is *pure motion error* with no geometry error in front of it.
``motion='predicted'`` asks the question the roadmap actually has to answer -- predicted
geometry *and* predicted motion, nothing given but the shape breakdown and the lifespans.

Those two are not the same number, and the reason is worth stating because it looks like it
should cancel. Points are regressed in crop space and converted to local by inverting the
layer matrix the renderer then re-applies, so if nothing intervened, a transform error would
divide straight back out and the picture would be identical to the teacher-forced one. What
intervenes is the **keyframe stage**: keys are chosen and values fitted on the *local* track,
and inverting a wrong matrix gives a differently-shaped local track, which is sparsified
differently. So the end-to-end cost of predicted motion is exactly what survives that
sparsification, and it has to be measured rather than reasoned about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..dataset import load_alpha
from ..metrics import iou, soft_iou
from ..ir import Key, RotoDoc
from ..keys import f1 as key_f1
from ..keys import select
from ..keys.refit import refit_key_values
from ..program import PROJ_DOF, matrix_from_affine, matrix_from_proj
from ..render.raster import config_from_meta, render_union
from ..sfx.json_ir import from_json_ir
from .data import LayerData, load_element
from .geometry import crop_to_local
from .net import RotoNet
from .smoothing import BOXCAR, smooth_track

ARTIST, PREDICTED = 'artist', 'predicted'
MOTION_SOURCES = (ARTIST, PREDICTED)
"""Whose layer-transform track a reconstruction is built and rendered through."""

DEFAULT_TOL_PX = 1.0
"""Keyframe tolerance, in crop pixels.

On *ground-truth* tracks the best tolerance is 0.1 px. On *predicted* tracks that value is
badly wrong, and the reason couples two components that look independent: the keyframe search
cannot be tuned below the geometry's own noise floor. A predicted track carries a per-frame
jitter of roughly the model's point error, so asking the search to reproduce it within 0.1 px
forces a key on almost every frame -- measured at 16.5x the artist's key count. The tolerance
has to sit above the model's noise, not above the artist's precision.

So the operating point is chosen against predicted tracks, and it is re-chosen whenever the
model changes. Measured per model in ``v1.1/results/operating_point_<run>.json``
(``scripts/sweep_operating_point.py``), and the answer is not stable across models: v1's own
model peaks at a 13-frame window and 0.5 px, while the converged v1.1 model peaks at a
13-frame window, 1.0 px, savgol and a key-value refit -- worth +0.0035 soft IoU and +0.027 key
F1 over these defaults, for no retraining.

The defaults below stay at v1's values deliberately, so that re-running any v1 number
reproduces it. The better operating point is a flag, not a silent change.
"""

SMOOTH_WINDOW = 9
"""Frames of temporal smoothing applied to a predicted track before knots are chosen.

The artist's true track is piecewise linear between sparse keys; the model's error is
approximately independent frame to frame. A short centred filter therefore removes a large
part of the noise while leaving genuine motion almost untouched, which lets the search see
the structure it is meant to find. Set to 1 to disable.

Measured in v1, at 1.0 px tolerance: turning smoothing on raised rendered soft-IoU on every
layer tried (0.9820 -> 0.9846, 0.9605 -> 0.9653, 0.7651 -> 0.7763) *and* cut the key count
from 3.01x the artist's to 0.99x. Both improve together because the keys it removes were
spent tracking noise, not motion. Which *filter* does the smoothing is a separate question --
see ``roto.model.smoothing``.
"""


@dataclass(slots=True)
class RebuildConfig:
    """Everything about turning a predicted track into keyframed splines, in one place.

    These are the axes of the operating-point sweep, so they belong together rather than as
    a growing tail of keyword arguments.
    """
    tol_px: float = DEFAULT_TOL_PX
    smooth: int = SMOOTH_WINDOW
    smooth_kind: str = BOXCAR
    """``'boxcar'`` reproduces v1; ``'savgol'`` preserves the motion peaks artists key."""
    refit_values: bool = False
    """Fit key values to the raw track once the key frames are chosen. See ``keys.refit``.

    Off by default so v1's behaviour is reproducible; the sweep decides whether it earns its
    place, and it is a strictly cheaper change than swapping the filter."""
    predicted_affine: bool = False
    """Score the *transform head's* output instead of the artist's track. See ``affine_doc``."""
    motion: str = ARTIST
    """``'artist'`` teacher-forces the layer transform track, as v1 and v1.1 both do.
    ``'predicted'`` uses the head's own track for **both** halves of the round trip -- the
    crop-to-local conversion the keys are chosen on, and the transform the render re-applies.

    That is the first de-teacher-forced number in this project, and the handover's decision
    tree asks for it by name. It is not the same as ``predicted_affine``: see this module's
    docstring for why the two cannot be collapsed, and why the difference is the keyframe
    stage rather than the transform algebra."""
    supersample: int | None = None
    """Render supersample; ``None`` means the dataset's own, which is the only like-for-like
    choice. An explicit value exists so the anti-aliasing axis can be measured on purpose."""
    key_bias: float = 0.0
    """How far the key-timing head may tighten the DP's tolerance locally. ``0`` is v1.2.

    This is the head's *only* route into the output. See ``keys.dp.local_tolerances`` for the
    form and ``keys.dp``'s header for why the head is not allowed to emit a key directly:
    over-keying is this project's measured signature failure, and a bias on a minimising
    objective cannot cause it because every extra knot still costs."""
    key_slack: float = 0.0
    """How far the head may *loosen* the DP's tolerance where it expects no key.

    The other half of ``key_bias``, and the half that can raise precision -- which is the
    binding constraint on key F1, since this project's measured failure is over-keying at
    precision 0.21-0.44 against recall 0.75-1.00. Tightening alone can only add keys;
    loosening is what lets the pair move keys without moving the key count, and the key count
    has to stay inside an editable economy. ``0`` is the one-sided form. See
    ``keys.dp.local_tolerances``."""
    key_thresh: float = 0.5
    """Threshold at which the head's *own* key set is read off, for reporting only.

    Nothing downstream uses it -- the DP sees the probability, not a decision. It exists so
    the head can be scored on its own terms beside the pipeline it feeds, which is the only
    way to tell a head that learned nothing from a bias that was set too low."""


@dataclass(slots=True)
class Reconstruction:
    layer_id: str
    doc: RotoDoc
    frames: np.ndarray
    soft_iou: np.ndarray
    iou: np.ndarray
    point_err_px: float
    keys_predicted: int
    keys_artist: int
    key_precision: float
    key_recall: float
    key_f1: float
    key_f1_strict: float = 0.0
    """``key_f1`` with the out-of-live-range artist keys **excluded** rather than clipped.

    7.91% of this archive's 24,335 artist keys sit outside their own shape's live range, and
    half of all shapes have at least one. ``rebuild`` always forces a knot at the first and
    last live frame, so a key clipped onto that boundary is matched for free -- worth 0.009 to
    0.040 key F1 per layer, which is the size of difference this project argues about (v1.2's
    constrained operating point was worth +0.026). ``key_f1`` keeps the clipped definition
    unchanged, because every number in v1, v1.1 and v1.2 used it and the plan's >= 0.42 gate
    was set against one of them. This is the one without the free credit."""
    shapes_with_no_in_range_key: int = 0
    baseline_pipeline_key_f1_random: float = 0.0
    key_f1_over_random: float = 0.0
    """``key_f1`` minus what the *same number of keys* placed at random would have scored.

    Not decoration, and not the same argument as the key-economy column. Key F1 with a
    one-frame matching tolerance is close to saturated by keying often, and the two
    densest-keyed layers in this archive carry 40% of its keys, so the aggregate is dominated
    by the layers where a high score is cheapest. Every key F1 in v1, v1.1 and v1.2 was quoted
    without this control; it is added here because v1.3's acceptance gate reads that number."""
    point_err_p95_px: float = 0.0
    """The 95th percentile of per-point error, in crop pixels, over every live control point.

    The mean is the number that gets quoted and the tail is the number that gets a shot
    rejected: a layer can average 0.9 px while a hundred points sit at 5. Reported next to
    the mean everywhere, because the two move independently -- smoothing pulls the mean down
    and can leave the tail where it was."""
    point_err_max_px: float = 0.0
    """The single worst live control point in the run.

    One point, so it is the noisiest statistic here and the least useful for comparing runs --
    kept because the handover asks for max beside p95, and because the *ratio* of the two says
    whether the tail is a population or an outlier."""
    head_key_precision: float = 0.0
    head_key_recall: float = 0.0
    head_key_f1: float = 0.0
    baseline_key_f1_all_live: float = 0.0
    baseline_key_f1_random: float = 0.0
    head_key_f1_over_best_baseline: float = 0.0
    """The head's matched key F1 **minus the better of two trivial baselines**: firing on
    every live frame, and firing on a random subset of the same size.

    This is the only one of the three that is a statement about the head, and it exists
    because the raw figure is not. Key density ranges 13x across these layers, so a head that
    learns nothing but each layer's base rate scores well: on the 12k probe, ``Layer_52`` read
    0.803 matched F1 against **0.797 for firing on every live frame**, while ``FAM blue`` --
    density 0.057 against Layer_52's 0.474 -- stopped firing at all and read 0.017. The head
    had learned the spread across layers and nothing inside any of them. Same species as the
    lesson this project already had about accuracy at a 6% positive rate, one level up."""
    """The key-timing head scored **on its own**, at ``RebuildConfig.key_thresh``.

    Not the same number as ``key_f1``, which scores the keys the DP actually chose. Both are
    reported because they can move in opposite directions and the difference is diagnostic: a
    head with good F1 and a pipeline with bad F1 means the bias is too low, and the reverse
    means the bias is doing damage. Zero when the checkpoint has no key head."""
    key_tol_min: float = 0.0
    key_tol_max: float = 0.0
    """The range of local tolerance the DP actually ran at, over every shape in the layer."""
    jitter_px: float = 0.0
    """Mean frame-to-frame change in the *error* of the predicted track, in crop pixels.

    The number the review identified as the system's master constraint: it is what forces
    smoothing, which forces the key tolerance, which caps key F1. Reported so the chain can
    be seen moving rather than inferred."""

    def summary(self) -> dict[str, Any]:
        # Worst-frame counts rather than only the minimum: one bad frame in 190 is a rejected
        # shot, so how many there are is the operational question, and a single min cannot
        # distinguish one outlier from a bad third of the track.
        return {
            'layer_id': self.layer_id,
            'frames': int(len(self.frames)),
            'mean_soft_iou': float(self.soft_iou.mean()),
            'min_soft_iou': float(self.soft_iou.min()),
            'p05_soft_iou': float(np.percentile(self.soft_iou, 5)),
            'frames_below_0.95': int((self.soft_iou < 0.95).sum()),
            'frames_below_0.90': int((self.soft_iou < 0.90).sum()),
            'mean_iou': float(self.iou.mean()),
            'point_err_px': self.point_err_px,
            'p95_point_err_px': self.point_err_p95_px,
            'max_point_err_px': self.point_err_max_px,
            'jitter_px': self.jitter_px,
            'keys_predicted': self.keys_predicted,
            'keys_artist': self.keys_artist,
            'key_ratio': self.keys_predicted / max(1, self.keys_artist),
            'key_precision': self.key_precision,
            'key_recall': self.key_recall,
            'key_f1': self.key_f1,
            'key_f1_strict': self.key_f1_strict,
            'shapes_with_no_in_range_key': self.shapes_with_no_in_range_key,
            'baseline_pipeline_key_f1_random': self.baseline_pipeline_key_f1_random,
            'key_f1_over_random': self.key_f1_over_random,
            'head_key_precision': self.head_key_precision,
            'head_key_recall': self.head_key_recall,
            'head_key_f1': self.head_key_f1,
            'baseline_key_f1_all_live': self.baseline_key_f1_all_live,
            'baseline_key_f1_random': self.baseline_key_f1_random,
            'head_key_f1_over_best_baseline': self.head_key_f1_over_best_baseline,
            'key_tol_min': self.key_tol_min,
            'key_tol_max': self.key_tol_max,
            'worst_frame': int(self.frames[int(np.argmin(self.soft_iou))]),
        }


def load_model(checkpoint: str | Path,
               device: str | torch.device | None = None) -> tuple[RotoNet, dict[str, Any]]:
    """Load a checkpoint onto ``device`` (GPU when one exists).

    ``arch`` carries ``in_frames`` and ``self_attn`` from v1.1 on; both default off in
    ``RotoNet``, so a v1 checkpoint written before they existed still loads unchanged.
    """
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    dev = torch.device(device) if device is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RotoNet(**ck['arch'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net.to(dev), ck


@torch.no_grad()
def untrained_queries(net: RotoNet, el: LayerData, seed: int = 0) -> tuple[int, int]:
    """Give a layer the checkpoint never saw its own **freshly initialised** query rows.

    Shape queries are per ``(layer, shape)``, so a layer withheld from training has no rows in
    the query table and there is no honest default. The two available choices are not
    equivalent:

    * Reuse another layer's rows -- which is what ``shape_base.get(name, 0)`` silently does --
      and the number measures nothing at all: a query vector trained to mean "shape 7 of FAM
      blue" applied to a different layer's shape 7.
    * Allocate new rows from the same initialisation the trained rows started at, which is
      what this does. Then the number means something specific and useful: **what the encoder
      alone produces**, with the memorisation capacity set to zero.

    That second reading is the one v2 needs. ``net.RotoNet`` has said since v1 that the query
    table is memorisation capacity, that the v1 number therefore does not transfer, and that
    v2 must replace these embeddings with queries the encoder produces -- "and the gap between
    the two is the honest measure of what the encoder still has to learn". This function is
    how that gap gets measured on the *current* model instead of being deferred again.

    Seeded, so the number is reproducible; and it is a floor rather than a prediction, since
    an untrained query is worse than any query v2 would derive from the picture.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for bank, n in ((net.shape_bank, el.n_shapes), (net.group_bank, el.n_groups)):
        base, dim = bank.num_embeddings, bank.embedding_dim
        grown = torch.nn.Embedding(base + n, dim)
        # nn.Embedding initialises N(0, 1); reproduce it explicitly rather than relying on
        # the constructor so the seed is ours and the distribution is stated.
        with torch.no_grad():
            grown.weight.normal_(generator=g)
            grown.weight[:base] = bank.weight
        grown = grown.to(bank.weight.device)
        if bank is net.shape_bank:
            net.shape_bank = grown
        else:
            net.group_bank = grown
        out.append(base)
    return out[0], out[1]


@torch.no_grad()
def predict(net: RotoNet, el: LayerData, shape_base: int = 0, group_base: int = 0,
            batch: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(points (F,S,Pmax,Cmax,2) in crop space, affine (F,G,dof), key prob (F,S) or None)``.

    ``shape_base``/``group_base`` are the layer's offsets into the query tables and must
    match training exactly; they come from the checkpoint.

    The temporal window is read from the network, not passed in: a model trained on 3
    stacked frames must be *given* 3 stacked frames, and getting that wrong would show up as
    a mysterious quality loss at inference rather than as an error. Window *alignment* travels
    the same way and for the same reason -- see ``net.RotoNet``.

    The second return is ``(F, G, affine_dim)``, 6 wide for a v1-style document-space head and
    8 for v1.1's crop-space projective one. ``affine_doc`` reads the width to know which it is
    holding.

    The third is the key-timing head's probability per (frame, shape), or ``None`` for a
    checkpoint without one -- which is every checkpoint before v1.3, and the reason the flag
    is read off the network rather than passed in: a caller that asked for key biasing from a
    model that has no key head would otherwise get a silent uniform bias instead of an error.
    """
    out = np.zeros_like(el.points)
    dof = getattr(net, 'affine_dim', 6)
    aff = np.zeros(el.affine.shape[:2] + (dof,), np.float32)
    has_key = bool(getattr(net, 'key_head', False))
    keyp = np.zeros((len(el.frames), el.n_shapes), np.float32) if has_key else None
    P, C = el.points.shape[2], el.points.shape[3]
    dev = next(net.parameters()).device
    shape_ids = (torch.arange(el.n_shapes) + shape_base)[None].to(dev)
    group_ids = (torch.arange(el.n_groups) + group_base)[None].to(dev)
    desc = torch.from_numpy(el.desc)[None].to(dev)
    in_frames = getattr(net, 'in_frames', 1)
    align = bool(getattr(net, 'align_window', False))
    for i in range(0, len(el.frames), batch):
        idx = np.arange(i, min(i + batch, len(el.frames)))
        a = torch.from_numpy(el.window(idx, in_frames, align)).to(dev)
        n = a.shape[0]
        pts, af, klog = net(a, shape_ids.expand(n, -1), group_ids.expand(n, -1),
                            desc.expand(n, -1, -1))
        out[idx] = pts[:, :, :P, :C].cpu().numpy()
        aff[idx] = af[:, :el.n_groups].cpu().numpy()
        if keyp is not None and klog is not None:
            keyp[idx] = torch.sigmoid(klog[:, :el.n_shapes]).cpu().numpy()
    return out, aff, keyp


def predict_crop_points(net: RotoNet, el: LayerData, shape_base: int = 0,
                        group_base: int = 0, batch: int = 8) -> np.ndarray:
    """Points only -- the common case. See :func:`predict`."""
    return predict(net, el, shape_base, group_base, batch)[0]


def to_local(el: LayerData, crop_pts: np.ndarray,
             matrices: np.ndarray | None = None) -> np.ndarray:
    """Crop-space predictions -> IR-native local normalised coordinates.

    ``matrices`` is ``(F, S, 4, 4)`` and defaults to the artist's own composed track. Passing
    the *predicted* track is what makes ``motion='predicted'`` a real de-teacher-forcing: the
    same matrix has to be inverted here and re-applied by the renderer, or the two halves
    disagree and the error is the mismatch rather than the head's.
    """
    mats = el.matrices if matrices is None else matrices
    out = np.zeros_like(crop_pts)
    c = el.crop
    for fi, f in enumerate(el.frames):
        off = c['offsets'][int(f)]
        for si in range(el.n_shapes):
            out[fi, si] = crop_to_local(
                crop_pts[fi, si], mats[fi, si], width=c['width'], height=c['height'],
                offset=off, scale=c['scale'], out_px=c['out_px'])
    return out


def track_jitter(crop_pts: np.ndarray, target: np.ndarray, live: np.ndarray,
                 point_mask: np.ndarray, out_px: float) -> float:
    """Frame-to-frame change in the prediction's *error*, in crop pixels.

    Differencing the error rather than the position is what separates jitter from motion: a
    shape travelling smoothly across the frame has a large position delta and no jitter.
    """
    if len(crop_pts) < 2:
        return 0.0
    err = crop_pts - target
    d = np.linalg.norm(err[1:] - err[:-1], axis=-1) * out_px
    m = point_mask[None] & (live[1:] & live[:-1])[..., None, None]
    return float(d[m].mean()) if m.any() else 0.0


def transform_matrices(el: LayerData, pred_affine: np.ndarray) -> np.ndarray:
    """``(F, G, 4, 4)`` document-space layer matrices from whatever the head predicted.

    The width says which representation is in hand, so one call site serves both and a
    v1 checkpoint keeps scoring exactly as it did:

    * **6 wide** -- v1's document-space affine. Filled straight in, ``m33`` forced to 1.
      That last part is the lossy step; see ``program.affine_from_matrix``.
    * **8 wide** -- v1.1's crop-space projective map. The prediction is the composed
      *local-to-crop* transform, so the window's own map has to be divided back out:
      ``M_doc = M_crop @ inv(C_f)``, with ``C_f`` from ``geometry.crop_matrix``. Exact, and
      per frame, because the crop window translates.
    """
    if pred_affine.shape[-1] == PROJ_DOF:
        crop_mats = matrix_from_proj(pred_affine)               # (F, G, 4, 4), local->crop
        inv = np.linalg.inv(el.crop_matrices)                   # (F, 4, 4)
        return np.einsum('fgij,fjk->fgik', crop_mats, inv)
    return matrix_from_affine(pred_affine)


def with_transforms(doc: RotoDoc, el: LayerData, mats: np.ndarray) -> RotoDoc:
    """Put ``mats`` ``(F, G, 4, 4)`` on every shape's transform carrier, in place.

    One function for both consumers -- :func:`affine_doc` and ``motion='predicted'`` -- so
    there is a single definition of what "render with the predicted track" means.

    A shape's **carrier** is the ancestor whose transform track the group matrix is written
    to. Two rules, and the second one is a v1.2 fix rather than a restatement:

    * If an ancestor already carries a transform, that is the carrier. There must be exactly
      one: two would each receive the group matrix and the renderer would apply it twice,
      which still draws a plausible picture in the wrong place and would be invisible to every
      aggregate. Asserted, not assumed.
    * **If no ancestor carries one, the innermost ancestor becomes the carrier.** v1 and v1.1
      skipped these shapes, and skipping them made the transform head's rendered number
      *flattering* on exactly the layers where it matters: 338 of ``Layer_52``'s 592 shapes and
      174 of ``mb_1``'s 473 have no transformed ancestor, so 513 of the archive's 2,753 shapes
      were rendered at the artist's own position no matter what the head predicted for them.
      Their target is not a no-op -- the crop-space target of an untransformed group is the
      window's own map, which moves every frame -- so the error was real and unscored.

    Writing to the innermost ancestor is only safe if that layer is not also an ancestor of a
    shape in a *different* group, which would have one group's matrix silently move another
    group's shapes. It holds throughout this archive and is pinned by a test.
    """
    for si, (ancestors, _) in enumerate(doc.shapes()):
        g = int(el.group_of[si])
        carriers = [l for l in ancestors if l.transform] or [ancestors[-1]]
        assert len(carriers) == 1, \
            f'{el.layer_id}: shape {si} has {len(carriers)} transformed ancestors; ' \
            'replacing both would apply the group matrix twice'
        carriers[0].transform = [Key(int(f), 'linear', mats[fi, g])
                                 for fi, f in enumerate(el.frames)]
    return doc


def predicted_shape_matrices(el: LayerData, pred_affine: np.ndarray) -> np.ndarray:
    """``(F, S, 4, 4)`` -- the predicted track fanned out per shape, as ``el.matrices`` is.

    ``to_local`` inverts a per-shape matrix; the head predicts per *group*. Groups are exactly
    the equivalence classes of identical composed tracks (``data._group_index``), so the fan-out
    is a lookup rather than a recomposition."""
    return transform_matrices(el, pred_affine)[:, el.group_of]


def affine_doc(el: LayerData, pred_affine: np.ndarray) -> RotoDoc:
    """The artist's own shapes, moved by the *predicted* transform track.

    This is how the transform head gets a number on the table. It is trained but v1 never
    consumed it, which spends gradient on an output nobody reads.

    Isolating it this way is deliberate, and it answers a different question from
    ``RebuildConfig.motion='predicted'`` rather than a weaker version of the same one.
    Holding the artist's control points fixed and varying only the transform is what makes
    the head's error visible in pixels, with no geometry error in front of it.

    **Feed it the artist's own track and this must return 1.000.** It does for v1.1's
    representation and does not for v1's -- 0.9100 on ``FAM red_1`` -- which is how the 6-number
    target was found to be lossy rather than merely badly framed. ``scripts/exp_affine_target.py``
    is that check.
    """
    doc = from_json_ir(json.loads((el.directory / 'target_ir.json').read_text()))
    return with_transforms(doc, el, transform_matrices(el, pred_affine))


def rebuild(el: LayerData, local_pts: np.ndarray, cfg: RebuildConfig | None = None,
            key_prob: np.ndarray | None = None) -> tuple[RotoDoc, dict[str, Any]]:
    """Choose keys per shape and write a document carrying only those keys.

    The keyframe search runs on the point positions only, not on Bezier handles. Handles are
    carried at whatever the chosen keys hold: they describe the curve *between* points, so
    letting them drive key timing would key on tangent wobble the picture barely shows. Two
    of thirteen layers have handles at all.

    Key *timing* is chosen on the smoothed track, because that is a question about structure
    and noise would corrupt it. Key *values* come from the smoothed track too unless
    ``refit_values`` is set, in which case they are fitted to the raw track -- see
    ``keys.refit`` for why storing the filter's own output biases every motion extreme.

    Key scoring reads the artist's keys from a second, untouched copy of the IR, so the
    comparison is never the rebuilt document against itself.

    ``key_prob`` is the key-timing head's per-(frame, shape) probability. It enters only as a
    local reshaping of the DP's tolerance -- tighter where a key is expected (``cfg.key_bias``)
    and looser where none is (``cfg.key_slack``); with both at zero, or no head,
    ``keys.select`` stays on its original code path and the result is v1.2's bit for bit. The head is *also* scored on its own, at ``cfg.key_thresh``, and the two numbers
    are reported side by side -- see ``Reconstruction.head_key_f1``.
    """
    cfg = cfg or RebuildConfig()
    doc = from_json_ir(json.loads((el.directory / 'target_ir.json').read_text()))
    truth_doc = from_json_ir(json.loads((el.directory / 'target_ir.json').read_text()))
    shapes = [s for _, s in doc.shapes()]
    truth_shapes = [s for _, s in truth_doc.shapes()]
    at = {int(f): i for i, f in enumerate(el.frames)}

    n_pred = n_true = 0
    precs, recs, f1s, weights = [], [], [], []
    hprecs, hrecs, hf1s, allf1s, randf1s = [], [], [], [], []
    rand_f1s: list[float] = []
    strict_f1s: list[float] = []
    strict_w: list[float] = []
    n_no_strict = 0
    tol_lo, tol_hi = float('inf'), 0.0
    # Seeded: the random baseline is a number a report quotes, so it has to be reproducible.
    rng = np.random.default_rng(0)
    for si, (shape, tshape) in enumerate(zip(shapes, truth_shapes)):
        live = np.array([int(f) for f in el.frames if el.live[at[int(f)], si]], np.int32)
        P = shape.n_points
        C = int(np.asarray(shape.path[0].value).shape[1])
        interp = {k.frame: k.interp for k in tshape.path}

        def value(frame: int) -> np.ndarray:
            return local_pts[at[int(frame)], si, :P, :C].astype(np.float64)

        if len(live) < 2:
            # Nothing to interpolate: one key at the frame the shape exists on.
            f = int(live[0]) if len(live) else int(el.frames[0])
            shape.path = [Key(f, interp.get(f, 'linear'), value(f))]
            n_pred += 1
            n_true += len({k.frame for k in tshape.path})
            continue

        raw = np.stack([value(f) for f in live])
        clean = smooth_track(raw, cfg.smooth, cfg.smooth_kind)
        # The head's probability on this shape's own live frames, in the order ``select``
        # indexes them. Sampled rather than passed whole because the DP works per shape over
        # its live span, and a probability aligned to the wrong axis would bias the wrong
        # frames -- silently, and in a way only a key F1 could show.
        kp = (key_prob[[at[int(f)] for f in live], si]
              if key_prob is not None and (cfg.key_bias or cfg.key_slack) else None)
        sel = select(clean[:, :, 0, :] * el.px_per_norm, live, cfg.tol_px, kp,
                     cfg.key_bias, cfg.key_slack)
        tol_lo, tol_hi = min(tol_lo, sel.tol_min), max(tol_hi, sel.tol_max)
        modes = [interp.get(int(f), 'linear') for f in sel.frames]
        if cfg.refit_values:
            vals = refit_key_values(raw, sel.frames, modes, live)
        else:
            pick = {int(f): i for i, f in enumerate(live)}
            vals = np.stack([clean[pick[int(f)]] for f in sel.frames])
        shape.path = [Key(int(f), modes[i], vals[i]) for i, f in enumerate(sel.frames)]

        truth = np.array(sorted({int(np.clip(k.frame, live[0], live[-1]))
                                 for k in tshape.path}), np.int32)
        # The same comparison with the out-of-live-range keys *excluded* rather than clipped
        # onto the boundary. 7.91% of this archive's 24,335 artist keys sit outside their
        # shape's live range -- affecting half of all shapes -- and `rebuild` always forces a
        # knot at the first and last live frame, so a clipped key is matched for free. Worth
        # 0.009 to 0.040 key F1 per layer, which is the size of difference this project argues
        # about. `key_f1` keeps the clipped definition, unchanged, because every number in v1,
        # v1.1 and v1.2 used it and the plan's >= 0.42 gate was set against one of them;
        # `key_f1_strict` is the version with the free credit removed. Both are reported.
        strict = np.array(sorted({int(k.frame) for k in tshape.path
                                  if live[0] <= k.frame <= live[-1]}), np.int32)
        if len(strict):
            sp, sr, ss = key_f1(sel.frames, strict, tolerance=1)
            strict_f1s.append(ss); strict_w.append(len(strict))
        else:
            # No in-range artist key at all: there is nothing to find, so the shape is left
            # out of the strict mean rather than scored zero for an impossible task.
            n_no_strict += 1
        p, r, s = key_f1(sel.frames, truth, tolerance=1)
        precs.append(p); recs.append(r); f1s.append(s); weights.append(len(truth))
        # The same trivial control the key *head* now gets, applied to the *pipeline* -- which
        # is the number the acceptance gate reads. Key F1 with a one-frame tolerance is nearly
        # saturated by keying often on a densely-keyed layer, and the two densest layers here
        # carry 40% of the archive's keys. So: what would the same number of keys, placed at
        # random over the same live frames, have scored? Anything the DP earns is the
        # difference. Seeded, because a report quotes it.
        pick = rng.permutation(len(live))[:len(sel.frames)]
        rand_f1s.append(key_f1(np.sort(live[np.sort(pick)]), truth, tolerance=1)[2])
        n_pred += len(sel.frames)
        n_true += len(truth)
        if key_prob is not None:
            # The head alone: its own key set at its own threshold, against the same truth
            # and the same one-frame tolerance, so the two rows are directly comparable.
            pr = key_prob[[at[int(f)] for f in live], si]
            own = live[pr > cfg.key_thresh]
            hp, hr, hs = key_f1(np.asarray(own, np.int32), truth, tolerance=1)
            hprecs.append(hp); hrecs.append(hr); hf1s.append(hs)
            # ...and two trivial baselines on exactly the same comparison, because without
            # them the number above is unreadable. Key density ranges 13x across these layers
            # (0.037 to 0.474), so a head that learns nothing but each layer's *base rate*
            # scores well: measured on the 12k probe, `Layer_52` read 0.803 against 0.797 for
            # firing on every live frame. A head metric that can be matched by "fire on
            # everything" is not measuring the head.
            allf1s.append(key_f1(np.asarray(live, np.int32), truth, tolerance=1)[2])
            n_fire = int((pr > cfg.key_thresh).sum())
            shuffled = rng.permutation(len(live))[:n_fire]
            rnd = np.sort(live[np.sort(shuffled)]) if n_fire else np.zeros(0, np.int32)
            randf1s.append(key_f1(np.asarray(rnd, np.int32), truth, tolerance=1)[2])

    w = np.array(weights, float)
    w = w / w.sum() if w.sum() else w
    stats = {
        'keys_predicted': int(n_pred), 'keys_artist': int(n_true),
        'key_precision': float(np.dot(w, precs)) if len(w) else 0.0,
        'key_recall': float(np.dot(w, recs)) if len(w) else 0.0,
        'key_f1': float(np.dot(w, f1s)) if len(w) else 0.0,
        'key_tol_min': 0.0 if tol_lo == float('inf') else float(tol_lo),
        'key_tol_max': float(tol_hi),
    }
    if len(rand_f1s) == len(w) and len(w):
        stats['baseline_pipeline_key_f1_random'] = float(np.dot(w, rand_f1s))
        stats['key_f1_over_random'] = stats['key_f1'] - stats['baseline_pipeline_key_f1_random']
    if strict_f1s:
        sw = np.array(strict_w, float)
        stats['key_f1_strict'] = float(np.dot(sw / sw.sum(), strict_f1s))
        stats['shapes_with_no_in_range_key'] = int(n_no_strict)
    if len(hf1s) == len(w) and len(w):
        stats.update(head_key_precision=float(np.dot(w, hprecs)),
                     head_key_recall=float(np.dot(w, hrecs)),
                     head_key_f1=float(np.dot(w, hf1s)),
                     # The two numbers that make the one above readable.
                     baseline_key_f1_all_live=float(np.dot(w, allf1s)),
                     baseline_key_f1_random=float(np.dot(w, randf1s)))
        stats['head_key_f1_over_best_baseline'] = (
            stats['head_key_f1'] - max(stats['baseline_key_f1_all_live'],
                                       stats['baseline_key_f1_random']))
    return doc, stats


def score_doc(el: LayerData, doc: RotoDoc, want: Sequence[int],
              supersample: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Render ``doc`` on ``want`` and compare to the stored alpha. Returns (soft, hard).

    The render config comes from the **dataset's own record** of how its alphas were drawn,
    not from a freshly constructed one. Through v1.2 it was the latter, which carried the
    supersample across and quietly reasserted the class defaults for the other three
    conventions. That could not produce a wrong number while ``datasets/v001`` agreed with
    those defaults, and it produces a wrong one on ``v002``: the artist's own program scores
    0.9925 on ``FAM blue_1`` instead of 1.000000, entirely because the scorer filled open
    zero-width shapes the dataset had stroked. See ``render.raster.config_from_meta``.
    """
    c = el.crop
    cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, supersample)
    softs, hards = [], []
    for f in want:
        x0, y0 = c['offsets'][int(f)]
        box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
        pred = render_union(doc, int(f), cfg, c['scale'], box)
        truth = load_alpha(el.directory, int(f))
        if pred.shape != truth.shape:
            pred = pred[:truth.shape[0], :truth.shape[1]]
        softs.append(soft_iou(pred, truth))
        hards.append(iou(pred, truth))
    return np.asarray(softs), np.asarray(hards)


def assemble(el: LayerData, crop_pts: np.ndarray, pred_affine: np.ndarray,
             cfg: RebuildConfig | None = None, frames: Sequence[int] | None = None,
             key_prob: np.ndarray | None = None) -> Reconstruction:
    """Predictions -> keys -> a document -> pixels -> numbers. The scoring half of
    :func:`reconstruct`, split out so it can be driven by arrays instead of a network.

    That split is what lets the strongest available check exist: hand it the *artist's own*
    control points and the artist's own transform track and every number must come out
    perfect. Three of the four bugs v1.1 found were found that way, and the two paths added
    here -- predicted motion, and the percentile metrics -- are checked the same way in
    ``tests/test_v12.py`` rather than trusted.
    """
    cfg = cfg or RebuildConfig()
    if cfg.motion not in MOTION_SOURCES:
        raise ValueError(f'unknown motion {cfg.motion!r}, want one of {MOTION_SOURCES}')
    if cfg.motion == PREDICTED and cfg.predicted_affine:
        raise ValueError("motion='predicted' and predicted_affine are two different "
                         'measurements -- end-to-end, and the head alone on the artist\'s '
                         'own points. Ask for one at a time.')

    mask = el.point_mask[None] & el.live[..., None, None]
    # Euclidean, in crop pixels. This was ``(|dx| + |dy|) / 2 * out_px``, which is neither
    # L1 nor L2 and reads low against the distance an artist would measure.
    err = np.linalg.norm(crop_pts - el.points, axis=-1)[mask] * el.out_px
    point_err, p95 = float(err.mean()), float(np.percentile(err, 95))
    pmax = float(err.max()) if err.size else 0.0
    jitter = track_jitter(crop_pts, el.points, el.live, el.point_mask, el.out_px)

    # The keys are chosen on the local track, so which matrix is inverted here is part of
    # the de-teacher-forcing rather than a detail of it -- see the module docstring.
    mats = predicted_shape_matrices(el, pred_affine) if cfg.motion == PREDICTED else None
    doc, kstats = rebuild(el, to_local(el, crop_pts, mats), cfg, key_prob)
    if cfg.motion == PREDICTED:
        doc = with_transforms(doc, el, transform_matrices(el, pred_affine))
    if cfg.predicted_affine:
        doc = affine_doc(el, pred_affine)

    want = list(frames) if frames is not None else [int(f) for f in el.frames]
    softs, hards = score_doc(el, doc, want, cfg.supersample)
    return Reconstruction(el.layer_id, doc, np.asarray(want), softs, hards,
                          point_err, point_err_p95_px=p95, point_err_max_px=pmax,
                          jitter_px=jitter, **kstats)


def reconstruct(layer_dir: str | Path, net: RotoNet, cfg: RebuildConfig | None = None,
                frames: Sequence[int] | None = None, shape_base: int = 0,
                group_base: int = 0) -> Reconstruction:
    el = load_element(layer_dir)
    crop_pts, pred_aff, key_prob = predict(net, el, shape_base, group_base)
    return assemble(el, crop_pts, pred_aff, cfg, frames, key_prob)
