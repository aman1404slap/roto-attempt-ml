"""v1.1 training: one shared network over every layer, batched a frame at a time.

Batches are drawn from a single layer, because the shape count sets the query count and
mixing layers in one batch would mean padding a 4-shape layer out to 1036. Each step samples
one layer and, depending on ``sampling``, either scattered frames or adjacent pairs.

How those frames are drawn turned out to matter more than it looks. A temporal term needs
two adjacent frames in the same batch to have anything to act on, and the obvious way to get
them -- one contiguous run -- measurably hurts: six consecutive frames of a roto layer are
nearly the same picture, so the step carries much less information than six scattered ones.
Sampling ``batch // 2`` anchors *with their successors* buys the adjacency the term needs
while keeping most of the diversity. See :func:`sample_indices`.

The loss is in **crop pixels**: predictions live in [0,1] across the alpha, so multiplying by
the crop's own pixel size makes one weight meaningful for every layer. Layer scales differ by
35x in document terms -- ``px_per_norm`` runs from 124 to 4365 -- and a loss stated in those
units would spend all its capacity on the largest layer and report that as progress.

Four terms, all in the same unit except the affine one:

* **point** -- L1 on control points, v1's only geometry term.
* **curve** -- L1 between the *drawn polylines*, via a fixed linear map. See ``curveloss``.
* **temporal** -- ``|dpred - dtarget|`` between adjacent frames. Punishes jitter without
  punishing real motion, because it is the *change* that is compared, not the position.
* **affine** -- the transform track. ``affine_space='doc'`` is v1's: L1 on the 6 document-space
  matrix entries, lifted by 20 because they are O(1) and unitless next to a loss in pixels.
  ``'crop'`` is v1.1's, and it removes both problems at once -- the target becomes the
  local-to-crop projective map (:func:`~roto.model.geometry.crop_matrix`), which is the space
  the picture depicts, and the loss becomes the distance the predicted transform *moves five
  probe points*, in crop pixels. That is the same unit as the point term, so the weight stops
  being a free parameter and the term stops being separately scaled guesswork.

Only *live* shapes contribute. A shape switched off at a frame has no meaningful control
points there: the artist left them wherever they last were, and asking the network to match
that teaches it to memorise a value nobody draws.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..program import AFFINE_DOF, PROJ_DOF
from .curveloss import PolylineMaps
from .curveloss import curve_loss as polyline_loss
from .data import LayerData, load_dataset
from .net import RotoNet


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
    steps: int = 6000
    batch: int = 6
    lr: float = 3e-4
    dim: int = 192
    depth: int = 3
    affine_weight: float = 20.0
    """Weight on the transform term.

    With ``affine_space='doc'`` this has to be a lift: the entries are O(1) and unitless while
    point error is in pixels, so without one the term is noise next to the geometry term and
    the track never converges. 20 was picked to make it audible, not measured.

    With ``affine_space='crop'`` the term is already in crop pixels, so 1.0 means "a pixel of
    transform error costs what a pixel of point error costs" -- the same reading
    ``temporal_weight`` has. Leaving it at 20 there would price the transform above the
    geometry by 20x."""
    affine_space: str = DOC
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
    in_frames: int = 1
    """Consecutive alphas stacked as input channels. 1 reproduces v1."""
    align_window: bool = False
    """Warp each window neighbour into the anchor's crop window before stacking.

    Off reproduces v1.1's first ladder, where the window was tested *misaligned* and read as
    worthless. See ``data.LayerData.window``: the crop offset twitches 1.09 crop px per step
    on average, which is larger than the 0.77 px of jitter the window was meant to remove, so
    the three channels disagreed about where the shape was by more than the quantity under
    test. Only meaningful when ``in_frames > 1``."""
    self_attn: bool = False
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
    curve_weight: float = 0.0
    """Weight on the polyline term, relative to the point term. 0.5 was the starting point;
    the point term is kept because it is what pins down a control polygon the artist can edit,
    while the curve term is what the render actually judges."""
    temporal_weight: float = 0.0
    """Weight on the frame-to-frame consistency term."""
    holdout_every: int = 0
    """Withhold every Nth frame from training and score it separately. 0 holds nothing."""
    sampling: str = RANDOM
    """How a step's frames are drawn: ``'random'`` (v1), ``'pairs'``, or ``'runs'``.
    See :func:`sample_indices` -- the temporal term needs ``'pairs'`` to have anything to
    act on, and ``'runs'`` is kept only because its cost is worth having on record."""
    sample_weight: str = 'frames'
    """``'frames'`` (v1: proportional to frame count) or ``'sqrt'`` (sqrt(frames * shapes)).

    v1's weighting let the two giant layers dominate wall-clock while the tiny ones overfit.
    ``'sqrt'`` compresses that range; which one wins is measured, not assumed."""
    log_every: int = 250
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


def _batch(el: LayerData, idx: np.ndarray, shape_base: int = 0, group_base: int = 0,
           in_frames: int = 1, device: torch.device | str = 'cpu',
           align: bool = False) -> dict[str, torch.Tensor]:
    a = torch.from_numpy(el.window(idx, in_frames, align))
    to = lambda t: t.to(device, non_blocking=True)
    return {
        'alpha': to(a),
        'points': to(torch.from_numpy(el.points[idx])),
        'live': to(torch.from_numpy(el.live[idx])),
        'affine': to(torch.from_numpy(el.affine[idx])),
        'proj_crop': to(torch.from_numpy(el.proj_crop[idx])),
        'probe': to(torch.from_numpy(el.probe_local)),
        'group_live': to(torch.from_numpy(el.group_live[idx])),
        'frames': to(torch.from_numpy(el.frames[idx].astype(np.int64))),
        'shape_ids': to((torch.arange(el.n_shapes) + shape_base)[None].expand(len(idx), -1)),
        'group_ids': to((torch.arange(el.n_groups) + group_base)[None].expand(len(idx), -1)),
        'desc': to(torch.from_numpy(el.desc)[None].expand(len(idx), -1, -1)),
        'point_mask': to(torch.from_numpy(el.point_mask)[None]),
    }


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


def losses(pred_pts: torch.Tensor, pred_aff: torch.Tensor, b: dict[str, torch.Tensor],
           out_px: float, cfg: TrainConfig,
           maps: PolylineMaps | None = None) -> tuple[torch.Tensor, dict[str, float]]:
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
    total = point_px + cfg.affine_weight * aff

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

    parts['total'] = float(total.detach())
    return total, parts


def layer_weights(els: Sequence[LayerData], mode: str) -> np.ndarray:
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

    els = load_dataset(dataset_root, with_local=False)   # training never reads it
    for e in els:
        _ = e.alphas                                   # materialise once, up front
    # Each layer owns a contiguous block of the query tables, so shape identity is per
    # (layer, shape) rather than per index. See net.RotoNet.
    shape_base, group_base, ns, ng = {}, {}, 0, 0
    for e in els:
        shape_base[e.layer_id], group_base[e.layer_id] = ns, ng
        ns += e.n_shapes
        ng += e.n_groups
    max_shapes, max_groups = ns, ng
    max_points = max(e.points.shape[2] for e in els)
    max_coords = max(e.points.shape[3] for e in els)
    splits = {e.layer_id: e.split(cfg.holdout_every) for e in els}
    n_held = sum(len(h) for _, h in splits.values())
    affine_dim = PROJ_DOF if cfg.affine_space == CROP else AFFINE_DOF
    print(f'{len(els)} layers | {sum(len(e.frames) for e in els)} frames '
          f'({n_held} held out) | queries {max_shapes} shape / {max_groups} group  '
          f'Pmax {max_points} Cmax {max_coords} | {device} | window {cfg.in_frames}'
          f'{" aligned" if cfg.align_window and cfg.in_frames > 1 else ""} '
          f'| self-attn {cfg.self_attn} | sampling {cfg.sampling} '
          f'| transform {cfg.affine_space} ({affine_dim}-dof'
          f'{f", decoder depth {cfg.affine_depth}" if cfg.affine_depth else ""})')

    net = RotoNet(max_shapes, max_groups, max_points, max_coords, cfg.dim, cfg.depth,
                  in_frames=cfg.in_frames, self_attn=cfg.self_attn,
                  affine_dim=affine_dim, align_window=cfg.align_window,
                  affine_depth=cfg.affine_depth or None).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    maps = {e.layer_id: PolylineMaps(e.n_points_per_shape, e.closed_per_shape,
                                     e.coords_per_shape, device)
            for e in els} if cfg.curve_weight else {}
    if maps:
        print(f'  curve loss covers {sum(m.n_shapes_covered for m in maps.values())} '
              f'of {max_shapes} shapes (Bezier and <3-point shapes sit out)')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup)
        * (0.5 * (1 + np.cos(np.pi * min(1.0, s / cfg.steps)))))

    weights = layer_weights(els, cfg.sample_weight)
    state = TrainState()
    t0 = time.time()
    run: dict[str, list[float]] = {}

    for step in range(cfg.steps):
        ei = int(rng.choice(len(els), p=weights))
        el = els[ei]
        idx = sample_indices(rng, splits[el.layer_id][0], cfg.batch, cfg.sampling)
        b = _batch(el, idx, shape_base[el.layer_id], group_base[el.layer_id],
                   cfg.in_frames, device, cfg.align_window)

        pts, aff = net(b['alpha'], b['shape_ids'], b['group_ids'], b['desc'])
        pts = pts[:, :, :el.points.shape[2], :el.points.shape[3]]
        loss, parts = losses(pts, aff, b, el.out_px, cfg, maps.get(el.layer_id))

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
                 'affine_depth': net.affine_depth},
        'config': asdict(cfg),
        'layers': [e.layer_id for e in els],
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
