"""v1 training: one shared network over every layer, batched a frame at a time.

Batches are drawn from a single layer, because the shape count sets the query count and
mixing layers in one batch would mean padding a 4-shape layer out to 1036. Each step
samples one layer (weighted by frame count, so a 191-frame layer is seen more often than
a 77-frame one) and a random batch of its frames.

The loss is in **crop pixels**: predictions live in [0,1] across the alpha, so multiplying by
the crop's own pixel size makes one weight meaningful for every layer. RotoLayer scales differ
by 35x in document terms -- ``px_per_norm`` runs from 124 to 4365 -- and a loss stated in those
units would spend all its capacity on the largest layer and report that as progress.

Only *live* shapes contribute. A shape that is switched off at a frame has no meaningful
control points there: the artist left them wherever they last were, and asking the network to
match that teaches it to memorise a value nobody draws.
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

from .data import LayerData, load_dataset
from .net import RotoNet


@dataclass(slots=True)
class TrainConfig:
    steps: int = 6000
    batch: int = 6
    lr: float = 3e-4
    dim: int = 192
    depth: int = 3
    affine_weight: float = 20.0
    """Affine entries are O(1) and unitless while point error is in pixels; without a lift the
    transform term is noise next to the geometry term and the track never converges."""
    log_every: int = 250
    seed: int = 0
    warmup: int = 200


@dataclass
class TrainState:
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def _batch(el: LayerData, idx: np.ndarray, shape_base: int = 0,
           group_base: int = 0) -> dict[str, torch.Tensor]:
    a = torch.from_numpy(el.alphas[idx]).unsqueeze(1)
    return {
        'alpha': a,
        'points': torch.from_numpy(el.points[idx]),
        'live': torch.from_numpy(el.live[idx]),
        'affine': torch.from_numpy(el.affine[idx]),
        'shape_ids': (torch.arange(el.n_shapes) + shape_base)[None].expand(len(idx), -1),
        'group_ids': (torch.arange(el.n_groups) + group_base)[None].expand(len(idx), -1),
        'desc': torch.from_numpy(el.desc)[None].expand(len(idx), -1, -1),
        'point_mask': torch.from_numpy(el.point_mask)[None],
    }


def losses(pred_pts: torch.Tensor, pred_aff: torch.Tensor, b: dict[str, torch.Tensor],
           out_px: float, affine_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    mask = (b['point_mask'] & b['live'][..., None, None]).unsqueeze(-1)
    n = mask.sum().clamp(min=1)
    point_px = (((pred_pts - b['points']).abs() * mask).sum() / n) * out_px
    aff = (pred_aff - b['affine']).abs().mean()
    total = point_px + affine_weight * aff
    return total, {'point_px': float(point_px.detach()), 'affine': float(aff.detach()),
                   'total': float(total.detach())}


def train(dataset_root: str | Path, out_dir: str | Path,
          cfg: TrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    els = load_dataset(dataset_root)
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
    print(f'{len(els)} layers | {sum(len(e.frames) for e in els)} frames | '
          f'queries {max_shapes} shape / {max_groups} group  Pmax {max_points} '
          f'Cmax {max_coords}')

    net = RotoNet(max_shapes, max_groups, max_points, max_coords, cfg.dim, cfg.depth)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup)
        * (0.5 * (1 + np.cos(np.pi * min(1.0, s / cfg.steps)))))

    weights = np.array([len(e.frames) for e in els], float)
    weights /= weights.sum()
    state = TrainState()
    t0 = time.time()
    run = {'point_px': [], 'affine': [], 'total': []}

    for step in range(cfg.steps):
        ei = int(rng.choice(len(els), p=weights))
        el = els[ei]
        idx = rng.choice(len(el.frames), size=min(cfg.batch, len(el.frames)), replace=False)
        b = _batch(el, np.sort(idx), shape_base[el.layer_id],
                   group_base[el.layer_id])

        pts, aff = net(b['alpha'], b['shape_ids'], b['group_ids'], b['desc'])
        pts = pts[:, :, :el.points.shape[2], :el.points.shape[3]]
        loss, parts = losses(pts, aff, b, el.out_px, cfg.affine_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k].append(v)
        state.step = step + 1

        if (step + 1) % cfg.log_every == 0:
            row = {'step': step + 1, 'lr': sched.get_last_lr()[0],
                   'elapsed_s': round(time.time() - t0, 1),
                   **{k: float(np.mean(v[-cfg.log_every:])) for k, v in run.items()}}
            state.history.append(row)
            print(f'  step {row["step"]:>5}  point {row["point_px"]:7.3f}px  '
                  f'affine {row["affine"]:.5f}  total {row["total"]:7.3f}  '
                  f'[{row["elapsed_s"]:.0f}s]')

    ckpt = {
        'state_dict': net.state_dict(),
        'arch': {'max_shapes': max_shapes, 'max_groups': max_groups,
                 'max_points': max_points, 'max_coords': max_coords,
                 'dim': cfg.dim, 'depth': cfg.depth},
        'config': asdict(cfg),
        'layers': [e.layer_id for e in els],
        'shape_base': shape_base,
        'group_base': group_base,
        'n_params': n_params,
    }
    torch.save(ckpt, out / 'v1.pt')
    summary = {'n_params': n_params, 'steps': cfg.steps, 'layers': len(els),
               'frames': sum(len(e.frames) for e in els),
               'wall_clock_s': round(time.time() - t0, 1),
               'final': state.history[-1] if state.history else {},
               'history': state.history}
    (out / 'train_log.json').write_text(json.dumps(summary, indent=2))
    print(f'\nsaved {out / "v1.pt"}  ({n_params/1e6:.2f}M params, '
          f'{summary["wall_clock_s"]:.0f}s)')
    return summary
