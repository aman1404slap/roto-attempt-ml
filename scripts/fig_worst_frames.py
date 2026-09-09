"""The worst frames of the worst layers, as overlays, with each layer's whole trace.

The handover asks for these alongside the regenerated contact sheet, and they answer the
question the contact sheet cannot: the sheet shows a seeded random frame per layer, which is
honest about the *typical* frame and silent about the one that would get the shot sent back.

Reads the run's own `score_*.json` for the per-frame numbers -- no re-scoring -- picks the
lowest-scoring layers, and renders each one's worst frame next to a median frame of the same
layer. `scoring_ceiling.json`, if present, draws the artist's own ceiling on the trace, so a
frame that scores badly because the *dataset* caps it there is visibly different from one the
model got wrong.

    python scripts/fig_worst_frames.py --run final_long_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                                # noqa: E402
from roto.model.figures import worst_frame_figure                       # noqa: E402
from roto.model.reconstruct import (ARTIST, MOTION_SOURCES,             # noqa: E402
                                    RebuildConfig, load_model, reconstruct)
from roto.model.smoothing import BOXCAR, KINDS                          # noqa: E402
from roto.sfx.json_ir import from_json_ir                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v12 import coverage_of, find_run                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='final_long_v2')
    ap.add_argument('--score', default=None,
                    help='score json to read the per-frame numbers from '
                         '(default: v1.2/results/score_<run>.json)')
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--ceiling', default='v1.2/results/scoring_ceiling.json')
    ap.add_argument('--out', default='v1.2/figures/worst_frames.png')
    ap.add_argument('--layers', type=int, default=2, help='how many worst layers to draw')
    # The rebuild settings must match the ones the score json was produced with, or the
    # figure shows a different reconstruction than the numbers it is captioned with.
    ap.add_argument('--tol', type=float, default=None)
    ap.add_argument('--smooth', type=int, default=None)
    ap.add_argument('--smooth-kind', default=BOXCAR, choices=list(KINDS))
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--motion', default=ARTIST, choices=list(MOTION_SOURCES))
    args = ap.parse_args()

    score = Path(args.score or f'v1.2/results/score_{args.run}.json')
    if not score.exists():
        raise SystemExit(f'{score} not found -- score the run first')
    d = json.loads(score.read_text())
    rb = d.get('rebuild', {})
    kw = {k: v for k, v in [('tol_px', args.tol), ('smooth', args.smooth)] if v is not None}
    cfg = RebuildConfig(smooth_kind=args.smooth_kind, refit_values=args.refit,
                        motion=args.motion, **kw)
    for k, v in rb.items():                     # the json is authoritative over the defaults
        if hasattr(cfg, k) and k not in kw and v is not None:
            setattr(cfg, k, v)

    ceil = {}
    if Path(args.ceiling).exists():
        ceil = {r['layer_id']: r for r in json.loads(Path(args.ceiling).read_text())
                ['per_layer']}

    net, ck = load_model(find_run(args.run))
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    worst = sorted(d['per_layer'], key=lambda r: r['mean_soft_iou'])[:args.layers]

    rows = []
    for lay in worst:
        name = lay['layer_id']
        p = Path(args.dataset) / name
        el = load_element(p)
        soft = np.asarray(lay['soft_iou_per_frame'], float)
        frames = np.asarray(lay['frame_index'], int)
        wf = int(frames[int(np.argmin(soft))])
        tf = int(frames[int(np.argsort(soft)[len(soft) // 2])])
        rec = reconstruct(p, net, cfg, frames=[wf, tf],
                          shape_base=sbase.get(name, 0), group_base=gbase.get(name, 0))
        rows.append({
            'dir': p, 'layer_id': f'{name}  ({el.n_shapes} shapes)', 'crop': el.crop,
            'out_px': el.out_px, 'artist': from_json_ir(json.loads(
                (p / 'target_ir.json').read_text())), 'model': rec.doc,
            'worst_frame': wf, 'worst_soft': float(rec.soft_iou[0]),
            'typical_frame': tf, 'typical_soft': float(rec.soft_iou[1]),
            'worst_coverage': coverage_of(p, wf), 'median_coverage': coverage_of(p, tf),
            'frames': frames, 'soft_per_frame': soft,
            'ceiling_per_frame': (np.asarray(ceil[name]['ceiling_per_frame'], float)
                                  if name in ceil else None),
        })
        print(f'{name[:46]:<48} worst {rows[-1]["worst_soft"]:.4f} @{wf} '
              f'({rows[-1]["worst_coverage"] * 100:.1f}% cover)   '
              f'median {rows[-1]["typical_soft"]:.4f} @{tf}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    worst_frame_figure(rows, args.out,
                       f'The frames that would get the shot sent back  —  {args.run}, '
                       f'its {args.layers} worst layers')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
