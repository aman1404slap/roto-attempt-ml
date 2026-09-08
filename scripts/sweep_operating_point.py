"""Re-choose the keyframe operating point against the model's *current* noise floor.

v1 picked 1.0 px tolerance with a 9-frame boxcar, and that choice was correct *for v1's
jitter*. The whole point of v1.1's temporal window and consistency term is to lower that
jitter, which makes the old operating point stale by construction -- the tolerance has to sit
above the model's noise, and the noise moved.

Four axes, because the review named four things that interact and they cannot be chosen
independently:

    smoothing window   1 (off), 5, 9, 13
    filter             boxcar (v1) vs savgol (keeps the peaks artists key)
    tolerance          0.5 .. 4.0 crop px
    key values         from the filtered track (v1) vs refitted to the raw track

The prediction is computed **once per layer** and reused across every combination: it does
not depend on any of these, and re-running it per cell would make the sweep 20x slower for
no different answer.

**Maximising rendered IoU alone is the wrong objective, and v1.1 measured how wrong.** Its
unconstrained optimum lands at 0.61x the artist's key count with key F1 0.36: rendered IoU is
happy with very few keys as long as the interpolation passes through the pixels, because it
never asks *where* the keys are. But the deliverable is an editable file. A key set 39% sparser
than the artist's, placed in positions the artist did not choose, is one an artist has to hunt
through -- and the IoU that bought it is a hundredth they cannot see. So the sweep reports two
answers: the unconstrained best, and the best whose key ratio stays inside
``--key-ratio-range`` (default 0.75-1.30, i.e. from a quarter sparser to a third denser than
the artist). The constrained one is the one to ship; the unconstrained one is kept because
the size of the gap between them is itself the finding.

    python scripts/sweep_operating_point.py --run full
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                               # noqa: E402
from roto.model.reconstruct import (RebuildConfig, load_model, predict,  # noqa: E402
                                    rebuild, score_doc, to_local)
from roto.model.smoothing import BOXCAR, SAVGOL                        # noqa: E402

LAYERS = [
    'FAM_0060_L1_A0003C007_v001__blue',
    'nfl_0200_bg01_v001_compplate_roto_v001__blue',
    'TVC_SHOTS_sh0260_BG01_v003_roto_v02__Layer_52',
]
"""Three layers spanning the quality range, the same three v1 swept, so the two tables are
comparable row for row: a near-perfect small layer, a mid one, and the 592-shape worst case."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='full')
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/results')
    ap.add_argument('--stride', type=int, default=3,
                    help='score every Nth frame; the sweep compares cells, not headlines')
    ap.add_argument('--windows', type=int, nargs='+', default=[1, 5, 9, 13])
    ap.add_argument('--tols', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--key-ratio-range', type=float, nargs=2, default=[0.75, 1.3],
                    metavar=('LO', 'HI'),
                    help='key economy an artist can still edit; the constrained optimum is '
                         'the best cell inside it')
    args = ap.parse_args()
    lo, hi = args.key_ratio_range

    net, ck = load_model(f'v1.1/runs/{args.run}/model.pt')
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    grid = list(itertools.product(args.windows, (BOXCAR, SAVGOL), args.tols, (False, True)))

    rows = []
    t0 = time.time()
    for name in LAYERS:
        d = Path(args.dataset) / name
        el = load_element(d)
        crop_pts, _ = predict(net, el, sbase.get(name, 0), gbase.get(name, 0))
        local = to_local(el, crop_pts)                     # once: independent of every axis
        frames = [int(f) for f in el.frames[::args.stride]]
        print(f'\n=== {name} ({len(el.frames)} frames, scoring {len(frames)}) ===')
        print(f'{"win":>4} {"filter":>7} {"tol":>5} {"refit":>6} {"keys/artist":>12} '
              f'{"F1":>6} {"softIoU":>8}')
        for win, kind, tol, refit in grid:
            if win <= 1 and kind == SAVGOL:
                continue                                   # no window, no filter to choose
            cfg = RebuildConfig(tol_px=tol, smooth=win, smooth_kind=kind,
                                refit_values=refit)
            doc, st = rebuild(el, local, cfg)
            soft, hard = score_doc(el, doc, frames)
            row = {'layer': name, 'window': win, 'filter': kind, 'tol_px': tol,
                   'refit': refit, 'keys_predicted': st['keys_predicted'],
                   'keys_artist': st['keys_artist'],
                   'key_ratio': st['keys_predicted'] / max(1, st['keys_artist']),
                   'key_f1': st['key_f1'], 'mean_soft_iou': float(soft.mean()),
                   'mean_iou': float(hard.mean())}
            rows.append(row)
            print(f'{win:>4} {kind:>7} {tol:>5.1f} {str(refit):>6} '
                  f'{row["key_ratio"]:>11.2f}x {row["key_f1"]:>6.3f} '
                  f'{row["mean_soft_iou"]:>8.4f}')
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f'operating_point_{args.run}.json').write_text(
                json.dumps({'run': args.run, 'stride': args.stride, 'rows': rows}, indent=2))

    # Best cell by mean soft IoU across layers, and the sparsest cell within 0.002 of it.
    # Both matter: an artist cares about the key count, and a hundredth of IoU they cannot
    # see is not worth twice the keys.
    print(f'\n{len(rows)} cells in {time.time() - t0:.0f}s')
    by_cell: dict[tuple, list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r['window'], r['filter'], r['tol_px'], r['refit']), []).append(r)
    agg = [{'cell': k,
            'mean_soft_iou': float(np.mean([x['mean_soft_iou'] for x in v])),
            'key_ratio': float(np.mean([x['key_ratio'] for x in v])),
            'key_f1': float(np.mean([x['key_f1'] for x in v]))}
           for k, v in by_cell.items() if len(v) == len(LAYERS)]
    agg.sort(key=lambda r: -r['mean_soft_iou'])
    best = agg[0]
    frugal = min([a for a in agg if a['mean_soft_iou'] >= best['mean_soft_iou'] - 0.002],
                 key=lambda r: r['key_ratio'])
    # The cell to ship: best rendered IoU among those an artist could still edit. Reported
    # even when the window is empty, so a sweep that cannot satisfy the constraint says so
    # rather than silently falling back to the unconstrained answer.
    inside = [a for a in agg if lo <= a['key_ratio'] <= hi]
    constrained = max(inside, key=lambda r: r['mean_soft_iou']) if inside else None
    fmt = 'window=%s filter=%s tol=%s refit=%s -> %.4f at %.2fx keys, F1 %.3f'
    args_of = lambda r: (*r['cell'], r['mean_soft_iou'], r['key_ratio'], r['key_f1'])
    print('\nbest soft IoU (unconstrained)  ' + fmt % args_of(best))
    print('sparsest within 0.002          ' + fmt % args_of(frugal))
    if constrained:
        print(f'best with keys in [{lo:.2f}, {hi:.2f}]    ' + fmt % args_of(constrained))
        print(f'  the constraint costs {best["mean_soft_iou"] - constrained["mean_soft_iou"]:+.4f} '
              f'soft IoU and buys {constrained["key_f1"] - best["key_f1"]:+.3f} key F1 '
              f'at {constrained["key_ratio"]:.2f}x the artist\'s keys instead of '
              f'{best["key_ratio"]:.2f}x')
    else:
        print(f'no cell has a key ratio in [{lo:.2f}, {hi:.2f}] -- '
              f'the sweep ranges {min(a["key_ratio"] for a in agg):.2f}x to '
              f'{max(a["key_ratio"] for a in agg):.2f}x')
    (Path(args.out) / f'operating_point_{args.run}.json').write_text(json.dumps(
        {'run': args.run, 'stride': args.stride, 'key_ratio_range': [lo, hi],
         'rows': rows, 'aggregate': agg, 'best': best,
         'sparsest_within_0.002': frugal, 'best_constrained': constrained}, indent=2))


if __name__ == '__main__':
    main()
