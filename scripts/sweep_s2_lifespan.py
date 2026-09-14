"""Choose the lifespan decode's operating point, on held-out frames only.

The S2b rung produces a probability per ``(frame, shape)``; turning that into a lifespan needs
three numbers -- ``alive_on``, ``alive_off`` and ``alive_min_gap`` -- and picking them on the
frames the model trained on would be fitting the decode to its own noise. That question was
settled once already at Step 2c, where the key-value refit won charter L4's ranking metric and
lost plan section 4's generalisation check, and the held-out evidence decided it. Same rule
here.

**Why this sweep matters more than the key-timing one did.** The design note measured what a
lifespan mistake costs: 0.0067 soft IoU per 1% of cells drawn wrongly, 0.0215 per 1% omitted,
against a render gate 0.01 wide. And moving every boundary by a single frame -- which is what a
head that has learned *which* shapes come and go, but not exactly *when*, produces -- costs
0.0102 late and 0.0374 early. So the thresholds are not a polish step; they are most of whether
S2b passes, and the asymmetry between them is the whole reason ``alive_off`` is swept separately
from ``alive_on`` rather than pinned to it.

The winner is chosen on **held-frame on-screen soft IoU** and nothing else. Every other column
is printed so the choice can be argued with, including the trained-frame score it is *not*
chosen on -- the gap between the two is what says whether the decode generalises.

    python scripts/sweep_s2_lifespan.py --run runs/v2/s2b_seed1
    python scripts/sweep_s2_lifespan.py --run runs/v2/s2b_seed1 --all-frames   # diagnostic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.v2.build import element_dirs                                # noqa: E402
from roto.v2.reconstruct import (PREDICTED, RebuildConfig, assemble,  # noqa: E402
                                 load_model, predict)
from roto.v2.traindata import load_element                            # noqa: E402

ON = (0.3, 0.5, 0.7)
OFF = (0.05, 0.2, 0.35)
GAP = (0, 3)
"""The grid. ``off <= on`` always -- the reverse makes a shape harder to switch on than off,
which inverts the measured 3.2 : 1 asymmetry, and ``alive_mask`` refuses it outright.

``on`` spans reluctant to eager. ``off`` spans "almost never let go" to "barely any
hysteresis"; 0.35 with ``on`` at 0.5 is nearly a plain threshold and is in the grid as the
control for whether the hysteresis earns its place at all. ``min_gap`` at 3 closes dropouts up
to two frames long, which is the length a one-frame-either-side boundary error produces."""


def cells():
    return [(on, off, gap) for on in ON for off in OFF for gap in GAP if off <= on]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='checkpoint directory, e.g. runs/v2/s2b_seed1')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--all-frames', action='store_true',
                    help='score every frame instead of the held-out ones. DIAGNOSTIC ONLY -- '
                         'a winner chosen this way is fitted to frames the model trained on')
    args = ap.parse_args()

    net, ck = load_model(Path(args.run) / 'model.pt')
    if not ck['arch'].get('alive_head'):
        raise SystemExit(f'{args.run} has no lifespan head; there is nothing to sweep. '
                         'Train an s2b rung first.')
    root = Path(args.dataset or ck['dataset'])
    sbase, gbase = dict(ck.get('shape_base', {})), dict(ck.get('group_base', {}))
    splits, withheld = ck.get('splits', {}), set(ck.get('withheld_elements', []))

    # Predict once per element and re-decode per cell: the network is the expensive part and
    # it does not depend on the thresholds. Only trained elements -- an element with freshly
    # initialised query rows reconstructs at 0.06 soft IoU, so including it here would let
    # noise pick the operating point.
    preds, els = {}, {}
    for d in element_dirs(root):
        el = load_element(d)
        if el.element_id in withheld or el.element_id not in sbase:
            continue
        els[el.element_id] = el
        preds[el.element_id] = predict(net, el, sbase[el.element_id],
                                       gbase.get(el.element_id, 0))
    print(f'{len(els)} trained elements from {root}')

    rows, t0 = [], time.time()
    for on, off, gap in cells():
        cfg = RebuildConfig(lifespan=PREDICTED, alive_on=on, alive_off=off, alive_min_gap=gap)
        held_s, held_w, train_s, train_w = [], [], [], []
        acc, fp, fn, tr_pred, tr_art, cw = [], [], [], 0, 0, []
        for eid, el in els.items():
            p = preds[eid]
            rec = assemble(el, p.points, p.affine, cfg, key_prob=p.key_prob,
                           alive_prob=p.alive_prob, point_count=p.count)
            pos = splits.get(eid, {})
            hp = np.array(pos.get('held', []), int)
            tp = np.array(pos.get('train', []), int)
            # On-screen only, per Step 2c: a frame where the layer is absent scores exactly
            # 1.0 for drawing nothing, and lifespan is precisely the quantity that decides
            # whether anything gets drawn there -- so scoring those would pay the head for
            # its own mistakes.
            on_screen = el.live.any(axis=1)
            for idx, s_acc, w_acc in ((hp, held_s, held_w), (tp, train_s, train_w)):
                keep = idx[on_screen[idx]] if len(idx) else idx
                if len(keep):
                    s_acc.append(float(rec.soft_iou[keep].mean())); w_acc.append(len(keep))
            acc.append(rec.lifespan_accuracy); fp.append(rec.lifespan_fp_rate)
            fn.append(rec.lifespan_fn_rate); cw.append(rec.lifespan_cells)
            tr_pred += rec.lifespan_transitions_pred
            tr_art += rec.lifespan_transitions_artist
        w = np.array(cw, float)
        av = lambda v, ws: float(np.average(v, weights=ws)) if len(v) else float('nan')
        rows.append({
            'alive_on': on, 'alive_off': off, 'alive_min_gap': gap,
            'held_soft_iou': av(held_s, held_w), 'train_soft_iou': av(train_s, train_w),
            'lifespan_accuracy': av(acc, w), 'fp_rate': av(fp, w), 'fn_rate': av(fn, w),
            'transition_ratio': tr_pred / max(1, tr_art),
        })
        r = rows[-1]
        print(f'  on {on:.2f} off {off:.2f} gap {gap}  held {r["held_soft_iou"]:.4f}  '
              f'train {r["train_soft_iou"]:.4f}  acc {r["lifespan_accuracy"]:.4f}  '
              f'FP {r["fp_rate"]:.4f} FN {r["fn_rate"]:.4f}  '
              f'trans {r["transition_ratio"]:.2f}x  [{time.time() - t0:.0f}s]', flush=True)

    key = 'train_soft_iou' if args.all_frames else 'held_soft_iou'
    best = max(rows, key=lambda r: r[key])
    print(f'\nwinner on {key}: alive_on={best["alive_on"]} alive_off={best["alive_off"]} '
          f'alive_min_gap={best["alive_min_gap"]}')
    print(f'  held {best["held_soft_iou"]:.4f}  train {best["train_soft_iou"]:.4f}  '
          f'gap {best["train_soft_iou"] - best["held_soft_iou"]:+.4f}')
    print(f'  lifespan accuracy {best["lifespan_accuracy"]:.4f} '
          f'(FP {best["fp_rate"]:.4f}, FN {best["fn_rate"]:.4f}); '
          f'transitions {best["transition_ratio"]:.2f}x the artist')
    if args.all_frames:
        print('  NOTE: chosen on TRAINED frames. Diagnostic only -- not an operating point.')
    out = args.out or f'{args.run}/lifespan_sweep.json'
    Path(out).write_text(json.dumps(
        {'run': args.run, 'dataset': str(root), 'selected_on': key,
         'winner': best, 'cells': rows}, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
