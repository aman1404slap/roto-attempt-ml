"""Score one trained run: reconstruct every layer, render it, and write the numbers.

One pass produces both splits. Held-out frames are scored from the same reconstruction as
the trained ones -- there is nothing to re-run, because the split is a property of *training*
and the reconstruction is the same document either way. Reporting them side by side is what
distinguishes a network that interpolates its memorisation from one that only recalls it.

    python scripts/report_v11.py --run full
    python scripts/report_v11.py --run final_long --refit --smooth-kind savgol
    python scripts/report_v11.py --run final_long --affine    # the transform head's own error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                                # noqa: E402
from roto.model.reconstruct import (RebuildConfig, load_model,          # noqa: E402
                                    reconstruct)
from roto.model.smoothing import BOXCAR, KINDS                          # noqa: E402
from roto.sfx.json_ir import from_json_ir                               # noqa: E402

MIN_COVERAGE = 0.02
"""Alpha coverage a frame needs before a figure may show it.

Several layers have frames where every shape is switched off, and an empty render against an
empty target scores a perfect IoU -- so an unfiltered random pick can put a blank panel
captioned 1.0000 at the top of the contact sheet."""


def pick_frame(layer_dir, rng):
    """A seeded random frame on which the layer actually draws something.

    Seeded and random, not best or worst: the best frame makes the figures a highlight reel
    and the worst makes them a bug report. This is the only choice that stays honest when the
    numbers move.
    """
    meta = json.loads((layer_dir / 'meta.json').read_text())
    frames, cover = meta['frames']['index'], meta['frames']['coverage']
    ok = [f for f, c in zip(frames, cover) if c >= MIN_COVERAGE]
    if not ok:
        ok = [frames[int(np.argmax(cover))]]
    return int(ok[int(rng.integers(len(ok)))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=None, help='name under v1.1/runs')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/results')
    ap.add_argument('--label', default=None)
    ap.add_argument('--tol', type=float, default=None)
    ap.add_argument('--smooth', type=int, default=None)
    ap.add_argument('--smooth-kind', default=BOXCAR, choices=list(KINDS))
    ap.add_argument('--refit', action='store_true', help='fit key values to the raw track')
    ap.add_argument('--affine', action='store_true',
                    help='score the predicted transform track on the artist\'s own shapes')
    ap.add_argument('--supersample', type=int, default=None)
    ap.add_argument('--figures', action='store_true', help='also draw the comparison figures')
    ap.add_argument('--figures-dir', default='v1.1/figures')
    ap.add_argument('--max-figures', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    ckpt = args.checkpoint or f'v1.1/runs/{args.run}/model.pt'
    label = args.label or args.run or Path(ckpt).stem
    if args.figures:
        global contact_sheet, element_figure, plt
        from matplotlib import pyplot as plt
        from roto.model.figures import contact_sheet, element_figure

    net, ck = load_model(ckpt)
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    splits = ck.get('splits', {})

    kw = {k: v for k, v in [('tol_px', args.tol), ('smooth', args.smooth)] if v is not None}
    cfg = RebuildConfig(smooth_kind=args.smooth_kind, refit_values=args.refit,
                        predicted_affine=args.affine, supersample=args.supersample, **kw)

    dirs = sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists())
    rng = np.random.default_rng(args.seed)
    rows, sheet = [], []
    for d in dirs:
        rec = reconstruct(d, net, cfg, shape_base=sbase.get(d.name, 0),
                          group_base=gbase.get(d.name, 0))
        s = rec.summary()
        held_pos = np.array(splits.get(d.name, {}).get('held', []), int)
        if len(held_pos):
            train_pos = np.array(splits[d.name]['train'], int)
            s['held_soft_iou'] = float(rec.soft_iou[held_pos].mean())
            s['train_soft_iou'] = float(rec.soft_iou[train_pos].mean())
            s['held_frames'] = int(len(held_pos))
        rows.append(s)
        if args.figures:
            fdir = Path(args.figures_dir); fdir.mkdir(parents=True, exist_ok=True)
            el = load_element(d)
            artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
            frame = pick_frame(d, rng)
            i = int(np.where(rec.frames == frame)[0][0])
            fig = element_figure(d, artist, rec.doc, frame, el.crop,
                                 float(rec.soft_iou[i]), float(rec.iou[i]), el.out_px,
                                 rec.layer_id, extra=f'{el.n_shapes} shapes')
            fig.savefig(fdir / f'{rec.layer_id}.png', dpi=140,
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            sheet.append({'dir': d, 'frame': frame, 'artist': artist, 'model': rec.doc,
                          'crop': el.crop, 'out_px': el.out_px,
                          'soft': float(rec.soft_iou[i]), 'layer_id': rec.layer_id})
        gap = (f'  held {s["held_soft_iou"]:.4f} (train {s["train_soft_iou"]:.4f})'
               if 'held_soft_iou' in s else '')
        print(f'{rec.layer_id[:52]:<53} soft IoU {s["mean_soft_iou"]:.4f}  '
              f'IoU {s["mean_iou"]:.4f}  pt {s["point_err_px"]:6.2f}px  '
              f'jit {s["jitter_px"]:5.2f}px  keys x{s["key_ratio"]:.2f}  '
              f'F1 {s["key_f1"]:.3f}{gap}')

    w = np.array([r['frames'] for r in rows], float)
    av = lambda k, wt=w: float(np.average([r[k] for r in rows], weights=wt))
    totals = {
        'label': label, 'checkpoint': str(ckpt), 'rebuild': asdict(cfg),
        'layers': len(rows), 'frames': int(w.sum()),
        'train_config': ck.get('config'), 'n_params': ck.get('n_params'),
        'mean_soft_iou': av('mean_soft_iou'), 'mean_iou': av('mean_iou'),
        'point_err_px': av('point_err_px'), 'jitter_px': av('jitter_px'),
        'keys_predicted': int(sum(r['keys_predicted'] for r in rows)),
        'keys_artist': int(sum(r['keys_artist'] for r in rows)),
        'key_f1': av('key_f1', np.array([r['keys_artist'] for r in rows], float)),
        'per_layer': rows,
    }
    totals['key_ratio'] = totals['keys_predicted'] / max(1, totals['keys_artist'])
    if any('held_soft_iou' in r for r in rows):
        hw = np.array([r.get('held_frames', 0) for r in rows], float)
        totals['held_soft_iou'] = float(np.average(
            [r.get('held_soft_iou', 0.0) for r in rows], weights=hw))
        totals['train_soft_iou'] = float(np.average(
            [r.get('train_soft_iou', 0.0) for r in rows], weights=hw))
        totals['held_frames'] = int(hw.sum())

    if args.figures and sheet:
        sheet.sort(key=lambda r: -r['soft'])
        contact_sheet(sheet[:args.max_figures],
                      Path(args.figures_dir) / 'contact_sheet.png',
                      f'roto v1.1 — {label} ({len(sheet)} layers)')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    suffix = ''.join(['_affine' if args.affine else '',
                      '_refit' if args.refit else '',
                      f'_{args.smooth_kind}' if args.smooth_kind != BOXCAR else '',
                      f'_ss{args.supersample}' if args.supersample else ''])
    (out / f'score_{label}{suffix}.json').write_text(json.dumps(totals, indent=2))
    extra = (f'   held {totals["held_soft_iou"]:.4f} vs train {totals["train_soft_iou"]:.4f}'
             if 'held_soft_iou' in totals else '')
    print(f'\n{label}{suffix}  OVERALL  soft IoU {totals["mean_soft_iou"]:.4f}   '
          f'IoU {totals["mean_iou"]:.4f}   point {totals["point_err_px"]:.2f}px   '
          f'jitter {totals["jitter_px"]:.2f}px   keys x{totals["key_ratio"]:.2f}   '
          f'key F1 {totals["key_f1"]:.3f}{extra}')


if __name__ == '__main__':
    main()
