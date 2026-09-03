"""Reconstruct every element from the trained model, score it, and write the v1 deliverables.

Produces, under the output directory:

    results/reconstruction.json   per-element numbers and the run's totals
    figures/<element_id>.png      artist splines | clean alpha | reconstruction
    figures/contact_sheet.png     up to ten elements on one page

The frame shown in each figure is chosen at random, with a fixed seed, from the frames where
the element actually draws something. Picking the best frame would make the figures a highlight
reel; picking the worst would make them a bug report. A seeded random frame is the only choice
that stays honest when the numbers move.

The coverage filter is not cosmetic. Several elements have frames where every shape is switched
off, and an empty render against an empty target scores a perfect IoU -- so an unfiltered
random pick can put a blank panel, captioned 1.0000, at the top of the contact sheet. That is
a meaningless number in the most prominent position on the page.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from matplotlib import pyplot as plt                                    # noqa: E402

from roto.model.data import load_element                                # noqa: E402
from roto.model.figures import contact_sheet, element_figure            # noqa: E402
from roto.model.reconstruct import (DEFAULT_TOL_PX, load_model,         # noqa: E402
                                    reconstruct)
from roto.sfx.json_ir import from_json_ir                               # noqa: E402


MIN_COVERAGE = 0.02
"""Alpha coverage a frame needs before it can be chosen for a figure.

Below this the element is essentially not drawn at that frame, and both the panel and the IoU
beside it stop meaning anything."""


def pick_frame(element_dir: Path, rng: np.random.Generator) -> int:
    """A seeded random frame on which the element actually draws something."""
    meta = json.loads((element_dir / 'meta.json').read_text())
    frames = meta['frames']['index']
    cover = meta['frames']['coverage']
    ok = [f for f, c in zip(frames, cover) if c >= MIN_COVERAGE]
    if not ok:                                   # never blank-by-choice; fall back to the best
        ok = [frames[int(np.argmax(cover))]]
    return int(ok[int(rng.integers(len(ok)))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--checkpoint', default='v1/model/v1.pt')
    ap.add_argument('--out', default='v1')
    ap.add_argument('--tol', type=float, default=DEFAULT_TOL_PX)
    ap.add_argument('--max-figures', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--figures-only', action='store_true',
                    help='reuse an existing reconstruction.json and redraw the figures; '
                         'renders one frame per element instead of all of them')
    args = ap.parse_args()

    net, ck = load_model(args.checkpoint)
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dirs = sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists())
    rows, sheet = [], []
    for d in dirs:
        frame = pick_frame(d, rng)
        rec = reconstruct(d, net, args.tol, shape_base=sbase.get(d.name, 0),
                          group_base=gbase.get(d.name, 0),
                          frames=[frame] if args.figures_only else None)
        s = rec.summary()
        rows.append(s)
        print(f'{rec.element_id[:52]:<53} soft IoU {s["mean_soft_iou"]:.4f}  '
              f'IoU {s["mean_iou"]:.4f}  pt {s["point_err_px"]:6.2f}px  '
              f'keys x{s["key_ratio"]:.2f}  F1 {s["key_f1"]:.3f}')

        el = load_element(d)
        artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        i = int(np.where(rec.frames == frame)[0][0])
        fig = element_figure(d, artist, rec.doc, frame, el.crop, float(rec.soft_iou[i]),
                             float(rec.iou[i]), el.out_px, rec.element_id,
                             extra=f'{el.n_shapes} shapes')
        fig.savefig(out / 'figures' / f'{rec.element_id}.png', dpi=140,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        sheet.append({'dir': d, 'frame': frame, 'artist': artist, 'model': rec.doc,
                      'crop': el.crop, 'out_px': el.out_px, 'soft': float(rec.soft_iou[i]),
                      'element_id': rec.element_id})

    sheet.sort(key=lambda r: -r['soft'])
    contact_sheet(sheet[:args.max_figures], out / 'figures' / 'contact_sheet.png',
                  f'roto v1 — reconstruction from clean alpha ({len(sheet)} elements)')

    w = np.array([r['frames'] for r in rows], float)
    totals = {
        'elements': len(rows),
        'frames': int(w.sum()),
        'tol_px': args.tol,
        'checkpoint': str(args.checkpoint),
        'n_params': ck.get('n_params'),
        'mean_soft_iou': float(np.average([r['mean_soft_iou'] for r in rows], weights=w)),
        'mean_iou': float(np.average([r['mean_iou'] for r in rows], weights=w)),
        'point_err_px': float(np.average([r['point_err_px'] for r in rows], weights=w)),
        'keys_predicted': int(sum(r['keys_predicted'] for r in rows)),
        'keys_artist': int(sum(r['keys_artist'] for r in rows)),
        'key_f1': float(np.average([r['key_f1'] for r in rows],
                                   weights=[r['keys_artist'] for r in rows])),
        'per_element': rows,
    }
    totals['key_ratio'] = totals['keys_predicted'] / max(1, totals['keys_artist'])
    if not args.figures_only:
        (out / 'results' / 'reconstruction.json').write_text(json.dumps(totals, indent=2))
    print(f'\nOVERALL  soft IoU {totals["mean_soft_iou"]:.4f}   IoU {totals["mean_iou"]:.4f}   '
          f'point {totals["point_err_px"]:.2f}px   keys x{totals["key_ratio"]:.2f}   '
          f'key F1 {totals["key_f1"]:.3f}')
    print(f'figures: {out / "figures"}')


if __name__ == '__main__':
    main()
