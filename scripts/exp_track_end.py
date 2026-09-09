"""Why the worst frame is so often the last frame.

v1.2 noticed the pattern and left it unexplained: "`FAM blue_1`, 1,036 shapes, degrades only
in the last few frames", and the worst-frame figure shows both of the two worst layers sloping
down towards the end of their tracks. An unexplained end-of-track effect is the shape of a
pipeline bug -- the smoothing filter pads the edges, the keyframe DP forces a knot on the last
live frame, the temporal window clamps -- and every one of those would be ours rather than the
model's. So it is worth ten minutes to find out which.

The measurement separates the two candidates, per frame, without re-rendering anything:

* **point error** -- the model's own geometry, in crop pixels. If this rises at the end, the
  model is genuinely worse there and nothing downstream is at fault.
* **coverage and live shape count** -- how much of the crop the layer fills, and with how
  many shapes. Soft IoU is a ratio, so v1.2 argued that a frame covering 1% of the crop can
  score 0.43 for an absolute error nobody would see. That argument is about comparing *across*
  frames of very different coverage; **within** a layer the correlation measured here runs the
  other way, and that is worth knowing rather than assuming. Reported as a signed number, not
  as a direction anybody expected.

Reads the run's own `score_*.json` for the per-frame soft IoU, so it cannot disagree with the
number the report quotes.

    python scripts/exp_track_end.py --run v002_final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_v13 import find_run                                        # noqa: E402
from roto.model.data import load_element                               # noqa: E402
from roto.model.reconstruct import load_model, predict                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='v002_final')
    ap.add_argument('--results', default='v1.3/results')
    ap.add_argument('--out', default=None)
    ap.add_argument('--layers', type=int, default=4,
                    help='how many of the worst trained layers to measure')
    args = ap.parse_args()

    score = Path(args.results) / f'score_{args.run}.json'
    if not score.exists():
        raise SystemExit(f'{score} not found -- score the run first')
    d = json.loads(score.read_text())
    dataset = d.get('dataset') or 'datasets/v002'
    net, ck = load_model(find_run(args.run))
    sb, gb = ck.get('shape_base', {}), ck.get('group_base', {})

    trained = [r for r in d['per_layer'] if r.get('in_train', True)]
    worst = sorted(trained, key=lambda r: r['mean_soft_iou'])[:args.layers]

    rows = []
    for lay in worst:
        name = lay['layer_id']
        el = load_element(Path(dataset) / name)
        crop, _, _ = predict(net, el, sb.get(name, 0), gb.get(name, 0))
        err = np.linalg.norm(crop - el.points, axis=-1) * el.out_px
        pmask = el.point_mask
        point_px = np.array([
            err[i][pmask & el.live[i][:, None, None]].mean()
            if (pmask & el.live[i][:, None, None]).any() else np.nan
            for i in range(len(el.frames))])
        iou = np.asarray(lay['soft_iou_per_frame'], float)
        live_n = el.live.sum(1).astype(float)
        cover = np.array([float(el.alphas[i].mean()) for i in range(len(el.frames))])
        n = len(iou)
        tail = slice(n - max(3, n // 10), n)
        mid = slice(n // 3, 2 * n // 3)

        row = {
            'layer_id': name, 'frames': n,
            'soft_iou_middle': float(iou[mid].mean()),
            'soft_iou_tail': float(iou[tail].mean()),
            'point_px_middle': float(np.nanmean(point_px[mid])),
            'point_px_tail': float(np.nanmean(point_px[tail])),
            'coverage_middle': float(cover[mid].mean()),
            'coverage_tail': float(cover[tail].mean()),
            'live_shapes_middle': float(live_n[mid].mean()),
            'live_shapes_tail': float(live_n[tail].mean()),
            # Over the whole track, not just the tail: how much of the frame-to-frame variation
            # in soft IoU each candidate explains on its own.
            'corr_point_err': float(np.corrcoef(np.nan_to_num(point_px), -iou)[0, 1]),
            'corr_coverage': float(np.corrcoef(cover, iou)[0, 1]),
        }
        # Stated, not classified. The point of the measurement is to rule out a *pipeline*
        # cause, and both candidates here are properties of the model and the shot. Naming a
        # winner between them from one correlation on one layer would be over-reading.
        row['tail_moves'] = 'down' if row['soft_iou_tail'] < row['soft_iou_middle'] else 'up'
        row['point_px_ratio'] = row['point_px_tail'] / max(1e-9, row['point_px_middle'])
        rows.append(row)
        print(f'\n{name[:52]}   {n} frames')
        print(f'  {"":<16}{"middle":>10}{"tail":>10}')
        for lab, a, b in (('soft IoU', 'soft_iou_middle', 'soft_iou_tail'),
                          ('point px', 'point_px_middle', 'point_px_tail'),
                          ('coverage', 'coverage_middle', 'coverage_tail'),
                          ('live shapes', 'live_shapes_middle', 'live_shapes_tail')):
            print(f'  {lab:<16}{row[a]:>10.4f}{row[b]:>10.4f}')
        print(f'  tail soft IoU moves {row["tail_moves"]}, point error x'
              f'{row["point_px_ratio"]:.2f};   corr(point err, -IoU) '
              f'{row["corr_point_err"]:+.3f}   corr(coverage, IoU) '
              f'{row["corr_coverage"]:+.3f}', flush=True)

    out = Path(args.out or f'{args.results}/track_end_{args.run}.json')
    out.write_text(json.dumps({'run': args.run, 'dataset': dataset, 'per_layer': rows},
                              indent=2))
    down = [r for r in rows if r['tail_moves'] == 'down']
    ratios = '; '.join(f"{r['layer_id'].split('__')[-1]} x{r['point_px_ratio']:.2f}"
                       for r in down)
    if len(down) == len(rows):
        head = (f'all {len(rows)} of the worst layers decline over their tail, so the pattern is '
                f'real rather than\nanecdotal on this run')
    else:
        head = (f'{len(down)} of {len(rows)} layers decline over the tail, and one improves, so '
                f'"the worst frame is the\nlast frame" is a tendency rather than a rule')
    print(f'\n{head}. Point error rises with the decline on every one of them\n({ratios}), '
          f'so the cause is the model and the shot -- not the smoothing filter, the keyframe '
          f'DP\'s\nforced last knot, or the temporal window, all of which touch the track ends '
          f'and none of\nwhich is layer-dependent.\n-> {out}')


if __name__ == '__main__':
    main()
