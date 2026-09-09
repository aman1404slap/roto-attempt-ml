"""Re-anchor every past number once, against the rebuilt dataset.

The plan's S2 asks for `v002_control` as "the new baseline; every past number re-anchored once".
Doing that honestly needs one correction that is easy to skip: **a v002 run and a v001 run are
not scored on the same layers.** `datasets/v002` withholds two layers from training, so a v002
row is a mean over 11 layers and every v1 / v1.1 / v1.2 row is a mean over 13. Comparing the
two directly would credit the rebuild with whatever the two dropped layers were worth, which on
this archive is not small -- `nfl_0080 mb_1` is 473 shapes and 154 frames.

So this script restricts the old rows to the same 11 layers and re-aggregates them with the
same weights, from the per-layer numbers already stored in `v1.2/results/score_*.json`. Nothing
is re-rendered and nothing is re-trained: `report_v12.py` kept `per_layer` on every run
precisely so a question like this costs nothing later.

Two comparisons come out, and only the second is a fair one:

* **as published** -- what each report quoted, over whatever layers it used. Useful only for
  finding a row in an old document.
* **on the common layers** -- the same 11 layers, same frame weighting. This is the
  re-baseline, and it is the number to quote from here on.

    python scripts/rebaseline_v13.py
    python scripts/rebaseline_v13.py --against v002_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OLD = [
    ('v1_control', 'v1.2/results/score_v1_control.json',
     "v1's configuration, 12k, seed 0"),
    ('v1_control_long', 'v1.2/results/score_v1_control_long.json',
     "v1's configuration, 40k, seed 0"),
    ('control_long_s1', 'v1.2/results/score_control_long_s1.json',
     'the same, seed 1'),
    ('control_long_s2', 'v1.2/results/score_control_long_s2.json',
     'the same, seed 2'),
    ('final_long', 'v1.2/results/score_final_long.json',
     "v1.1's headline: sqrt weighting, 40k"),
    ('final_long_v2', 'v1.2/results/score_final_long_v2.json',
     '+ aligned window + crop-space transform, 40k'),
    ('final_v2_s1', 'v1.2/results/score_final_v2_s1.json', 'the same, seed 1'),
    ('final_long_v2_constrained_refit',
     'v1.2/results/score_final_long_v2_constrained_refit.json',
     "v1.2's quoted operating point"),
]
"""The rows earlier rounds quoted. Read from their stored per-layer numbers, not re-scored."""

METRICS = ('mean_soft_iou', 'point_err_px', 'p95_point_err_px', 'jitter_px', 'key_f1')


def aggregate(rows: list[dict]) -> dict:
    """The same weighting `model.report.run_totals` uses: frames, except keys, which use keys."""
    w = np.array([r['frames'] for r in rows], float)
    kw = np.array([r['keys_artist'] for r in rows], float)
    av = lambda k, wt=w: float(np.average([r[k] for r in rows], weights=wt))
    return {
        'layers': len(rows), 'frames': int(w.sum()),
        'mean_soft_iou': av('mean_soft_iou'),
        'worst_layer_soft_iou': min(r['mean_soft_iou'] for r in rows),
        'worst_frame_soft_iou': min(r['min_soft_iou'] for r in rows),
        'point_err_px': av('point_err_px'),
        'p95_point_err_px': av('p95_point_err_px'),
        'jitter_px': av('jitter_px'),
        'key_f1': av('key_f1', kw),
        'key_ratio': (sum(r['keys_predicted'] for r in rows)
                      / max(1, sum(r['keys_artist'] for r in rows))),
        'frames_below_0.90': int(sum(r['frames_below_0.90'] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--against', default='v002_control',
                    help='the v002 run whose trained layers define the common set')
    ap.add_argument('--results', default='v1.3/results')
    ap.add_argument('--out', default='v1.3/results/rebaseline.json')
    args = ap.parse_args()

    anchor_path = Path(args.results) / f'score_{args.against}.json'
    if not anchor_path.exists():
        raise SystemExit(f'no {anchor_path} -- score the anchor run first')
    anchor = json.loads(anchor_path.read_text())
    common = sorted(r['layer_id'] for r in anchor['per_layer'] if r['in_train'])

    rows = []
    for name, path, what in OLD:
        p = Path(path)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        per = {r['layer_id']: r for r in d['per_layer']}
        missing = [l for l in common if l not in per]
        if missing:
            print(f'{name}: missing {len(missing)} of the common layers -- skipped')
            continue
        rows.append({'run': name, 'what': what, 'dataset': 'datasets/v001',
                     'as_published': {k: d[k] for k in METRICS if k in d},
                     'published_layers': d['layers'],
                     'common': aggregate([per[l] for l in common])})

    # v1.3's own rungs that belong on the *old* dataset -- the backfill seeds, whose whole
    # purpose is to finish a v1.1 comparison. They are scored into v1.3/results but they are
    # v001 rows, so they are labelled as such rather than swept up with the v002 glob.
    for p in sorted(Path(args.results).glob('score_final_long_s*.json')):
        if any(x in p.stem for x in ('_affine', '_e2e', '_kb', '_ship')):
            continue
        d = json.loads(p.read_text())
        per = {r['layer_id']: r for r in d['per_layer']}
        if any(l not in per for l in common):
            continue
        rows.append({'run': d['label'], 'what': "v1.1's final_long at a second seed (v1.3)",
                     'dataset': 'datasets/v001',
                     'as_published': {k: d[k] for k in METRICS if k in d},
                     'published_layers': d['layers'],
                     'common': aggregate([per[l] for l in common])})

    # Every v002 run scored so far, on the same common set.
    for p in sorted(Path(args.results).glob('score_v002_*.json')):
        d = json.loads(p.read_text())
        if any(x in p.stem for x in ('_affine', '_e2e', '_kb')):
            continue
        per = {r['layer_id']: r for r in d['per_layer']}
        if any(l not in per for l in common):
            continue
        rows.append({'run': d['label'], 'what': 'v1.3', 'dataset': d.get('dataset'),
                     'as_published': {k: d[k] for k in METRICS if k in d},
                     'published_layers': d['layers'],
                     'common': aggregate([per[l] for l in common])})

    anchor_row = next((r for r in rows if r['run'] == args.against), None)
    payload = {'common_layers': common, 'anchor': args.against, 'rows': rows}
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f'\n=== re-baselined on the {len(common)} layers every run trains on ===')
    print(f'{"run":<34} {"pub":>7} {"softIoU":>8} {"d(anchor)":>10} {"worstL":>7} '
          f'{"pt px":>6} {"p95":>6} {"jit":>5} {"keys":>6} {"F1":>6} {"<0.90":>6}')
    for r in rows:
        c = r['common']
        pub = r['as_published'].get('mean_soft_iou')
        delta = (c['mean_soft_iou'] - anchor_row['common']['mean_soft_iou']
                 if anchor_row else float('nan'))
        print(f'{r["run"]:<34} {pub:>7.4f} {c["mean_soft_iou"]:>8.4f} {delta:>+10.4f} '
              f'{c["worst_layer_soft_iou"]:>7.4f} {c["point_err_px"]:>6.2f} '
              f'{c["p95_point_err_px"]:>6.2f} {c["jitter_px"]:>5.2f} '
              f'{c["key_ratio"]:>5.2f}x {c["key_f1"]:>6.3f} {c["frames_below_0.90"]:>6}')
    print(f'\n`pub` is what the row\'s own report quoted, over its own layer set '
          f'({rows[0]["published_layers"]} layers for the v001 rows). `softIoU` is the same '
          f'run re-aggregated\non the common {len(common)}, which is the only column that '
          f'compares.\n-> {args.out}')


if __name__ == '__main__':
    main()
