"""What a metric does when nothing changes but the seed.

Every ladder in v1 and v1.1 is one run per configuration, and both reports guess at the noise
floor -- "differences under about 0.003 soft IoU are noise" -- from nothing. The handover asks
for the measurement instead, and it is the row that decides how the rest of the tables may be
read: a ladder whose rungs differ by less than its own spread is a ladder of coin flips.

Two schedules, because there is no reason a run stopped mid-descent has the same spread as one
that has flattened, and the whole v1.1 ladder is read at 12k while its headline is read at 40k.

Reports the spread of the aggregate *and* of the worst-case columns, since those are what v1.2
quotes. Per layer too: a stable mean can hide a layer that swings, and if it does, the layer is
where the next experiment should look.

    python scripts/exp_noise_floor.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.report import spread                                    # noqa: E402

METRICS = ['mean_soft_iou', 'worst_layer_soft_iou', 'worst_frame_soft_iou',
           'point_err_px', 'p95_point_err_px', 'jitter_px', 'key_ratio', 'key_f1']

EFFECTS = [0.0008, 0.0032, 0.005, 0.010, 0.020, 0.050]
"""Differences in soft IoU the ladder has actually claimed, from the aligned window's +0.0008
to the two hard layers' +0.05, priced in runs below."""


def seeds_needed(sd: float, delta: float) -> int:
    """Runs per arm to resolve a difference of ``delta`` given a per-run sd of ``sd``.

    The standard two-sample sizing, ``n = 2 (z_(a/2) + z_b)^2 s^2 / d^2`` at 5% and 80%, which
    is 15.7 s^2/d^2. It is the number that says what a one-run-per-rung ladder can and cannot
    resolve, and it is why v1.2 proposes no new headline: at this spread, most of the
    differences v1.1's 12k ladder reports are not measurable with one run each, in either
    direction. Reported as an order of magnitude, not as a protocol -- with three seeds the sd
    itself is loose."""
    return int(np.ceil(15.7 * (sd / delta) ** 2))

GROUPS = {
    '12k': ['v1_control', 'control_s1', 'control_s2'],
    '40k': ['v1_control_long', 'control_long_s1', 'control_long_s2'],
    '40k headline': ['final_long_v2', 'final_v2_s1'],
}
"""v1's own configuration at three seeds, at both schedules the ladder is read at. `v1_control`
and `v1_control_long` are seed 0 -- v1.1's own rungs, re-scored under v1.2's columns, so the
floor is measured on the same runs the ladder quotes rather than on a fresh set.

The third group is the *headline* configuration rather than the control, at two seeds. It answers
a narrower question and the handover does not ask for it: v1.1 reports that `final_long_v2` gives
up 0.0029 of geometry to make the transform head work, and 0.0029 is exactly the size of claim a
single seed cannot support. Two runs give a range and no sd, which is the honest form of it."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='v1.2/results')
    ap.add_argument('--out', default='v1.2/results/noise_floor.json')
    args = ap.parse_args()
    res = Path(args.results)

    out = {}
    for label, names in GROUPS.items():
        runs, seeds = [], []
        for n in names:
            p = res / f'score_{n}.json'
            if p.exists():
                d = json.loads(p.read_text())
                runs.append(d)
                seeds.append(d.get('seed'))
        if len(runs) < 2:
            print(f'{label}: {len(runs)} run(s) scored -- skipping')
            continue
        s = spread(runs, METRICS)
        s['runs'] = names[:len(runs)]
        s['seeds'] = seeds
        # Per layer, on the headline metric only: which layer is unstable, if any.
        per = {}
        ids = [r['layer_id'] for r in runs[0]['per_layer']]
        for lid in ids:
            vals = [next(x['mean_soft_iou'] for x in r['per_layer'] if x['layer_id'] == lid)
                    for r in runs]
            per[lid] = {'range': float(np.max(vals) - np.min(vals)),
                        'values': [float(v) for v in vals]}
        s['per_layer_soft_iou'] = per
        # What this spread means for a ladder read one run per rung.
        sd = float(np.std([r['mean_soft_iou'] for r in runs], ddof=1))
        s['soft_iou_sd'] = sd
        s['seeds_needed'] = {f'{d:.4f}': seeds_needed(sd, d) for d in EFFECTS}
        out[label] = s

        print(f'\n=== {label}, {len(runs)} seeds {seeds} ===')
        for m in METRICS:
            v = s[m]
            print(f'  {m:<24} {v["mean"]:>9.4f}  range {v["range"]:>7.4f}  '
                  f'[{v["min"]:.4f}, {v["max"]:.4f}]')
        worst = max(per.items(), key=lambda kv: kv[1]['range'])
        print(f'  least stable layer: {worst[0][:44]} range {worst[1]["range"]:.4f}')
        print(f'  runs per arm to resolve a soft-IoU difference (sd {sd:.4f}):')
        print('    ' + '   '.join(f'{d:+.4f} -> {n}'
                                  for d, n in zip(EFFECTS, s['seeds_needed'].values())))

    Path(args.out).write_text(json.dumps(out, indent=2))
    if out:
        print('\nthe number every other table has to be read against:')
        for label, s in out.items():
            print(f'  {label}: soft IoU varies by {s["mean_soft_iou"]["range"]:.4f} '
                  f'across {s["n_runs"]} seeds of one configuration')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
