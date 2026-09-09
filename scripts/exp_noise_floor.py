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


D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534}
"""Expected range of ``n`` draws from a unit normal -- the control-chart ``d2`` constant.

Needed because this round quotes **two**-seed ranges and v1.2 quoted **three**-seed ranges, and
a range grows with the number of draws whatever the underlying spread does. Comparing 2-seed
±0.0008 against 3-seed ±0.0022 directly would credit the smaller sample for being smaller:
the honest comparison divides each by its own ``d2``, which turns both into an estimate of the
same quantity. It is a crude estimator at these sample sizes and it is the right *kind* of
number, which is why it is reported beside the raw range rather than instead of it.
"""


def sd_from_range(rng: float, n: int) -> float | None:
    """Estimate the per-run sd from an observed range of ``n`` runs. See :data:`D2`."""
    d = D2.get(int(n))
    return rng / d if d else None


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
    'control, 40k': ['v002_control', 'v002_control_s1'],
    'headline, 40k': ['v002_final', 'v002_final_s1'],
    'keytime, 40k': ['v002_keytime', 'v002_keytime_s1'],
}
"""The v1.3 groups: two seeds each, at the only schedule this round quotes.

Two rather than three, and 40k rather than both schedules, because v1.2 already answered the
question three seeds were for -- the spread is **±0.0069 at 12k against ±0.0022 at 40k**, so the
converged schedule is three times quieter and the 12k ladder is not where anything is decided
any more. What two seeds buy is a *range*, which is the honest statistic for two runs and is
what the plan's "two seeds per quoted number" asks for; the sd and the sizing table below need
three and are reported as unavailable rather than computed from two.

Each configuration gets its own group because a spread is a property of a configuration, not of
the project: v1.2 measured the headline as *quieter* than the control (±0.0015 against ±0.0022)
and its worst frame as far quieter (±0.019 against ±0.34). A key head is new machinery on the
same geometry, so it gets its own row rather than borrowing one.

The v1.2 groups, for reference, were `12k`/`40k` on v1's configuration and `40k headline`, all
on `datasets/v001`. `--groups` re-points this at any set of scored runs."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='v1.3/results')
    ap.add_argument('--out', default='v1.3/results/noise_floor.json')
    ap.add_argument('--groups', action='append', default=[], metavar='LABEL=RUN,RUN',
                    help='override the groups; repeatable, e.g. '
                         '--groups "40k=v1_control_long,control_long_s1,control_long_s2"')
    args = ap.parse_args()
    groups = GROUPS
    if args.groups:
        groups = {}
        for spec in args.groups:
            label, _, names = spec.partition('=')
            groups[label] = [n for n in names.split(',') if n]
    res = Path(args.results)

    out = {}
    for label, names in groups.items():
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
        # Range corrected for how many runs produced it, so a two-seed row and a three-seed
        # row are comparable. See D2.
        for m in METRICS:
            if m in s:
                s[m]['sd_from_range'] = sd_from_range(s[m]['range'], len(runs))
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
        # An sd from two runs is a number, not an estimate: it has one degree of freedom and
        # the sizing table below is quadratic in it, so two seeds would produce a runs-per-arm
        # figure with an enormous confidence interval and no warning on it. Reported only
        # where three or more runs exist; v1.2's three-seed sd of 0.0011 at 40k stands as the
        # figure to size against until another three-seed group is run.
        if len(runs) >= 3:
            sd = float(np.std([r['mean_soft_iou'] for r in runs], ddof=1))
            s['soft_iou_sd'] = sd
            s['seeds_needed'] = {f'{d:.4f}': seeds_needed(sd, d) for d in EFFECTS}
        else:
            # Two seeds give a range. Turned into an sd estimate through the control-chart
            # constant so the sizing table can still be printed -- flagged as an estimate,
            # because at n=2 it has one degree of freedom and the table is quadratic in it.
            sd = sd_from_range(s['mean_soft_iou']['range'], len(runs))
            s['soft_iou_sd'] = sd
            s['soft_iou_sd_is_estimated_from_range'] = True
            s['seeds_needed'] = ({f'{d:.4f}': seeds_needed(sd, d) for d in EFFECTS}
                                 if sd else None)
            s['sizing_note'] = (f'sd estimated from a {len(runs)}-seed range via d2; a third '
                                'seed would tighten it')
        out[label] = s

        print(f'\n=== {label}, {len(runs)} seeds {seeds} ===')
        for m in METRICS:
            v = s[m]
            est = v.get('sd_from_range')
            print(f'  {m:<24} {v["mean"]:>9.4f}  range {v["range"]:>7.4f}  '
                  f'[{v["min"]:.4f}, {v["max"]:.4f}]'
                  + (f'   sd~{est:.4f}' if est is not None else ''))
        worst = max(per.items(), key=lambda kv: kv[1]['range'])
        print(f'  least stable layer: {worst[0][:44]} range {worst[1]["range"]:.4f}')
        if sd and s.get('seeds_needed'):
            est = ' (estimated)' if s.get('soft_iou_sd_is_estimated_from_range') else ''
            print(f'  runs per arm to resolve a soft-IoU difference (sd {sd:.4f}{est}):')
            print('    ' + '   '.join(f'{d:+.4f} -> {n}'
                                      for d, n in zip(EFFECTS, s['seeds_needed'].values())))
        if 'sizing_note' in s:
            print(f'  {s["sizing_note"]}')

    Path(args.out).write_text(json.dumps(out, indent=2))
    if out:
        print('\nthe number every other table has to be read against:')
        for label, s in out.items():
            est = s['mean_soft_iou'].get('sd_from_range')
            print(f'  {label}: soft IoU varies by {s["mean_soft_iou"]["range"]:.4f} '
                  f'across {s["n_runs"]} seeds of one configuration'
                  + (f' (sd~{est:.4f})' if est is not None else ''))
        print('  for reference, v1.2 measured ±0.0069 over 3 seeds at 12k and ±0.0022 over 3 '
              'at 40k\n  on datasets/v001 -- sd~0.0041 and sd~0.0013 by the same correction.')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
