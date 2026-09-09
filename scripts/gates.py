"""The v1.3 acceptance gates: what "as good as v1.x can be" means, checked rather than argued.

The plan's S4 states eleven thresholds and one rule about them -- *"Gates met -> declare
reconstruction on the sample drop done and stop polishing it. Gates missed -> the failing gate
names the next rung; nothing else gets added."* That rule only works if the gates are read
mechanically, so this script reads them mechanically and exits non-zero on a miss. There is
nothing to interpret at the end of a round if the interpretation was fixed at the start.

**The gate configuration is the hardest one in the plan, and deliberately.** S4 says "through
the constrained operating point and with predicted motion end to end (no teacher-forced
transform)", so the numbers gated are not the headline numbers: they are the headline
configuration scored with the key economy an artist can edit *and* with the artist's motion
track taken away. v1.2 priced both -- the constraint costs 0.0010 soft IoU and predicted
motion costs 0.0049 -- so the gate sits about 0.006 below the number a report would otherwise
lead with. That is the point: the gate describes a delivery, and a delivery has neither the
artist's motion track nor permission to place keys wherever rendered IoU likes them.

**Gates are read on the layers the run trained on.** A v002 run is scored on all 13 layers and
trained on 11; the two it never saw have freshly initialised query rows
(``reconstruct.untrained_queries``) and cannot reconstruct by construction, so a gate like
"every layer >= 0.90" is a statement about the 11. The other two are a separate measurement --
what the encoder alone gives -- and they are printed beside the gates rather than inside them,
because diluting a quality gate with a capacity measurement would make it unreadable in both
directions.

**Two seeds, meaned, per the plan's own rule.** A gate decided by one run is a gate decided by
the seed: v1.2 measured ±0.0022 soft IoU at 40k and ±0.34 on the worst single frame. Where only
one seed exists the gate is reported as such and does not pass.

    python scripts/gates.py                                   # the default candidate
    python scripts/gates.py --candidate v002_keytime --seeds v002_keytime v002_keytime_s1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PASS, FAIL, PENDING = 'PASS', 'FAIL', 'PENDING'

KEY_RATIO_RANGE = (0.75, 1.30)
"""Key economy an artist can still edit: from a quarter sparser to a third denser than theirs.

v1.1 chose the band and v1.2 measured what it costs -- 0.0010 soft IoU, a third of the 40k
noise floor -- against +0.026 key F1 and 0.76x the artist's keys instead of 0.58x. Rendered IoU
alone prefers 0.58x because it never asks *where* the keys are; the deliverable is a file
somebody edits."""


def gate(name: str, got, want: str, ok: bool | None, note: str = '') -> dict:
    return {'gate': name, 'measured': got, 'requirement': want,
            'status': PENDING if ok is None else (PASS if ok else FAIL), 'note': note}


def mean_of(runs: list[dict], key: str, default=None):
    vals = [r[key] for r in runs if key in r]
    return float(np.mean(vals)) if vals else default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='v1.3/results')
    ap.add_argument('--candidate', default='v002_final',
                    help='the shipping candidate; the gate rows are read on its seeds')
    ap.add_argument('--seeds', nargs='+', default=None,
                    help='score labels of the candidate seeds; default <candidate> and '
                         '<candidate>_s1')
    ap.add_argument('--suffix', default='_e2e_refit',
                    help='the gate configuration\'s score-file suffix: predicted motion '
                         'through the constrained operating point')
    ap.add_argument('--out', default='v1.3/results/gates.json')
    args = ap.parse_args()

    res = Path(args.results)
    labels = args.seeds or [args.candidate, f'{args.candidate}_s1']
    runs, missing = [], []
    for lab in labels:
        p = res / f'score_{lab}{args.suffix}.json'
        (runs.append(json.loads(p.read_text())) if p.exists() else missing.append(p.name))

    ceiling_path = res / 'scoring_ceiling.json'
    ceiling = json.loads(ceiling_path.read_text()) if ceiling_path.exists() else None
    seat_path = res / 'silhouette_check.json'
    seat = json.loads(seat_path.read_text()) if seat_path.exists() else None

    rows: list[dict] = []
    n = len(runs)
    two = n >= 2

    if not runs:
        rows.append(gate('the candidate is scored at the gate configuration', 'nothing found',
                         f'score_<seed>{args.suffix}.json for {args.candidate}', False,
                         'missing: ' + ', '.join(missing)))
    else:
        lo, hi = KEY_RATIO_RANGE
        # Which of mean and worst is right differs per gate, and the choice is made here
        # rather than left implicit. A **mean** across seeds is right for the aggregate
        # quality rows -- the plan says "2-seed mean". A **worst** is right for the two
        # per-layer floors: a layer that fails on either seed is a layer that fails, and
        # averaging two seeds of a floor invents a run nobody trained. So the per-layer rows
        # below are read across every layer of every seed at once, and the note says which
        # seed the failing layer came from.
        per = [(r['label'], x) for r in runs for x in r['per_layer']
               if x['layer_id'] not in r.get('withheld_layers', [])]
        worst_layer = min(per, key=lambda t: t[1]['mean_soft_iou'])
        worst_pct = max(per, key=lambda t: t[1]['frames_below_0.90'] / max(1, t[1]['frames']))
        wp = 100.0 * worst_pct[1]['frames_below_0.90'] / max(1, worst_pct[1]['frames'])

        siou = mean_of(runs, 'mean_soft_iou')
        rows += [
            gate('soft IoU, 2-seed mean', f'{siou:.4f}', '>= 0.97', two and siou >= 0.97,
                 f'{n} seed(s): ' + ', '.join(f'{r["mean_soft_iou"]:.4f}' for r in runs)
                 + ('' if two else ' -- one seed is not a 2-seed mean')),
            gate('every trained layer', f'{worst_layer[1]["mean_soft_iou"]:.4f}', '>= 0.90',
                 worst_layer[1]['mean_soft_iou'] >= 0.90,
                 f'worst is {worst_layer[1]["layer_id"][:34]} on {worst_layer[0]}'),
            gate('frames below 0.90, per layer', f'{wp:.2f}%', '<= 1% per layer', wp <= 1.0,
                 f'worst is {worst_pct[1]["layer_id"][:34]} on {worst_pct[0]} '
                 f'({worst_pct[1]["frames_below_0.90"]}/{worst_pct[1]["frames"]} frames)'),
            gate('point error, mean', f'{mean_of(runs, "point_err_px"):.2f} px',
                 '<= 1.2 px (256 crop)', mean_of(runs, 'point_err_px') <= 1.2),
            gate('point error, p95', f'{mean_of(runs, "p95_point_err_px"):.2f} px',
                 '<= 3 px (256 crop)', mean_of(runs, 'p95_point_err_px') <= 3.0,
                 'max is one control point and is reported, not gated: '
                 f'{max(r["max_point_err_px"] for r in runs):.0f} px'),
            gate('key economy', f'{mean_of(runs, "key_ratio"):.2f}x',
                 f'within [{lo:.2f}, {hi:.2f}]x artist',
                 lo <= mean_of(runs, 'key_ratio') <= hi),
            gate('key F1', f'{mean_of(runs, "key_f1"):.3f}', '>= 0.42',
                 mean_of(runs, 'key_f1') >= 0.42,
                 f'the keytime head\'s own target is >= 0.50. Gated on the established '
                 f'definition, which the plan\'s threshold was set against; strict '
                 f'(out-of-live-range artist keys excluded rather than clipped onto the live '
                 f'boundary, where a forced knot matches them for free) is '
                 f'{mean_of(runs, "key_f1_strict", 0.0):.3f}, and the same keys placed at '
                 f'random score {mean_of(runs, "baseline_pipeline_key_f1_random", 0.0):.3f}'),
        ]
        if any(r.get('head_key_f1') for r in runs):
            # The plan's parenthetical -- "key F1 >= 0.42 (keytime head's target: >= 0.50)" --
            # raises the bar on the *same* metric when the key-timing head is the candidate.
            # It is not a gate on the head's own thresholded key set: that number is not a
            # statement about the head at all unless its trivial baselines are beside it, and
            # they are, so it is reported as a diagnostic rather than gated.
            f1 = mean_of(runs, 'key_f1')
            hf, hb = mean_of(runs, 'head_key_f1'), mean_of(runs, 'head_key_f1_over_best_baseline', 0.0)
            rows.append(gate('key F1, the keytime head\'s raised target', f'{f1:.3f}',
                             '>= 0.50', f1 >= 0.50,
                             f'the same pipeline metric at a higher bar, because the candidate '
                             f'carries the head. The head\'s *own* key set scores {hf:.3f}, '
                             f'which is {hb:+.3f} against the better of firing on every live '
                             f'frame and firing at random -- so the head works as a bias on '
                             f'the keyframe search and not as a key detector'))
        if any('held_gap' in r for r in runs):
            g = mean_of(runs, 'held_gap')
            rows.append(gate('held-out-frame gap', f'{g:+.4f}', '<= 0.002', abs(g) <= 0.002,
                             'every 7th frame, withheld at build time'))
        else:
            rows.append(gate('held-out-frame gap', 'no split in this run', '<= 0.002', None,
                             'the run opted out of the dataset\'s split'))

    if ceiling is None:
        rows.append(gate('scoring ceiling', 'not measured', '= 1.000000', None,
                         'run scripts/exp_scoring_ceiling.py'))
    else:
        q = ceiling.get('worst_frame_ceiling_quantised')
        rows.append(gate('scoring ceiling', f'{q:.9f} on every frame' if q else 'no quantised '
                         'column', '= 1.000000 (post-rebuild)', q is not None and q >= 1.0,
                         f'through the dataset\'s own uint16 round trip; raw float '
                         f'{ceiling["mean_ceiling"]:.9f}, worst pixel '
                         f'{ceiling["max_abs_pixel"]:.2e} against a half-quantum of '
                         f'{ceiling["half_quantum"]:.2e}'))

    if seat is None:
        rows.append(gate('one .sfx opened, rendered and edited in Silhouette',
                         'not run -- needs a seat', 'no manual repair', None,
                         'files are written and ready in v1.3/sfx/; the checklist is '
                         'v1.3/sfx/README.md. Record the result in '
                         'v1.3/results/silhouette_check.json to close this row'))
    else:
        rows.append(gate('one .sfx opened, rendered and edited in Silhouette',
                         seat.get('summary', 'recorded'), 'no manual repair',
                         bool(seat.get('passed')), seat.get('note', '')))

    payload = {'candidate': args.candidate, 'configuration': args.suffix,
               'seeds': labels, 'seeds_found': n, 'rows': rows,
               'passed': sum(r['status'] == PASS for r in rows),
               'failed': sum(r['status'] == FAIL for r in rows),
               'pending': sum(r['status'] == PENDING for r in rows)}
    if runs and runs[0].get('held_layers'):
        h = runs[0]['held_layers']
        payload['held_layers'] = {
            'layers': runs[0]['withheld_layers'], 'mean_soft_iou': h['mean_soft_iou'],
            'point_err_px': h['point_err_px'],
            'note': 'never trained on; freshly initialised query rows. Not a gate -- this is '
                    'the encoder with the memorisation capacity set to zero, which is the '
                    'number v2 dynamic queries have to beat.'}

    Path(args.out).write_text(json.dumps(payload, indent=2))

    width = max(len(r['gate']) for r in rows)
    print(f'\n=== v1.3 acceptance gates — {args.candidate} at {args.suffix or "default"} '
          f'({n} seed(s)) ===')
    for r in rows:
        mark = {PASS: 'PASS', FAIL: 'FAIL', PENDING: '....'}[r['status']]
        print(f'[{mark}] {r["gate"]:<{width}}  {str(r["measured"]):>22}  '
              f'(want {r["requirement"]})')
        if r['note']:
            print(f'{"":>7} {"":<{width}}  {r["note"]}')
    if 'held_layers' in payload:
        h = payload['held_layers']
        print(f'\n  beside the gates, not inside them: the {len(h["layers"])} layers this run '
              f'never trained on\n  score {h["mean_soft_iou"]:.4f} soft IoU at '
              f'{h["point_err_px"]:.1f} px with untrained query rows.')
    print(f'\n{payload["passed"]} passed, {payload["failed"]} failed, '
          f'{payload["pending"]} pending -> {args.out}')
    sys.exit(1 if payload['failed'] else 0)


if __name__ == '__main__':
    main()
