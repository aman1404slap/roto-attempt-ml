"""Score one or more v2 seeds into the frozen table.

Scoring was ad-hoc through Step 2 -- a ``python -c`` in the write-up -- which was fine while
there was one configuration and two seeds. S2 has three rungs and a decode with three knobs,
so the invocation becomes something a result has to be traceable to.

Charter L4 is enforced rather than documented here: a single seed prints a header saying it is
not a mean, and the spread column is on every row so "is this difference weather" is checkable
from the table itself.

    python scripts/score_v2.py s1 s2a                       # both, 2 seeds each
    python scripts/score_v2.py s2b --lifespan predicted --alive-on 0.5 --alive-off 0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.v2.reconstruct import ARTIST, PREDICTED, RebuildConfig      # noqa: E402
from roto.v2.score import frozen_table, score_run                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run names under runs/v2, e.g. s0 s1 s2a')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--lifespan', choices=(ARTIST, PREDICTED), default=ARTIST)
    ap.add_argument('--point-count', choices=(ARTIST, PREDICTED), default=ARTIST)
    ap.add_argument('--alive-on', type=float, default=0.5)
    ap.add_argument('--alive-off', type=float, default=0.2)
    ap.add_argument('--alive-min-gap', type=int, default=0)
    args = ap.parse_args()

    cfg = RebuildConfig(lifespan=args.lifespan, point_count=args.point_count,
                        alive_on=args.alive_on, alive_off=args.alive_off,
                        alive_min_gap=args.alive_min_gap)
    for name in args.runs:
        runs = []
        for s in args.seeds:
            ck = Path('runs/v2') / f'{name}_seed{s}' / 'model.pt'
            if not ck.exists():
                print(f'  (no {ck}, skipping seed {s})')
                continue
            runs.append(dict(score_run(ck, args.dataset, cfg, seed=s), seed=s))
        if not runs:
            print(f'{name}: nothing to score')
            continue
        print()
        print(frozen_table(runs, label=f'v2 {name.upper()}'))
        out = Path('runs/v2') / f'{name}_summary.json'
        out.write_text(json.dumps(runs, indent=2, default=float))
        (Path('runs/v2') / f'{name}_table.txt').write_text(
            frozen_table(runs, label=f'v2 {name.upper()}'))
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
