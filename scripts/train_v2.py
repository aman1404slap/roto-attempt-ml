"""Train one named v2 configuration.

The v2 ladder, one rung per charter S4 stage. Every rung is two seeds at 40k steps, ~22 min a
seed on the current GPU, and each is named here rather than assembled at a shell prompt so a
result can be traced back to the exact configuration that produced it.

**S0** -- geometry and motion, structure given. v1.1's ``HEADLINE`` unchanged, because Step 2
exists to produce an *anchor* and an anchor measured with a term nobody has measured before
anchors nothing. Result: 0.9717 on-screen soft IoU, 0.830 px, 1.119x keys.

**S1** -- the key-timing head. Run, closed, does not pass: over-random key F1 0.0720 -> 0.0914
against a 0.15 bar, and the head scored alone is *below* its own best trivial baseline. Kept
here because the S2 rungs are read against it, and because its stop condition was written
before the run and honoured.

**S2** -- lifespans and point counts, in three rungs rather than one. Charter S2 allows one
interface change per re-baseline and S2 as written changes four things at once; splitting it
is what lets a failure say *which* change failed.

* ``s2a`` -- plan section 5's loss reweight **alone**, structure still given. At S2 identity is
  still given, so correspondence is exact and the dot term is still legitimate: the reweight is
  preparation for S3 and at S2 it can only cost geometry. Measured before it is mixed with two
  new heads.
* ``s2b`` -- the lifespan head. The rung the render gate is actually about.
* ``s2c`` -- the point-count head, with ``n_points`` out of the query input.

The S2 rungs' gate and their pre-written stop conditions are in ``v2-s2-design-note.md``; what
each was priced against before any of it was built is ``scripts/exp_s2_baselines.py``.

    python scripts/train_v2.py s2a --seed 1
    python scripts/train_v2.py s2b --seed 1 --steps 4000     # a probe, not a result
    python scripts/train_v2.py                               # s3a, the live rung

The rungs themselves are ``roto.v2.rungs``; this is the shell front end to them.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.v2.rungs import DEFAULT_RUN, RUNS                           # noqa: E402
from roto.v2.train import train                                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', choices=sorted(RUNS), nargs='?', default=DEFAULT_RUN)
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--runs-root', default='runs/v2',
                    help='where run directories are written. An API-triggered run passes a '
                         'scratch directory that is synced to S3 afterwards.')
    ap.add_argument('--out', default=None,
                    help='the run directory itself; default <runs-root>/<run>_seed<seed>')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=None,
                    help='override the 40k schedule; anything shorter is a probe, not a result')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cfg = replace(RUNS[args.run], seed=args.seed)
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    if args.device:
        cfg = replace(cfg, device=args.device)
    out = args.out or f'{args.runs_root}/{args.run}_seed{args.seed}'
    print(f'{args.run} seed {args.seed} -> {out}')
    train(args.dataset, out, cfg)


if __name__ == '__main__':
    main()
