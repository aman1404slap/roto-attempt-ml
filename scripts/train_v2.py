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
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.v2.train import TrainConfig, train                          # noqa: E402


S0 = TrainConfig(steps=40000, batch=6, log_every=500)
"""v1.1's ``final_long_v2`` / v1.3's ``HEADLINE``, which is what ``TrainConfig`` defaults to.

Stated as a named constant anyway, so the ladder below reads as a chain of deltas from one
anchor rather than as five independent configurations."""

S1 = replace(S0, key_weight=1.0, key_balance='per_element')
"""S0 plus the key-timing head. ``key_pos_weight=0`` means *measure it from the dataset*, which
on v003 gives 2.14 -- not plan section 4's assumed 13, which came from a 7% key rate this
dataset does not have (ours is 31.8%)."""

S2_LOSS = dict(point_weight=0.25, curve_weight=1.0)
"""Plan section 5's reweight: the curve term becomes primary and dot-L1 falls to 0.25x."""

S2A = replace(S1, **S2_LOSS)
"""The reweight alone, on top of S1. Answers one question and no others: what does plan
section 5's loss shift cost geometry, before any structural head exists?

On top of S1 rather than S0 because S1 is what charter S4's ``render within 0.01 of S1``
compares to, and because S1's key head cost nothing measurable (0.9717 -> 0.9712, inside the
0.0022 seed spread) -- so carrying it keeps the comparison one change wide."""

S2B = replace(S2A, alive_weight=1.0, alive_balance='per_element')
"""S2A plus the lifespan head. ``alive_pos_weight=0`` measures the positive weight from the
data, as the key head does; v003's live rate is 0.627, so the global value is 0.60 and the
per-element ones span the 0.093-1.000 live-rate range."""

S2C = replace(S2B, count_weight=1.0, count_style_weight=0.25, give_point_count=False)
"""S2B plus the point-count head, and ``n_points`` removed from the query input.

The removal is the training wheel coming off; it does **not** make the count unmemorisable,
because the per ``(element, shape)`` query row can carry it and the loss will put it there.
That is why the design note reports this head's accuracy and gates nothing on it."""

S3A = replace(S2C, n_slots=256)
"""S2C with the per-element query table replaced by **one bank of 256 slots shared across
every element**. Nothing else changes.

This is the rung that matters, and the only one worth running until it shows something. The S2
checkpoints reconstruct at 0.9652 with their query rows and **0.0377** with those rows freshly
initialised, so the table is carrying ~96% of the result and S3's whole job is to replace it.
The question here is narrow: does deriving the queries from the encoder move that 0.0377 at
all? The design note's stop condition says report it and stop if it does not -- rungs b and c
would be refinements on a mechanism that does not exist.

Slower per step than S2 by design: 256 queries on every element against S2's 1-247, so roughly
12 steps/s against 28."""

RUNS: dict[str, TrainConfig] = {'s0': S0, 's1': S1, 's2a': S2A, 's2b': S2B, 's2c': S2C,
                                's3a': S3A}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', choices=sorted(RUNS))
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--out', default=None, help='default runs/v2/<run>_seed<seed>')
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
    out = args.out or f'runs/v2/{args.run}_seed{args.seed}'
    print(f'{args.run} seed {args.seed} -> {out}')
    train(args.dataset, out, cfg)


if __name__ == '__main__':
    main()
