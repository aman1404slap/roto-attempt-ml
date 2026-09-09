"""Train one named v1.3 configuration.

v1.3 is the consolidation round: the ground everything stood on was rebuilt once
(``datasets/v002``), so every past number has to be re-anchored once, and there is exactly one
new modelling idea to test. The ladder is short on purpose -- the plan's own rule is two seeds
per quoted number at the 40k schedule, and v1.2 measured why: at 12k the seed spread is
±0.0069 soft IoU, which is larger than most differences v1.1's ladder reported.

**Every v002 run inherits the dataset's split, and that is the point.** ``datasets/v002``
records a frame holdout (every 7th) and a layer holdout (two layers) at build time, and
``train()`` reads both. So a v002 run trains on **1306 frames of 11 layers and 2276 shapes**
rather than 1810 frames of 13 layers and 2753 shapes, and carries a held-frame gap natively
instead of needing a separate ``final_holdout`` rung. It also means **v002 rows are not
comparable to v1.1's row for row** -- 28% fewer frames and 17% fewer shapes. That is what
"re-baseline once" means, and ``v002_control`` is the new anchor. ``v002_final_all`` exists to
price exactly that gap.

Four groups:

**The re-baseline.** ``v002_control`` is v1's own configuration at the converged schedule --
the row every past number is re-read against. ``v002_final`` is the shipping candidate.

**The comparability row.** ``v002_final_all`` is the headline configuration with the split
switched off: all 13 layers, all 1810 frames. The only v002 row that reads against
``final_long_v2`` directly, and the difference between it and ``v002_final`` is the price of
the split rather than a guess at it.

**The one modelling change.** ``v002_keytime`` adds the key-timing head. Key F1 has sat at
0.36-0.41 for three rounds while geometry went 0.92 to 0.97, and v1.2 pinned the diagnosis:
the constrained operating point moved key F1 to 0.408 from the keyframe stage alone, with no
retraining and no new signal, while geometry sat still. So the plateau is a timing plateau,
and no component has ever been shown *when* artists key.

``v002_keytime_probe`` is a 12k sanity rung and it paid for itself in fourteen minutes: the
head's loss fell, its precision and recall left zero, and **it had learned nothing** -- against
the trivial baseline of firing on every live frame it scored +0.000. With one global positive
weight the cheapest thing to learn is each layer's key density, which ranges 13x across these
layers. ``v002_keytime_probe2`` is the same rung with the positive weight balanced *within*
each layer, which removes that exploit; ``v002_keytime`` is the 40k version of whichever
survives.

**The backfill.** ``final_long_s1`` is a second seed of v1.1's headline on ``datasets/v001``,
which settles the cheapest open question v1.2 left: whether the 0.0025 of geometry
``final_long_v2`` gives up for a working transform head is real or one spread wide.

    python scripts/train_v13.py v002_control
    python scripts/train_v13.py v002_keytime --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.train import TrainConfig, train                      # noqa: E402

BASE = TrainConfig(steps=12000, batch=6, log_every=500)
"""v1's configuration, exactly as v1.1's ladder states it (``train_v11.BASE``)."""

V11_GEOMETRY = dict(in_frames=3, temporal_weight=1.0, sampling='pairs', curve_weight=0.5,
                    self_attn=True, sample_weight='sqrt')
"""``full_sqrt_weight``: the geometry configuration every v1.1 transform rung holds fixed."""

HEADLINE = replace(BASE, **V11_GEOMETRY, affine_space='crop', affine_weight=0.25,
                   align_window=True, steps=40000)
"""v1.1's ``final_long_v2``, unchanged. The configuration v1.2 recommended building on: it
gives up 0.0025 of geometry (one spread wide, hence ``final_long_s1``) for a transform head
that works -- 0.9563 rendered against ``final_long``'s 0.7931, reproducible to 0.0012."""

KEY_WEIGHT = 1.0
"""Weight on the key-timing term.

Not in pixels and not derivable, unlike every other weight here, so it is argued rather than
converted. At convergence the point term lands near 1 crop px and a class-balanced BCE at this
archive's measured 9.6% positive rate lands in the same order, so 1.0 makes the two terms comparable
without the geometry giving anything up. ``v002_keytime_probe`` is the cheap check that this
is roughly right before 90 minutes go into two seeds of it; the *bias* that decides how much
the head actually changes -- ``RebuildConfig.key_bias`` -- is a scoring-time knob and is swept
for free afterwards.
"""

LADDER: dict[str, TrainConfig] = {
    # --- the re-baseline ----------------------------------------------------------
    # v1's own configuration at the converged schedule, on the rebuilt dataset. Everything
    # v1, v1.1 and v1.2 published is re-read against this row and nothing else.
    'v002_control': replace(BASE, steps=40000),
    # The shipping candidate: v1.1's headline configuration on the rebuilt dataset.
    'v002_final': replace(HEADLINE),
    # The same with the dataset's split switched off -- all 13 layers, all 1810 frames. The
    # one v002 row that reads against v1.1's `final_long_v2` directly, so the re-baseline can
    # be attributed to the data rather than to the split. `use_split=False` is a flag on the
    # run, not a second dataset: one set of alphas, one recorded split, one opt-out.
    'v002_final_all': replace(HEADLINE, use_split=False),

    # --- the key-timing head ------------------------------------------------------
    # 12k, one seed: not a quality rung (12k is below the noise floor) but a sanity rung, and
    # it earned its 14 minutes immediately. Its BCE fell and its precision and recall left
    # zero -- and it had still learned nothing: scored against the trivial baseline of firing
    # on every live frame it came out **+0.000** on `Layer_52` (0.803 against 0.797) and
    # -0.094 on `FAM blue`, where it stopped firing at all. With one global positive weight
    # the cheapest thing to learn is each layer's key *density*, which ranges 13x across these
    # layers and needs no timing information whatever. `key_balance='global'` is set explicitly
    # so this rung stays reproducible as the negative result it is.
    'v002_keytime_probe': replace(HEADLINE, steps=12000, key_weight=KEY_WEIGHT,
                                  key_balance='global'),
    # The same 14 minutes with the exploit removed: positives and negatives balanced *within*
    # each layer, so matching a base rate gains nothing and the only way down is `when`.
    'v002_keytime_probe2': replace(HEADLINE, steps=12000, key_weight=KEY_WEIGHT,
                                   key_balance='per_layer'),
    'v002_keytime': replace(HEADLINE, key_weight=KEY_WEIGHT, key_balance='per_layer'),

    # --- the backfill, on datasets/v001 ------------------------------------------
    # v1.1's `final_long` at seed 1. v1.2 left this as the cheapest open question in the
    # project: `final_long` 0.9707 against `final_long_v2`'s two-seed mean of 0.9682 is
    # 0.0025, about one spread wide, so the geometry v1.1 reports trading for a working
    # transform head may well be free. Two seeds per arm is enough at 40k (sd 0.0011).
    'final_long_s1': replace(BASE, **V11_GEOMETRY, steps=40000, seed=1),
}

V001_RUNS = ('final_long_s1',)
"""Rungs that belong on the *old* dataset, because their whole purpose is to complete a v1.1
comparison. Pointing them at v002 would answer a different question."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', choices=sorted(LADDER))
    ap.add_argument('--dataset', default=None,
                    help='default: datasets/v002, or datasets/v001 for the backfill rungs')
    ap.add_argument('--out', default='v1.3/runs')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None,
                    help='overrides the rung\'s seed; the run directory gets an _s<N> suffix')
    args = ap.parse_args()

    cfg = LADDER[args.name]
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    name = args.name
    if args.seed is not None and args.seed != cfg.seed:
        cfg = replace(cfg, seed=args.seed)
        name = f'{args.name}_s{args.seed}'
    dataset = args.dataset or ('datasets/v001' if args.name in V001_RUNS else 'datasets/v002')
    out = Path(args.out) / name
    print(f'=== {name}  ({dataset}) ===\n{json.dumps(asdict(cfg), indent=2)}', flush=True)
    train(dataset, out, cfg)


if __name__ == '__main__':
    main()
