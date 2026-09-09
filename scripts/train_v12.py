"""Train one named v1.2 configuration.

v1.2 answers the handover in `v1.1-to-v1.2-plan.md`, whose method is to stop reading single
runs as verdicts. Two kinds of rung:

**Seed repeats.** Every ladder in v1 and v1.1 is one seed per configuration, so a +0.003 row
and a lucky row are indistinguishable -- the handover's S2 calls that out and asks for the
spread. `control_s*` repeats v1's configuration at 12k and `control_long_s*` at 40k, because
the noise floor is a property of the *schedule* as much as the model: a run that has not
converged is noisier than one that has, and the ladder is read at both lengths.

**The decision tree's transform branch.** `affine_deep` gives the transform head a decoder of
its own depth. See `TrainConfig.affine_depth` for why that is a capacity-versus-schedule
question rather than the fix the handover assumed.

    python scripts/train_v12.py control_s1
    python scripts/train_v12.py affine_deep
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
"""v1's configuration, exactly as v1.1's ladder states it. `train_v11.LADDER['v1_control']`."""

V11_FULL = replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                   curve_weight=0.5, self_attn=True, sample_weight='sqrt')
"""`full_sqrt_weight`: the geometry configuration every v1.1 transform rung holds fixed."""

LADDER: dict[str, TrainConfig] = {
    # --- the run-to-run noise floor (plan S2, last row of the ledger) ----------------
    # v1's configuration, unchanged, at two more seeds. `seed` drives both the weight init
    # and the layer/frame draw, so a repeat varies everything a rerun would vary.
    'control_s1': replace(BASE, seed=1),
    'control_s2': replace(BASE, seed=2),
    # And the same at the converged schedule. Both are needed because the 12k ladder and the
    # 40k rows are read against each other throughout v1.1, and there is no reason a run
    # stopped mid-descent should have the same spread as one that has flattened.
    'control_long_s1': replace(BASE, steps=40000, seed=1),
    'control_long_s2': replace(BASE, steps=40000, seed=2),
    # The headline configuration's own spread. The claim it has to support is the one that
    # cost something -- `final_long_v2` gives up 0.0029 of geometry for +0.163 of transform --
    # and 0.0029 is exactly the size of difference a single seed cannot speak to.
    'final_v2_s1': replace(V11_FULL, affine_space='crop', affine_weight=0.25,
                           align_window=True, steps=40000, seed=1),

    # --- the decision tree's transform branch (plan S4) -----------------------------
    # "Doesn't -> give the head its own small decoder depth before giving up." It already has
    # its own decoder; this doubles its depth, holding `affine_crop_aligned` fixed otherwise
    # so the row reads against that rung's 0.8830 transform / 0.9061 geometry directly.
    'affine_deep': replace(V11_FULL, affine_space='crop', affine_weight=1.0,
                           align_window=True, affine_depth=6),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name', choices=sorted(LADDER))
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.2/runs')
    ap.add_argument('--steps', type=int, default=None)
    args = ap.parse_args()

    cfg = LADDER[args.name]
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    out = Path(args.out) / args.name
    print(f'=== {args.name} ===\n{json.dumps(asdict(cfg), indent=2)}', flush=True)
    train(args.dataset, out, cfg)


if __name__ == '__main__':
    main()
