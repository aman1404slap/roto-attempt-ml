"""Train one named v1.1 configuration.

The configurations are a **ladder**: each rung turns on exactly one thing the review asked
for, so the effect of each change is attributable rather than inferred from a single combined
run. ``v1_control`` is not v1's checkpoint -- it is v1's *configuration* retrained under
v1.1's contiguous frame sampling, which is the only way to read the ladder's first rung as
one variable rather than two.

    python scripts/train_v11.py v1_control
    python scripts/train_v11.py final_long --steps 40000
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

LADDER: dict[str, TrainConfig] = {
    # v1's configuration, retrained under v1.1's contiguous sampling.
    'v1_control': replace(BASE),
    # 1a: three consecutive alphas as channels, so the net can denoise time itself.
    # NOTE: this rung and the next are the *misaligned* pair. Each neighbour is stacked at its
    # own crop offset, and the window twitches 1.09 crop px per step -- more than the 0.77 px
    # of jitter the window exists to remove. Kept so the ladder stays reproducible, but the
    # `_aligned` rows below are the ones that actually test the idea.
    'window': replace(BASE, in_frames=3),
    # 1b: penalise |d pred - d target| between adjacent frames. Weight 1.0 means a pixel of
    # jitter costs exactly what a pixel of position error costs -- both terms are in pixels.
    # This is also where sampling switches to pairs: the term needs adjacent frames in the
    # batch, and 'runs' -- the obvious way to get them -- costs 1.7 px of point error.
    'window_temporal': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs'),
    # 3: L1 between the drawn polylines, at half the point term's weight.
    'window_temporal_curve': replace(BASE, in_frames=3, temporal_weight=1.0,
                                     sampling='pairs', curve_weight=0.5),
    # 9: queries negotiate with each other. The two worst v1 layers are the shape-dense ones.
    'full': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                    curve_weight=0.5, self_attn=True),
    # 12: does sqrt(frames * shapes) beat weighting by frame count alone?
    'full_sqrt_weight': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                                curve_weight=0.5, self_attn=True, sample_weight='sqrt'),
    # v1's configuration on the *long* schedule. Without this row, 'final_long' cannot be
    # read: at 12k the v1.1 configuration scored below v1's, and at 40k it scores far above
    # it, so the gain could be the configuration or simply the schedule. This is the control
    # that separates them, and it is the only honest way to attribute the headline.
    'v1_control_long': replace(BASE, steps=40000),
    # What 'runs' sampling costs, held at v1's configuration so sampling is the only variable.
    'v1_control_runsampling': replace(BASE, sampling='runs'),
    # The headline run: the best rung of the ladder, on a longer schedule. It carries
    # sample_weight='sqrt' because that was the largest single gain measured anywhere in the
    # ladder (+0.0088 soft IoU over frame-count weighting, self-attention held fixed). The
    # review filed it under "small items -- measure, don't assume"; measuring said yes.
    'final_long': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                          curve_weight=0.5, self_attn=True, sample_weight='sqrt',
                          steps=40000),
    # 4: the same, with every 7th frame withheld from training and scored separately.
    'final_holdout': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                             curve_weight=0.5, self_attn=True, sample_weight='sqrt',
                             steps=40000, holdout_every=7),

    # --- v1.1 review, item 1: the window, this time actually aligned -------------------
    # `window` and `window_temporal` above were measured with the three channels displaced
    # from each other by the crop window's own twitch. These two repeat them with each
    # neighbour warped into the anchor's window, which is the first honest test of S1a.
    'window_aligned': replace(BASE, in_frames=3, align_window=True),
    'window_temporal_aligned': replace(BASE, in_frames=3, align_window=True,
                                       temporal_weight=1.0, sampling='pairs'),

    # --- v1.1 review, item 2: the transform head, reframed ----------------------------
    # Both rungs hold `full_sqrt_weight`'s geometry configuration fixed and change only the
    # transform target, so they read against `full_sqrt_weight_affine` (0.6059) directly.
    # `affine_weight` drops 20 -> 1 because the crop-space term is already in crop pixels;
    # at 20 it would price transform error twenty times above point error.
    #
    # Two rungs rather than one, because the review raises two suspicions and they are
    # separable: `affine_crop` changes the space alone, `affine_crop_aligned` also gives the
    # head the aligned window -- motion being the one quantity no single frame can show.
    'affine_crop': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                           curve_weight=0.5, self_attn=True, sample_weight='sqrt',
                           affine_space='crop', affine_weight=1.0),
    'affine_crop_aligned': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                                   curve_weight=0.5, self_attn=True, sample_weight='sqrt',
                                   affine_space='crop', affine_weight=1.0,
                                   align_window=True),

    # What the transform term costs the geometry, and whether it is avoidable. At weight 1.0
    # the crop-space term ends at 2.16 px against the point term's 3.31 -- roughly 30% of the
    # total loss, where the old document-space term at weight 20 was ~7% -- and the two heads
    # share an encoder, so it perturbs geometry. Measured at 0.9102 against
    # `full_sqrt_weight`'s 0.9197. This rung prices the dial rather than arguing about it.
    'affine_crop_light': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                                 curve_weight=0.5, self_attn=True, sample_weight='sqrt',
                                 affine_space='crop', affine_weight=0.25,
                                 align_window=True),

    # The headline run rebuilt on both fixes, for the 40k schedule that is the only one at
    # which anything here has converged.
    #
    # `affine_weight=0.25` is measured, not assumed, and this is the one place the two heads
    # have to be traded off against each other. At 12k, against `full_sqrt_weight`'s 0.9197
    # geometry / 0.6059 transform:
    #
    #     weight 1.0   ->  0.9061 geometry (-0.0136),  0.8830 transform (+0.2771)
    #     weight 0.25  ->  0.9151 geometry (-0.0046),  0.8484 transform (+0.2425)
    #
    # 0.25 buys 88% of the transform gain for 34% of the geometry cost, and geometry is the
    # headline while the transform head is still not consumed. See `affine_crop_light`.
    'final_long_v2': replace(BASE, in_frames=3, temporal_weight=1.0, sampling='pairs',
                             curve_weight=0.5, self_attn=True, sample_weight='sqrt',
                             affine_space='crop', affine_weight=0.25, align_window=True,
                             steps=40000),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name', choices=sorted(LADDER))
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/runs')
    ap.add_argument('--steps', type=int, default=None)
    args = ap.parse_args()

    cfg = LADDER[args.name]
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    out = Path(args.out) / args.name
    print(f'=== {args.name} ===\n{json.dumps(asdict(cfg), indent=2)}')
    train(args.dataset, out, cfg)


if __name__ == '__main__':
    main()
