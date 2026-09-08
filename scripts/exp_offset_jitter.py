"""How much does the tracking crop twitch, and would smoothing it be worth a rebuild?

The crop window's per-frame offset is derived from a 0.25-scale survey render and rounded to
whole pixels, so it is quantised to roughly 4 source pixels. While the offsets are handed to
the model this costs nothing -- the mapping into crop space is exact either way. It starts to
matter once the network is shown three consecutive frames, because a window that twitches
puts motion into the stack that the layer never had.

The question is therefore not "is the offset track noisy" but "is it noisy *in crop pixels*,
at the scale the network sees". A 4-pixel jump in a crop scaled by 0.077 is a third of a
pixel and beneath notice; the same jump at scale 2.2 is nine pixels and would dominate.

    python scripts/exp_offset_jitter.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.smoothing import SAVGOL, smooth_track                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/results/offset_jitter.json')
    ap.add_argument('--window', type=int, default=9)
    args = ap.parse_args()

    rows = []
    print(f'{"layer":<47} {"scale":>7} {"src px":>8} {"crop px":>8} {"residual":>9}')
    for d in sorted(Path(args.dataset).iterdir()):
        meta_path = d / 'meta.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        crop = meta['crop']
        frames = meta['frames']['index']
        off = np.array([crop['offsets'][str(f)] for f in frames], float)
        scale = crop['scale']

        step = np.linalg.norm(np.diff(off, axis=0), axis=1)
        # Residual against a smoothed track: the part of the motion that is quantisation
        # rather than the layer actually travelling.
        smoothed = smooth_track(off, args.window, SAVGOL)
        resid = np.linalg.norm(off - smoothed, axis=1)
        row = {'layer': d.name, 'scale': scale, 'frames': len(frames),
               'mean_step_src_px': float(step.mean()) if len(step) else 0.0,
               'max_step_src_px': float(step.max()) if len(step) else 0.0,
               'mean_step_crop_px': float(step.mean() * scale) if len(step) else 0.0,
               'max_step_crop_px': float(step.max() * scale) if len(step) else 0.0,
               'mean_residual_crop_px': float(resid.mean() * scale),
               'max_residual_crop_px': float(resid.max() * scale)}
        rows.append(row)
        print(f'{d.name[:46]:<47} {scale:>7.3f} {row["mean_step_src_px"]:>8.2f} '
              f'{row["mean_step_crop_px"]:>8.2f} {row["mean_residual_crop_px"]:>9.3f}')

    worst = max(rows, key=lambda r: r['mean_residual_crop_px'])
    summary = {'window': args.window, 'per_layer': rows, 'worst_layer': worst['layer'],
               'worst_mean_residual_crop_px': worst['mean_residual_crop_px'],
               'worst_max_residual_crop_px': worst['max_residual_crop_px']}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f'\nworst layer by crop-space offset jitter: {worst["layer"]}')
    print(f'  mean residual {worst["mean_residual_crop_px"]:.3f} crop px, '
          f'max {worst["max_residual_crop_px"]:.3f} crop px')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
