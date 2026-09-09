"""Does the smoothing change help where the review said it would -- at the motion extremes?

Review S2 asks for one specific validation: rendered soft IoU "specifically on frames of
maximum per-shape velocity, before/after". That is the right test to demand, because it is
where a boxcar's bias lives. A moving average attenuates motion extremes by construction, and
v1 stored key *values* from the smoothed track, so a reconstructed bounce lands short of the
artist's. Averaged over a whole shot that error is diluted by the many frames where nothing is
moving fast; on the fast frames it should be visible.

So each variant is scored twice: on the fastest decile of frames, and on all frames. A change
that is real at the extremes and invisible overall shows up as a difference between those two
columns, which a single headline number cannot express.

Velocity is measured on the *target* track, not the prediction -- the question is which frames
the artist's shapes were moving fastest on, and that must not depend on the model being scored.

    python scripts/exp_peak_frames.py --run final_long
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                               # noqa: E402
from roto.model.reconstruct import (RebuildConfig, load_model, predict,  # noqa: E402
                                    rebuild, score_doc, to_local)
from roto.model.smoothing import BOXCAR, SAVGOL                        # noqa: E402


def frame_velocity(el) -> np.ndarray:
    """Mean per-point speed of the target track at each frame, in crop pixels.

    Centred difference where possible, so a frame is fast because motion passes *through* it
    rather than because it happens to sit next to a jump.
    """
    pts = el.points                                     # (F, S, P, C, 2) crop space, [0,1]
    live = el.live[..., None, None] & el.point_mask[None]
    v = np.zeros(len(pts))
    for i in range(len(pts)):
        a, b = max(0, i - 1), min(len(pts) - 1, i + 1)
        if b == a:
            continue
        d = np.linalg.norm(pts[b] - pts[a], axis=-1) / (b - a) * el.out_px
        m = live[i]
        v[i] = float(d[m].mean()) if m.any() else 0.0
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='final_long')
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/results')
    ap.add_argument('--decile', type=float, default=0.10,
                    help='fraction of fastest frames to treat as the motion extremes')
    args = ap.parse_args()

    net, ck = load_model(f'v1.1/runs/{args.run}/model.pt')
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    variants = [
        ('v1: boxcar, values from filter', RebuildConfig(smooth_kind=BOXCAR)),
        ('savgol, values from filter', RebuildConfig(smooth_kind=SAVGOL)),
        ('boxcar, values refit to raw', RebuildConfig(smooth_kind=BOXCAR, refit_values=True)),
        ('savgol, values refit to raw', RebuildConfig(smooth_kind=SAVGOL, refit_values=True)),
    ]

    rows = []
    dirs = sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists())
    for d in dirs:
        el = load_element(d)
        crop_pts, _, _ = predict(net, el, sbase.get(d.name, 0), gbase.get(d.name, 0))
        local = to_local(el, crop_pts)
        vel = frame_velocity(el)
        k = max(1, int(round(len(vel) * args.decile)))
        fast = [int(el.frames[i]) for i in np.argsort(-vel)[:k]]
        allf = [int(f) for f in el.frames]
        print(f'\n=== {d.name[:52]} ===')
        print(f'    fastest {k} of {len(allf)} frames, '
              f'{vel.max():.2f} px/frame peak vs {vel.mean():.2f} mean')
        for label, cfg in variants:
            doc, st = rebuild(el, local, cfg)
            fast_soft, _ = score_doc(el, doc, fast)
            all_soft, _ = score_doc(el, doc, allf)
            rows.append({'layer': d.name, 'variant': label,
                         'fast_soft_iou': float(fast_soft.mean()),
                         'all_soft_iou': float(all_soft.mean()),
                         'peak_velocity_px': float(vel.max()),
                         'key_ratio': st['keys_predicted'] / max(1, st['keys_artist'])})
            print(f'    {label:<32} fast {rows[-1]["fast_soft_iou"]:.4f}   '
                  f'all {rows[-1]["all_soft_iou"]:.4f}   '
                  f'keys {rows[-1]["key_ratio"]:.2f}x')
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f'peak_frames_{args.run}.json').write_text(
            json.dumps({'run': args.run, 'decile': args.decile, 'rows': rows}, indent=2))

    print(f'\n{"variant":<34} {"fast frames":>12} {"all frames":>11} {"delta":>8}')
    base = None
    for label, _ in variants:
        got = [r for r in rows if r['variant'] == label]
        # Weight by frame count so a 191-frame layer is not outvoted by a 77-frame one.
        f = float(np.mean([r['fast_soft_iou'] for r in got]))
        a = float(np.mean([r['all_soft_iou'] for r in got]))
        base = base or (f, a)
        print(f'{label:<34} {f:>12.4f} {a:>11.4f} {f - base[0]:>+8.4f}')


if __name__ == '__main__':
    main()
