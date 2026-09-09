"""What the *artist's own program* scores against the stored alphas. The ceiling every
model number in v1, v1.1 and v1.2 is measured against.

The handover's Bucket A says pipeline error must be provably zero, and the strongest form of
that test is the one v1.1 used to find three of its four bugs: render the answer and check it
comes out perfect. Run it against `datasets/v001` and it does not. The artist's own shapes,
rendered by today's renderer at the dataset's own supersample, score **0.9968 on the worst
frame** rather than 1.000.

The cause is exact and it is ours. v1's review asked for one interpolation law across a
Catmull-Rom track instead of dropping the first and last segment to linear (S6), and v1.1
made that change -- correctly. But `datasets/v001` was rendered on 2026-09-03, *before* it, so
the stored alphas are drawn under v1's law and every score since is computed under v1.1's. 52%
of this archive's keys are Catmull-Rom, so the disagreement is real wherever a track's first or
last segment is being sampled.

Two independent confirmations, and neither needed new data:

* v1.1's own `results/cr_variants.json` scores the artist's program against these alphas under
  both laws: `linear_ends` 0.9999999, `clamped` 0.9983-0.9999. It was read as "the two laws
  agree to 0.0017, so the change is immaterial" -- true, and it is also a statement about which
  law the targets were built with, which is the part that matters here.
* The renderer at commit 2d1d53e -- the code that wrote the alphas -- reproduces them to
  0.999999 (16-bit PNG quantisation), where today's reproduces them to 0.9968.

So this is not a bug to fix in the renderer: v1.1's law is the right one. It is a **rebuild**
that is owed, and the handover already schedules one. What this script contributes is the size
of the debt, per layer and per frame, so that until the rebuild lands:

* a model number can be quoted against a measured ceiling rather than against 1.000, and
* the worst-frame column the handover asks for can be read honestly -- a frame whose ceiling
  is 0.9968 is not a frame where the model failed by 0.0032.

    python scripts/exp_scoring_ceiling.py                  # every frame, all 13 layers
    python scripts/exp_scoring_ceiling.py --stride 7        # faster
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.dataset import load_alpha                                     # noqa: E402
from roto.ir import CATMULLROM, HOLD, Key, RotoDoc, sample              # noqa: E402
from roto.metrics import soft_iou                                       # noqa: E402
from roto.model.data import load_element                                # noqa: E402
from roto.render.raster import RenderConfig, render_union               # noqa: E402
from roto.sfx.json_ir import from_json_ir                               # noqa: E402


def sample_v1(keys: Sequence[Key], frame: float) -> Any:
    """``ir.sample`` as it stood at commit 2d1d53e, for attribution only.

    The one difference: Catmull-Rom applied *only* to interior segments, with the first and
    last segment falling back to linear. Kept here rather than in ``roto.ir`` because it is
    not a convention anyone should be able to select -- it is the law a frozen dataset happens
    to have been rendered under, and this script exists to price that.
    """
    if not keys:
        raise ValueError('empty track')
    if len(keys) == 1 or frame <= keys[0].frame:
        return keys[0].value
    if frame >= keys[-1].frame:
        return keys[-1].value
    i = 0
    for j, k in enumerate(keys):
        if k.frame <= frame:
            i = j
        else:
            break
    k0, k1 = keys[i], keys[i + 1]
    if k0.interp == HOLD or k1.frame == k0.frame:
        return k0.value
    u = (frame - k0.frame) / (k1.frame - k0.frame)
    if k0.interp == CATMULLROM and 0 < i < len(keys) - 2:
        p0, p1, p2, p3 = keys[i - 1].value, k0.value, k1.value, keys[i + 2].value
        if np.shape(p0) == np.shape(p1) == np.shape(p2) == np.shape(p3):
            return 0.5 * (2 * p1 + (-p0 + p2) * u
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return k0.value * (1 - u) + k1.value * u


def redrawn_under_v1_law(doc: RotoDoc, frames: Sequence[int]) -> RotoDoc:
    """The same document with every shape's path re-keyed densely under :func:`sample_v1`.

    Dense integer keys are lossless for this renderer -- it only ever samples integer frames
    -- so the render isolates the interpolation law and nothing else."""
    for _, shape in doc.shapes():
        shape.path = [Key(int(f), 'linear', np.asarray(sample_v1(shape.path, int(f)), float))
                      for f in frames]
    return doc


def render_at(doc: RotoDoc, el, frame: int, cfg: RenderConfig) -> np.ndarray:
    c = el.crop
    x0, y0 = c['offsets'][int(frame)]
    box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
    return render_union(doc, int(frame), cfg, c['scale'], box)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.2/results/scoring_ceiling.json')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--attribute', action='store_true', default=True,
                    help='also render under v1\'s interpolation law, to attribute the gap')
    args = ap.parse_args()

    rows = []
    for d in sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists()):
        el = load_element(d)
        cfg = RenderConfig(supersample=el.crop['supersample'])
        artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        frames = [int(f) for f in el.frames[::args.stride]]
        soft = []
        for f in frames:
            truth = load_alpha(d, f)
            pred = render_at(artist, el, f, cfg)[:truth.shape[0], :truth.shape[1]]
            soft.append(soft_iou(pred, truth))
        soft = np.asarray(soft)
        worst_i = int(np.argmin(soft))
        row = {'layer_id': el.layer_id, 'frames': len(frames),
               'mean_ceiling': float(soft.mean()), 'min_ceiling': float(soft.min()),
               'worst_frame': frames[worst_i],
               'frames_below_0.999': int((soft < 0.999).sum()),
               'ceiling_per_frame': [round(float(x), 6) for x in soft],
               'frame_index': frames}
        if args.attribute:
            v1doc = redrawn_under_v1_law(
                from_json_ir(json.loads((d / 'target_ir.json').read_text())), el.frames)
            f = frames[worst_i]
            truth = load_alpha(d, f)
            pred = render_at(v1doc, el, f, cfg)[:truth.shape[0], :truth.shape[1]]
            row['worst_frame_under_v1_law'] = soft_iou(pred, truth)
        rows.append(row)
        extra = (f'   under v1\'s law {row["worst_frame_under_v1_law"]:.6f}'
                 if 'worst_frame_under_v1_law' in row else '')
        print(f'{el.layer_id[:46]:<48} ceiling mean {row["mean_ceiling"]:.6f}  '
              f'worst {row["min_ceiling"]:.6f} @{row["worst_frame"]}  '
              f'<0.999 {row["frames_below_0.999"]:>3}/{row["frames"]:<3}{extra}', flush=True)

    w = np.array([r['frames'] for r in rows], float)
    worst = min(rows, key=lambda r: r['min_ceiling'])
    totals = {
        'stride': args.stride, 'layers': len(rows), 'frames': int(w.sum()),
        'mean_ceiling': float(np.average([r['mean_ceiling'] for r in rows], weights=w)),
        'worst_layer_ceiling': min(r['mean_ceiling'] for r in rows),
        'worst_frame_ceiling': worst['min_ceiling'],
        'worst_frame': worst['worst_frame'], 'worst_frame_layer': worst['layer_id'],
        'frames_below_0.999': int(sum(r['frames_below_0.999'] for r in rows)),
        'cause': 'ir.sample Catmull-Rom endpoint law changed (v1 review S6) after '
                 'datasets/v001 was rendered; 52% of this archive\'s keys are catmullrom',
        'fix': 'rebuild the dataset -- the same rebuild the handover already schedules for '
               'the measured render conventions -- then re-baseline once',
        'per_layer': rows,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(totals, indent=2))
    print(f'\nceiling: mean {totals["mean_ceiling"]:.6f}, worst layer '
          f'{totals["worst_layer_ceiling"]:.6f}, worst frame '
          f'{totals["worst_frame_ceiling"]:.6f} ({totals["worst_frame_layer"][:30]} '
          f'@{totals["worst_frame"]}), {totals["frames_below_0.999"]}/{totals["frames"]} '
          f'frames below 0.999\n-> {out}')


if __name__ == '__main__':
    main()
