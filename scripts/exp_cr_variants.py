"""Does the Catmull-Rom endpoint rule matter in pixels? (review §6)

`ir.sample` used to apply Catmull-Rom only when both outer neighbours existed, silently
dropping to linear on the first and last segment of every CR track. FAM -- 8 of the 13
training layers -- is 100% Catmull-Rom, so this looked like a data-corrupting bug, and the
review's own measurement said it was immaterial. This script is why we believe that.

It matters that the two variants are isolated by patching the *interpolation function alone*.
The first attempt at this comparison re-rendered FAM and found gaps of up to 4.7 points
against the stored alphas, which looked like a confirmation and was nothing of the kind -- the
gap was an unrelated supersample mismatch in the scorer. Two changes were in flight at once,
and only isolating one of them told the truth.

    python scripts/exp_cr_variants.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import roto.ir as ir                                                   # noqa: E402
import roto.render.raster as raster                                    # noqa: E402
from roto.dataset import load_alpha                                    # noqa: E402
from roto.ir import CATMULLROM, HOLD                                   # noqa: E402
from roto.metrics import soft_iou                                      # noqa: E402
from roto.render.raster import RenderConfig, render_union              # noqa: E402
from roto.sfx.json_ir import from_json_ir                             # noqa: E402


def make_sample(variant: str):
    """``ir.sample`` with the endpoint rule forced to one variant.

    ``linear_ends`` is v1's behaviour; ``clamped`` is v1.1's. Everything else is identical,
    which is the point -- one line differs between the two renders being compared.
    """
    def sample(keys, frame):
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
        if k0.interp == CATMULLROM:
            if variant == 'linear_ends' and not (0 < i < len(keys) - 2):
                return k0.value * (1 - u) + k1.value * u
            a = keys[i - 1].value if i > 0 else k0.value
            b = keys[i + 2].value if i + 2 < len(keys) else k1.value
            if np.shape(a) == np.shape(k0.value) == np.shape(k1.value) == np.shape(b):
                p0, p1, p2, p3 = a, k0.value, k1.value, b
                return 0.5 * (2 * p1 + (-p0 + p2) * u
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        return k0.value * (1 - u) + k1.value * u
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/results/cr_variants.json')
    ap.add_argument('--stride', type=int, default=17)
    args = ap.parse_args()

    original = ir.sample
    layers = sorted(p for p in Path(args.dataset).glob('FAM*') if (p / 'meta.json').exists())
    rows = []
    try:
        for d in layers:
            meta = json.loads((d / 'meta.json').read_text())
            crop = meta['crop']
            ss = int(meta['render']['supersample'])
            doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
            n_cr = sum(1 for _, s in doc.shapes() for k in s.path if k.interp == CATMULLROM)
            frames = meta['frames']['index'][::args.stride]
            row = {'layer': d.name, 'catmullrom_keys': n_cr, 'frames': len(frames)}
            for variant in ('linear_ends', 'clamped'):
                ir.sample = raster.sample = make_sample(variant)
                scores = []
                for f in frames:
                    x0, y0 = crop['offsets'][str(f)]
                    box = (x0, y0, crop['out_px'][0] / crop['scale'],
                           crop['out_px'][0] / crop['scale'])
                    pred = render_union(doc, int(f), RenderConfig(supersample=ss),
                                        crop['scale'], box)
                    truth = load_alpha(d, int(f))
                    pred = pred[:truth.shape[0], :truth.shape[1]]
                    scores.append(soft_iou(pred, truth))
                row[variant] = float(np.mean(scores))
            row['difference'] = row['clamped'] - row['linear_ends']
            rows.append(row)
            print(f'{d.name[:46]:<47} linear_ends {row["linear_ends"]:.6f}  '
                  f'clamped {row["clamped"]:.6f}  diff {row["difference"]:+.6f}')
    finally:
        ir.sample = raster.sample = original

    worst = max(abs(r['difference']) for r in rows)
    summary = {'stride': args.stride, 'layers': len(rows),
               'max_abs_difference': worst, 'per_layer': rows,
               'conclusion': ('immaterial at this archive key density; the endpoint clamp is '
                              'kept for hygiene because dense targets for 8 of 13 layers flow '
                              'through this function and a sparser-keyed shot would not be as '
                              'forgiving')}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f'\nlargest difference over {len(rows)} FAM layers: {worst:.6f} soft IoU')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
