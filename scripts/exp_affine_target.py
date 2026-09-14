"""Is the transform target a lossless description of the artist's transform track?

The check that must come out at 1.000, and the reason to run it: v1's transform head was
reported as unusable at 0.756 rendered, and a head cannot be judged against a target its own
representation cannot express. So hand ``affine_doc`` the **artist's own** track, round-tripped
through each representation, and render. Any shortfall is the representation's ceiling and
nothing to do with the model.

Two representations, and one of them fails:

* ``affine`` -- v1's 6 document-space matrix entries. Assumes layer transforms are affine.
  Eleven of thirteen layers are; ``FAM red_1`` carries ``m03`` up to 0.028 and
  ``TVC_sh0260 Layer_52`` runs ``m33`` from 0.672 to 1.801, and on those the round trip is
  not the identity.
* ``proj_crop`` -- v1.1's 8-number normalised homography of the *local-to-crop* map. Lossless
  by construction, and the space the network can actually see.

    python scripts/exp_affine_target.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                                # noqa: E402
from roto.geometry import local_to_crop                           # noqa: E402
from roto.model.reconstruct import (affine_doc, score_doc,              # noqa: E402
                                    transform_matrices)


def induced_px(el, mats) -> tuple[float, float]:
    """Mean and max displacement of every live control point, in crop pixels.

    The rendered score is the number that matters, but it saturates: a layer can be
    displaced badly on two shapes and still read 0.99 because the other 590 are fine. This
    reports the geometry directly so a small IoU loss cannot hide a large local error.
    """
    c, errs = el.crop, []
    for fi in range(0, len(el.frames), 7):
        f = int(el.frames[fi])
        for si in range(el.n_shapes):
            if not el.live[fi, si]:
                continue
            p = el.local[fi, si][el.point_mask[si]]
            if not p.size:
                continue
            kw = dict(width=c['width'], height=c['height'], offset=c['offsets'][f],
                      scale=c['scale'], out_px=c['out_px'])
            a = local_to_crop(p, el.matrices[fi, si], **kw)
            b = local_to_crop(p, mats[fi, int(el.group_of[si])], **kw)
            errs.append(np.linalg.norm(a - b, axis=-1) * c['out_px'])
    e = np.concatenate(errs) if errs else np.zeros(1)
    return float(e.mean()), float(e.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/results/affine_target.json')
    ap.add_argument('--stride', type=int, default=7, help='score every Nth frame')
    args = ap.parse_args()

    rows = []
    print(f'{"layer":<47} {"affine(6) soft":>15} {"mean/max px":>14}   '
          f'{"proj_crop(8) soft":>17} {"mean/max px":>14}')
    for d in sorted(Path(args.dataset).iterdir()):
        if not (d / 'meta.json').exists():
            continue
        el = load_element(d)
        frames = [int(f) for f in el.frames[::args.stride]]
        row = {'layer': d.name, 'frames_scored': len(frames)}
        for key, target in [('affine', el.affine), ('proj_crop', el.proj_crop)]:
            t = target.astype(np.float64)
            soft, hard = score_doc(el, affine_doc(el, t), frames)
            mean_px, max_px = induced_px(el, transform_matrices(el, t))
            row[key] = {'soft_iou': float(soft.mean()), 'iou': float(hard.mean()),
                        'mean_px': mean_px, 'max_px': max_px}
        rows.append(row)
        print(f'{d.name[:46]:<47} {row["affine"]["soft_iou"]:>15.4f} '
              f'{row["affine"]["mean_px"]:>6.2f}/{row["affine"]["max_px"]:<7.2f} '
              f'{row["proj_crop"]["soft_iou"]:>17.4f} '
              f'{row["proj_crop"]["mean_px"]:>6.2f}/{row["proj_crop"]["max_px"]:<7.2f}',
              flush=True)

    w = np.array([r['frames_scored'] for r in rows], float)
    summary = {'stride': args.stride, 'per_layer': rows, 'ceiling': {
        k: {'soft_iou': float(np.average([r[k]['soft_iou'] for r in rows], weights=w)),
            'worst_layer': min(rows, key=lambda r: r[k]['soft_iou'])['layer'],
            'worst_soft_iou': min(r[k]['soft_iou'] for r in rows),
            'max_px': max(r[k]['max_px'] for r in rows)}
        for k in ('affine', 'proj_crop')}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print('\nCEILING -- the artist\'s own transform track, round-tripped and re-rendered:')
    for k, c in summary['ceiling'].items():
        print(f'  {k:<10} {c["soft_iou"]:.4f} soft IoU   worst layer {c["worst_layer"][:40]} '
              f'at {c["worst_soft_iou"]:.4f}   worst point error {c["max_px"]:.2f} crop px')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
