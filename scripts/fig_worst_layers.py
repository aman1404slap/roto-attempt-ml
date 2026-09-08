"""The two hardest layers under two checkpoints, same frame, side by side.

The figure the v1.1 report leads with. Eight easy layers dilute two hard ones, so the
aggregate moved +0.051 while ``FAM blue_1`` and ``sh0260 Layer_52`` -- the two worst rows in
v1's table, and the two the review predicted were the cheapest win available -- moved +0.039
and +0.038. Only on those two is the change something an artist would call different work
rather than a different number, and putting the same frame under both models is the only way
to show that instead of asserting it.

    python scripts/fig_worst_layers.py --before v1/model/v1.pt --after v1.1/runs/final_long/model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                                # noqa: E402
from roto.model.figures import comparison_figure                        # noqa: E402
from roto.model.reconstruct import (RebuildConfig, load_model,          # noqa: E402
                                    reconstruct)
from roto.model.smoothing import KINDS                                  # noqa: E402
from roto.sfx.json_ir import from_json_ir                               # noqa: E402

HARD = ['FAM_0060_L1_A0003C007_v001__blue_1',
        'TVC_SHOTS_sh0260_BG01_v003_roto_v02__Layer_52']
"""The two shape-dense layers: 1,036 and 592 shapes, and v1's two worst rows."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', default='v1/model/v1.pt')
    ap.add_argument('--after', default='v1.1/runs/final_long/model.pt')
    ap.add_argument('--before-label', default='v1')
    ap.add_argument('--after-label', default='v1.1')
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.1/figures/worst_layers.png')
    ap.add_argument('--layers', nargs='+', default=HARD)
    # The after model is scored at its own swept operating point, the before model at v1's,
    # because that is how each was published; the panels are reconstructions, not a sweep.
    ap.add_argument('--after-smooth', type=int, default=13)
    ap.add_argument('--after-smooth-kind', default='savgol', choices=list(KINDS))
    ap.add_argument('--after-refit', action='store_true', default=True)
    args = ap.parse_args()

    nets = {}
    for key, ckpt in [('before', args.before), ('after', args.after)]:
        net, ck = load_model(ckpt)
        nets[key] = (net, ck.get('shape_base', {}), ck.get('group_base', {}))

    cfgs = {'before': RebuildConfig(),
            'after': RebuildConfig(smooth=args.after_smooth,
                                   smooth_kind=args.after_smooth_kind,
                                   refit_values=args.after_refit)}

    rows = []
    for name in args.layers:
        d = Path(args.dataset) / name
        el = load_element(d)
        # The middle of the track: a frame both models are asked about on equal terms, and
        # not the best or worst either one manages.
        frame = int(el.frames[len(el.frames) // 2])
        row = {'dir': d, 'frame': frame, 'out_px': el.out_px, 'crop': el.crop,
               'layer_id': el.layer_id, 'n_shapes': el.n_shapes,
               'artist': from_json_ir(json.loads((d / 'target_ir.json').read_text())),
               'before_label': args.before_label, 'after_label': args.after_label}
        for key in ('before', 'after'):
            net, sbase, gbase = nets[key]
            rec = reconstruct(d, net, cfgs[key], frames=[frame],
                              shape_base=sbase.get(name, 0), group_base=gbase.get(name, 0))
            row[key] = rec.doc
            row[f'{key}_soft'] = float(rec.soft_iou[0])
        rows.append(row)
        print(f'{el.layer_id[:52]:<53} {args.before_label} {row["before_soft"]:.4f}  ->  '
              f'{args.after_label} {row["after_soft"]:.4f}  '
              f'({row["after_soft"] - row["before_soft"]:+.4f})', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    comparison_figure(rows, args.out,
                      f'The two layers the review predicted would move  —  '
                      f'{args.before_label} against {args.after_label}, same frame')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
