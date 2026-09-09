"""Score one trained run the way v1.2 reports: means *and* tails.

Same reconstruction as `report_v11.py` -- predict, choose keys, rebuild, render, compare --
with the handover's worst-case columns attached, per-frame soft IoU kept so the worst frames
can be looked at afterwards without re-scoring, and two axes v1.1 did not have:

    --affine            the transform head alone: the artist's control points, the model's
                        motion. Pure motion error, nothing in front of it.
    --motion predicted  the first de-teacher-forced reconstruction: the model's geometry AND
                        the model's motion, through the keyframe stage. See
                        `roto.model.reconstruct` for why that is not the same number.

    python scripts/report_v12.py --run final_long_v2 --smooth 9 --tol 0.5 --refit
    python scripts/report_v12.py --run final_long_v2 --affine
    python scripts/report_v12.py --run final_long_v2 --motion predicted --label final_long_v2_e2e
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                                # noqa: E402
from roto.model.reconstruct import (ARTIST, MOTION_SOURCES,             # noqa: E402
                                    RebuildConfig, load_model, reconstruct)
from roto.model.report import run_totals                                # noqa: E402
from roto.model.smoothing import BOXCAR, KINDS                          # noqa: E402
from roto.sfx.json_ir import from_json_ir                               # noqa: E402

RUN_DIRS = ('v1.2/runs', 'v1.1/runs')
"""Where `--run` looks, in order. v1.2's own rungs first, then v1.1's, so a v1.1 rung can be
re-scored under the new columns without copying its checkpoint."""

MIN_COVERAGE = 0.02
"""Alpha coverage a frame needs before a figure may show it. An empty render against an empty
target scores a perfect IoU, so an unfiltered pick can caption a blank panel 1.0000."""


def find_run(name: str) -> str:
    for d in RUN_DIRS:
        p = Path(d) / name / 'model.pt'
        if p.exists():
            return str(p)
    raise SystemExit(f'no checkpoint for run {name!r} under {", ".join(RUN_DIRS)}')


def pick_frame(layer_dir: Path, rng: np.random.Generator) -> int:
    """A seeded random frame on which the layer actually draws something. Seeded and random,
    not best or worst: the best makes the figures a highlight reel, the worst a bug report."""
    meta = json.loads((layer_dir / 'meta.json').read_text())
    frames, cover = meta['frames']['index'], meta['frames']['coverage']
    ok = [f for f, c in zip(frames, cover) if c >= MIN_COVERAGE]
    if not ok:
        ok = [frames[int(np.argmax(cover))]]
    return int(ok[int(rng.integers(len(ok)))])


def coverage_of(layer_dir: Path, frame: int) -> float:
    """How much of the crop the layer actually fills at ``frame``.

    Reported beside every worst frame on purpose. Soft IoU is a ratio, so a frame where the
    layer covers 0.4% of the crop can score badly for an absolute error nobody would see,
    and a worst-frame table that does not say so invites chasing the wrong frames."""
    meta = json.loads((layer_dir / 'meta.json').read_text())
    by = dict(zip(meta['frames']['index'], meta['frames']['coverage']))
    return float(by.get(frame, float('nan')))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None, help=f'name under {" or ".join(RUN_DIRS)}')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.2/results')
    ap.add_argument('--label', default=None)
    ap.add_argument('--tol', type=float, default=None)
    ap.add_argument('--smooth', type=int, default=None)
    ap.add_argument('--smooth-kind', default=BOXCAR, choices=list(KINDS))
    ap.add_argument('--refit', action='store_true', help='fit key values to the raw track')
    ap.add_argument('--affine', action='store_true',
                    help="the transform head alone, on the artist's own shapes")
    ap.add_argument('--motion', default=ARTIST, choices=list(MOTION_SOURCES),
                    help="'predicted' drops the teacher-forced transform track entirely")
    ap.add_argument('--supersample', type=int, default=None)
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--figures-dir', default='v1.2/figures')
    ap.add_argument('--max-figures', type=int, default=13)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    ckpt = args.checkpoint or find_run(args.run)
    label = args.label or args.run or Path(ckpt).stem
    if args.figures:
        from matplotlib import pyplot as plt
        from roto.model.figures import contact_sheet, element_figure

    net, ck = load_model(ckpt)
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    splits, tcfg = ck.get('splits', {}), (ck.get('config') or {})

    kw = {k: v for k, v in [('tol_px', args.tol), ('smooth', args.smooth)] if v is not None}
    cfg = RebuildConfig(smooth_kind=args.smooth_kind, refit_values=args.refit,
                        predicted_affine=args.affine, motion=args.motion,
                        supersample=args.supersample, **kw)

    dirs = sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists())
    rng = np.random.default_rng(args.seed)
    rows, sheet = [], []
    for d in dirs:
        rec = reconstruct(d, net, cfg, shape_base=sbase.get(d.name, 0),
                          group_base=gbase.get(d.name, 0))
        s = rec.summary()
        # Kept per layer, not aggregated: 1,810 floats for the whole archive, and it is what
        # makes the worst-frame figures and any later distribution question free.
        s['soft_iou_per_frame'] = [round(float(x), 6) for x in rec.soft_iou]
        s['frame_index'] = [int(f) for f in rec.frames]
        s['worst_frame_coverage'] = coverage_of(d, s['worst_frame'])
        held_pos = np.array(splits.get(d.name, {}).get('held', []), int)
        if len(held_pos):
            train_pos = np.array(splits[d.name]['train'], int)
            s['held_soft_iou'] = float(rec.soft_iou[held_pos].mean())
            s['train_soft_iou'] = float(rec.soft_iou[train_pos].mean())
            s['held_frames'] = int(len(held_pos))
        rows.append(s)
        if args.figures:
            fdir = Path(args.figures_dir); fdir.mkdir(parents=True, exist_ok=True)
            el = load_element(d)
            artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
            frame = pick_frame(d, rng)
            i = int(np.where(rec.frames == frame)[0][0])
            fig = element_figure(d, artist, rec.doc, frame, el.crop,
                                 float(rec.soft_iou[i]), float(rec.iou[i]), el.out_px,
                                 rec.layer_id, extra=f'{el.n_shapes} shapes')
            fig.savefig(fdir / f'{rec.layer_id}.png', dpi=140,
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            sheet.append({'dir': d, 'frame': frame, 'artist': artist, 'model': rec.doc,
                          'crop': el.crop, 'out_px': el.out_px,
                          'soft': float(rec.soft_iou[i]), 'layer_id': rec.layer_id})
        print(f'{rec.layer_id[:44]:<45} soft {s["mean_soft_iou"]:.4f}  '
              f'worst frame {s["min_soft_iou"]:.4f} @{s["worst_frame"]}'
              f'({s["worst_frame_coverage"]*100:.1f}% cover)  '
              f'<.95 {s["frames_below_0.95"]:>3}/{s["frames"]:<3}  '
              f'pt {s["point_err_px"]:5.2f}/p95 {s["p95_point_err_px"]:6.2f}px  '
              f'jit {s["jitter_px"]:4.2f}  keys x{s["key_ratio"]:.2f}  '
              f'F1 {s["key_f1"]:.3f}', flush=True)

    totals = {'label': label, 'checkpoint': str(ckpt), 'rebuild': asdict(cfg),
              'train_config': tcfg, 'steps': tcfg.get('steps'), 'seed': tcfg.get('seed'),
              'n_params': ck.get('n_params'), **run_totals(rows)}

    if args.figures and sheet:
        sheet.sort(key=lambda r: -r['soft'])
        contact_sheet(sheet[:args.max_figures],
                      Path(args.figures_dir) / 'contact_sheet.png',
                      f'roto v1.2 — {label} ({len(sheet)} layers)')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    suffix = ''.join(['_affine' if args.affine else '',
                      '_e2e' if args.motion != ARTIST else '',
                      '_refit' if args.refit else '',
                      f'_{args.smooth_kind}' if args.smooth_kind != BOXCAR else '',
                      f'_ss{args.supersample}' if args.supersample else ''])
    (out / f'score_{label}{suffix}.json').write_text(json.dumps(totals, indent=2))
    held = (f'   held {totals["held_soft_iou"]:.4f} vs train {totals["train_soft_iou"]:.4f} '
            f'(gap {totals["held_gap"]:+.4f})' if 'held_soft_iou' in totals else '')
    print(f'\n{label}{suffix}  soft IoU {totals["mean_soft_iou"]:.4f}   '
          f'worst layer {totals["worst_layer_soft_iou"]:.4f} ({totals["worst_layer"][:28]})   '
          f'worst frame {totals["worst_frame_soft_iou"]:.4f} '
          f'({totals["worst_frame_layer"][:24]} @{totals["worst_frame"]})\n'
          f'{" ":<{len(label) + len(suffix)}}  point {totals["point_err_px"]:.2f}px  '
          f'p95 {totals["p95_point_err_px"]:.2f}px   jitter {totals["jitter_px"]:.2f}px   '
          f'keys x{totals["key_ratio"]:.2f}   F1 {totals["key_f1"]:.3f}   '
          f'frames <0.95 {totals["frames_below_0.95"]}/{totals["frames"]} '
          f'({totals["frames_below_0.95_pct"]:.1f}%){held}')


if __name__ == '__main__':
    main()
