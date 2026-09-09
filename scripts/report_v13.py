"""Score one trained run the way v1.3 reports: means, tails, and the split.

Same reconstruction as `report_v12.py` -- predict, choose keys, rebuild, render, compare --
with three things v1.2 did not have:

    the split           A v002 run trains on 11 of 13 layers. Every layer is still scored,
                        and the two it never saw are reported *apart* from the rest rather
                        than diluted into the aggregate. Their query rows do not exist in the
                        checkpoint, so they are given freshly initialised ones -- see
                        `reconstruct.untrained_queries` for why that, and not another layer's.
    --key-bias          How far the key-timing head may tighten the DP's tolerance where it
                        expects a key. The head's only route into the output.
    --key-slack         How far it may loosen it where it expects none -- the half that can
                        raise precision, which is what caps key F1.
    --key-thresh        Where the head's *own* key set is read off, for reporting only.
                        Swept rather than fixed at 0.5, because a class-balanced loss shifts
                        the calibrated threshold by log(pos_weight) and 0.5 is not it.

The conventions a prediction is rendered with now come from the dataset's own `meta.json`
rather than from a fresh `RenderConfig` -- see `render.raster.config_from_meta`. That is not a
refinement: on `datasets/v002` the old path scored the *artist's own program* at 0.9925 on the
layer with 279 open strokes.

    python scripts/report_v13.py --run v002_final
    python scripts/report_v13.py --run v002_keytime --key-bias 0.5 --smooth 9 --tol 0.5 --refit
    python scripts/report_v13.py --run v002_final --motion predicted --label v002_final_e2e
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
                                    RebuildConfig, assemble, load_model, predict,
                                    untrained_queries)
from roto.model.report import by_training_status                        # noqa: E402
from roto.model.smoothing import BOXCAR, KINDS                          # noqa: E402
from roto.sfx.json_ir import from_json_ir                               # noqa: E402

RUN_DIRS = ('v1.3/runs', 'v1.2/runs', 'v1.1/runs')
"""Where `--run` looks, in order: v1.3's own rungs first, then the earlier ones, so any
v1.1/v1.2 checkpoint can be re-scored under v1.3's columns without copying it."""

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
    """How much of the crop the layer fills at ``frame``.

    Reported beside every worst frame on purpose. Soft IoU is a ratio, so a frame where the
    layer covers 0.4% of the crop can score badly for an absolute error nobody would see, and
    a worst-frame table that does not say so invites chasing the wrong frames."""
    meta = json.loads((layer_dir / 'meta.json').read_text())
    by = dict(zip(meta['frames']['index'], meta['frames']['coverage']))
    return float(by.get(frame, float('nan')))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None, help=f'name under {" or ".join(RUN_DIRS)}')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--dataset', default=None,
                    help="default: the dataset the checkpoint records it was trained on")
    ap.add_argument('--out', default='v1.3/results')
    ap.add_argument('--label', default=None)
    ap.add_argument('--tol', type=float, default=None)
    ap.add_argument('--smooth', type=int, default=None)
    ap.add_argument('--smooth-kind', default=BOXCAR, choices=list(KINDS))
    ap.add_argument('--refit', action='store_true', help='fit key values to the raw track')
    ap.add_argument('--key-bias', type=float, default=0.0,
                    help="how far the key-timing head may tighten the DP's tolerance")
    ap.add_argument('--key-slack', type=float, default=0.0,
                    help='how far the head may loosen the tolerance where it expects no key')
    ap.add_argument('--key-thresh', type=float, default=0.5,
                    help="threshold for the head's own key set (reporting only)")
    ap.add_argument('--affine', action='store_true',
                    help="the transform head alone, on the artist's own shapes")
    ap.add_argument('--motion', default=ARTIST, choices=list(MOTION_SOURCES),
                    help="'predicted' drops the teacher-forced transform track entirely")
    ap.add_argument('--supersample', type=int, default=None)
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--figures-dir', default='v1.3/figures')
    ap.add_argument('--max-figures', type=int, default=13)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    ckpt = args.checkpoint or find_run(args.run)
    label = args.label or args.run or Path(ckpt).stem
    if args.figures:
        from matplotlib import pyplot as plt
        from roto.model.figures import contact_sheet, element_figure

    net, ck = load_model(ckpt)
    sbase, gbase = dict(ck.get('shape_base', {})), dict(ck.get('group_base', {}))
    splits, tcfg = ck.get('splits', {}), (ck.get('config') or {})
    withheld = list(ck.get('withheld_layers', []))
    # The dataset the checkpoint was trained on. Never guessed: a v1.1 or v1.2 checkpoint has
    # no `dataset` field, and defaulting it to v002 would score a model trained on v001's
    # alphas against targets drawn under three different render conventions -- which reads as
    # a quality loss and is the exact class of mistake this round spent its first hour on. So
    # a checkpoint that does not say demands `--dataset` rather than picking one.
    dataset = args.dataset or ck.get('dataset')
    if not dataset:
        raise SystemExit(
            f'{ckpt} records no dataset (it predates v1.3), so pass --dataset explicitly. '
            'Scoring it against the wrong alphas would read as a quality loss: v001 and v002 '
            'differ in three render conventions and one interpolation law.')

    kw = {k: v for k, v in [('tol_px', args.tol), ('smooth', args.smooth)] if v is not None}
    cfg = RebuildConfig(smooth_kind=args.smooth_kind, refit_values=args.refit,
                        predicted_affine=args.affine, motion=args.motion,
                        supersample=args.supersample, key_bias=args.key_bias,
                        key_slack=args.key_slack, key_thresh=args.key_thresh, **kw)

    dirs = sorted(p for p in Path(dataset).iterdir() if (p / 'meta.json').exists())
    rng = np.random.default_rng(args.seed)
    rows, sheet = [], []
    for d in dirs:
        el = load_element(d)
        frozen = el.layer_id in withheld
        if frozen or el.layer_id not in sbase:
            # No query rows exist for this layer. Fresh ones, seeded, rather than another
            # layer's -- see reconstruct.untrained_queries.
            sb, gb = untrained_queries(net, el, seed=args.seed)
        else:
            sb, gb = sbase[el.layer_id], gbase.get(el.layer_id, 0)
        crop_pts, pred_aff, key_prob = predict(net, el, sb, gb)
        rec = assemble(el, crop_pts, pred_aff, cfg, key_prob=key_prob)
        s = rec.summary()
        s['in_train'] = not frozen
        # Kept per layer, not aggregated: 1,810 floats for the whole archive, and it is what
        # makes the worst-frame figures and any later distribution question free.
        s['soft_iou_per_frame'] = [round(float(x), 6) for x in rec.soft_iou]
        s['frame_index'] = [int(f) for f in rec.frames]
        s['worst_frame_coverage'] = coverage_of(d, s['worst_frame'])
        held_pos = np.array(splits.get(el.layer_id, {}).get('held', []), int)
        if len(held_pos) and not frozen:
            train_pos = np.array(splits[el.layer_id]['train'], int)
            s['held_soft_iou'] = float(rec.soft_iou[held_pos].mean())
            s['train_soft_iou'] = float(rec.soft_iou[train_pos].mean())
            s['held_frames'] = int(len(held_pos))
        rows.append(s)
        if args.figures and not frozen:
            fdir = Path(args.figures_dir); fdir.mkdir(parents=True, exist_ok=True)
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
        print(f'{"  " if frozen else ""}{rec.layer_id[:44]:<45}'
              f'{" [HELD OUT]" if frozen else ""} soft {s["mean_soft_iou"]:.4f}  '
              f'worst frame {s["min_soft_iou"]:.4f} @{s["worst_frame"]}'
              f'({s["worst_frame_coverage"]*100:.1f}% cover)  '
              f'<.90 {s["frames_below_0.90"]:>3}/{s["frames"]:<3}  '
              f'pt {s["point_err_px"]:5.2f}/p95 {s["p95_point_err_px"]:6.2f}px  '
              f'jit {s["jitter_px"]:4.2f}  keys x{s["key_ratio"]:.2f}  '
              f'F1 {s["key_f1"]:.3f}/{s["baseline_pipeline_key_f1_random"]:.3f}rnd'
              + (f'  head F1 {s["head_key_f1"]:.3f} '
                 f'({s["head_key_f1_over_best_baseline"]:+.3f} vs baseline)'
                 if s.get('head_key_f1') else ''),
              flush=True)

    totals = {'label': label, 'checkpoint': str(ckpt), 'dataset': dataset,
              'rebuild': asdict(cfg), 'train_config': tcfg, 'steps': tcfg.get('steps'),
              'seed': tcfg.get('seed'), 'n_params': ck.get('n_params'),
              'split_source': ck.get('split_source'),
              **by_training_status(rows, withheld)}

    if args.figures and sheet:
        sheet.sort(key=lambda r: -r['soft'])
        contact_sheet(sheet[:args.max_figures],
                      Path(args.figures_dir) / 'contact_sheet.png',
                      f'roto v1.3 — {label} ({len(sheet)} trained layers)')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    suffix = ''.join(['_affine' if args.affine else '',
                      '_e2e' if args.motion != ARTIST else '',
                      '_refit' if args.refit else '',
                      f'_{args.smooth_kind}' if args.smooth_kind != BOXCAR else '',
                      f'_kb{args.key_bias:g}' if args.key_bias else '',
                      f'_ks{args.key_slack:g}' if args.key_slack else '',
                      f'_ss{args.supersample}' if args.supersample else ''])
    (out / f'score_{label}{suffix}.json').write_text(json.dumps(totals, indent=2))
    held = (f'   held frames {totals["held_soft_iou"]:.4f} vs trained '
            f'{totals["train_soft_iou"]:.4f} (gap {totals["held_gap"]:+.4f})'
            if 'held_soft_iou' in totals else '')
    print(f'\n{label}{suffix}  [{totals["layers"]} trained layers]  '
          f'soft IoU {totals["mean_soft_iou"]:.4f}   '
          f'worst layer {totals["worst_layer_soft_iou"]:.4f} ({totals["worst_layer"][:28]})   '
          f'worst frame {totals["worst_frame_soft_iou"]:.4f} '
          f'({totals["worst_frame_layer"][:24]} @{totals["worst_frame"]})\n'
          f'{" ":<{len(label) + len(suffix)}}  point {totals["point_err_px"]:.2f}px  '
          f'p95 {totals["p95_point_err_px"]:.2f}px   jitter {totals["jitter_px"]:.2f}px   '
          f'keys x{totals["key_ratio"]:.2f}   F1 {totals["key_f1"]:.3f} '
          f'(strict {totals["key_f1_strict"]:.3f}, '
          f'{totals["key_f1_over_random"]:+.3f} over the same keys placed at random)'
          + (f'   head F1 {totals["head_key_f1"]:.3f} '
             f'(P {totals["head_key_precision"]:.2f} R {totals["head_key_recall"]:.2f})'
             if totals['head_key_f1'] else '')
          + (f'\n{" ":<{len(label) + len(suffix)}}  the head against trivial baselines: '
             f'{totals["head_key_f1"]:.3f} vs {totals["baseline_key_f1_all_live"]:.3f} '
             f'(fire on every live frame) and {totals["baseline_key_f1_random"]:.3f} '
             f'(random, same count) '
             f'-> {totals["head_key_f1_over_best_baseline"]:+.3f} over the better of them'
             if totals['head_key_f1'] else '')
          + f'\n{" ":<{len(label) + len(suffix)}}  frames <0.90 '
            f'{totals["frames_below_0.90"]}/{totals["frames"]} '
            f'({totals["frames_below_0.90_pct"]:.2f}%), worst layer '
            f'{totals["worst_layer_frames_below_0.90_pct"]:.2f}% '
            f'({totals["worst_layer_frames_below_0.90"][:26]}){held}')
    if totals.get('held_layers'):
        h = totals['held_layers']
        print(f'{" ":<{len(label) + len(suffix)}}  layers never trained on '
              f'({", ".join(x[:26] for x in totals["withheld_layers"])}): '
              f'soft IoU {h["mean_soft_iou"]:.4f}, point {h["point_err_px"]:.1f}px '
              f'-- untrained query rows, so this is the encoder alone')


if __name__ == '__main__':
    main()
