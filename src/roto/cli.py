"""Command line surface, deliberately identical to ``roto_toolkit.py``.

    roto inventory  <data_root>
    roto parse      <shot_dir> [-o out.json]
    roto verify     <shot_dir>  [--scale 0.35] [--stride 5] [--viz-dir viz/]
    roto verify-all <data_root> [--scale 0.35] [--stride 5] [--report report.csv]

plus the clean-alpha dataset commands, which use no EXR at all:

    roto elements   [--data-root ...] [--all]
    roto dataset    <out_dir> [--data-root ...] [--element ID] [--size 512] [--stride 1]

Same names, same flags, same defaults, so anything already scripted against the toolkit keeps
working. What changed is underneath: per-key interpolation instead of a global mode, sRGB
matte decoding, anti-aliased edges, and soft-IoU reported next to thresholded IoU.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .data.shots import Shot, find_shot, find_shots
from .eval.match import Assignment, assign_channels, element_doc
from .eval.metrics import dice, error_profile, iou, soft_iou
from .matte.exr import load_channels
from .render.raster import RenderConfig, render_union
from .sfx.json_ir import write_json_ir
from .sfx.read import read_sfx

FIELDS = ['shot', 'matte_dir', 'channel', 'layer', 'frames_tested', 'mean_iou',
          'min_iou', 'mean_dice', 'mean_soft_iou', 'worst_frame', 'worst_edge_only']


# ---- inventory -----------------------------------------------------------------

def cmd_inventory(args: argparse.Namespace) -> list[dict]:
    rows = []
    for shot in find_shots(args.data_root):
        doc = read_sfx(shot.sfx, validate=False)
        st = doc.stats()
        rows.append({'shot': shot.name, **st})
        mattes = ', '.join(f'{k}({len(v)}f)' for k, v in shot.mattes.items())
        print(f'{shot.name}')
        print(f'  {doc.width}x{doc.height}  start {doc.start_frame}  {doc.duration} frames  '
              f'dialect {doc.dialect}')
        print(f'  {len(doc.roots)} top layers ({", ".join(r.name for r in doc.roots)})')
        print(f'  {st["layers"]} layers, {st["shapes"]} shapes '
              f'({st["ephemeral"]} ephemeral, {st["open_strokes"]} open strokes), '
              f'{st["points"]} points, {st["keys"]} path keys')
        print(f'  {st["tracked_layers"]} tracked layers '
              f'({st["animated_transforms"]} animated), types {st["shape_types"]}')
        print(f'  mattes: {mattes or "none"}')
    return rows


# ---- parse ---------------------------------------------------------------------

def cmd_parse(args: argparse.Namespace) -> Path:
    shot = find_shot(args.shot_dir)
    if shot.sfx is None:
        raise SystemExit(f'no .sfx found under {args.shot_dir}')
    doc = read_sfx(shot.sfx)
    out = Path(args.output) if args.output else Path(args.shot_dir) / 'roto_ir.json'
    write_json_ir(doc, out)
    st = doc.stats()
    print(f'IR written: {out}  (layers={st["layers"]}, shapes={st["shapes"]}, '
          f'path keyframes={st["keys"]}, {out.stat().st_size / 1e6:.1f} MB)')
    return out


# ---- verify --------------------------------------------------------------------

def _verify_shot(shot: Shot, scale: float, stride: int, viz_dir: str | None,
                 quiet: bool = False) -> list[dict]:
    if not shot.is_usable:
        print(f'skip {shot.name}: sfx={shot.sfx is not None} mattes={len(shot.mattes)}')
        return []
    doc = read_sfx(shot.sfx)
    cfg = RenderConfig()
    assignments = assign_channels(doc, shot.mattes, scale=scale,
                                  sample_frames=shot.sample_frames(3), cfg=cfg)

    if not quiet:
        print(f'\n=== {shot.name}  ({doc.width}x{doc.height}, start {doc.start_frame}) ===')
        for a in assignments:
            print(f'  {a.matte} [{a.channel}]  <->  {a.label}  '
                  f'(sample soft IoU {a.score:.3f})')

    rows = []
    for a in assignments:
        frames = shot.mattes[a.matte]
        tested = sorted(frames)[::stride]
        ious, dices, softs = [], [], []
        worst = (2.0, None)
        for f in tested:
            truth = load_channels(frames[f], scale).get(a.channel)
            if truth is None:
                continue
            pred = render_union(element_doc(doc, a.members), f - doc.start_frame,
                                cfg, scale)
            if pred.shape != truth.shape:
                continue
            v = iou(pred, truth)
            ious.append(v)
            dices.append(dice(pred, truth))
            softs.append(soft_iou(pred, truth))
            if v < worst[0]:
                worst = (v, (f, pred, truth))
        if not ious:
            continue

        prof = error_profile(worst[1][1], worst[1][2]) if worst[1] else None
        rows.append({
            'shot': shot.name, 'matte_dir': a.matte, 'channel': a.channel,
            'layer': a.label, 'frames_tested': len(ious),
            'mean_iou': float(np.mean(ious)), 'min_iou': float(np.min(ious)),
            'mean_dice': float(np.mean(dices)), 'mean_soft_iou': float(np.mean(softs)),
            'worst_frame': worst[1][0] if worst[1] else '',
            'worst_edge_only': bool(prof.is_edge_only) if prof else '',
        })
        if not quiet:
            tag = ('edge-only' if prof and prof.is_edge_only else 'STRUCTURAL') if prof else ''
            print(f'  {a.matte} [{a.channel}] over {len(ious)} frames: '
                  f'mean IoU {np.mean(ious):.3f}  min {np.min(ious):.3f}  '
                  f'mean Dice {np.mean(dices):.3f}  soft IoU {np.mean(softs):.3f}  '
                  f'worst f{worst[1][0]} {tag}')

        if viz_dir and worst[1]:
            f, pred, truth = worst[1]
            _write_viz(Path(viz_dir), f'{shot.name}_{a.matte}_{a.channel}_f{f}_worst.png',
                       pred, truth)
    return rows


def _write_viz(directory: Path, name: str, pred: np.ndarray, truth: np.ndarray) -> None:
    import cv2
    directory.mkdir(parents=True, exist_ok=True)
    viz = np.zeros((*pred.shape, 3), np.uint8)
    viz[..., 1] = (np.clip(truth, 0, 1) * 255).astype(np.uint8)   # green = delivered matte
    viz[..., 2] = (np.clip(pred, 0, 1) * 255).astype(np.uint8)    # red = ours; overlap = yellow
    cv2.imwrite(str(directory / name), viz)


def cmd_verify(args: argparse.Namespace) -> list[dict]:
    return _verify_shot(find_shot(args.shot_dir), args.scale, args.stride, args.viz_dir)


def cmd_verify_all(args: argparse.Namespace) -> list[dict]:
    rows = []
    for shot in find_shots(args.data_root):
        rows += _verify_shot(shot, args.scale, args.stride, args.viz_dir)
    if args.report:
        with open(args.report, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f'\nreport written: {args.report}')
    if rows:
        print(f'\nOVERALL: mean IoU {np.mean([r["mean_iou"] for r in rows]):.3f}  '
              f'mean soft IoU {np.mean([r["mean_soft_iou"] for r in rows]):.3f}  '
              f'across {len(rows)} matte channels / '
              f'{sum(r["frames_tested"] for r in rows)} frame checks')
        bad = [r for r in rows if r['worst_edge_only'] is False]
        if bad:
            print(f'  {len(bad)} channel(s) with STRUCTURAL error at their worst frame: '
                  f'{", ".join(r["shot"] + "/" + r["channel"] for r in bad)}')
    return rows


# ---- clean-alpha dataset --------------------------------------------------------

def _select(args: argparse.Namespace):
    from .dataset import discover
    els = discover(args.data_root)
    if args.element:
        wanted = set(args.element)
        els = [e for e in els if e.element_id in wanted]
        if not els:
            raise SystemExit(f'no element matched {sorted(wanted)}')
        return els
    return [e for e in els if e.is_target]


def cmd_elements(args: argparse.Namespace) -> list:
    from .dataset import discover
    els = discover(args.data_root)
    shown = els if args.all else [e for e in els if e.is_target]
    for e in shown:
        st = e.stats
        flag = ' ' if e.is_target else 'x'
        print(f'{flag} {e.element_id:<58} {st.shapes:>5}sh {st.groups:>4}grp '
              f'{st.keys:>6}k  k/live {st.keys_per_live_frame:>5.2f}  '
              f'open {st.open_frac:>4.2f}  {", ".join(e.excluded_by)}')
    keep = [e for e in els if e.is_target]
    print(f'\n{len(keep)}/{len(els)} elements are v1 targets, '
          f'across {len({e.shot for e in keep})} shots '
          f'({sum(e.stats.shapes for e in keep)} shapes, '
          f'{sum(e.stats.keys for e in keep)} artist keys)')
    return shown


def cmd_dataset(args: argparse.Namespace) -> list[Path]:
    from .dataset import CropConfig, build
    cfg = CropConfig(size=args.size, supersample=args.supersample, stride=args.stride)
    out = []
    for e in _select(args):
        b = build(e, args.data_root, args.out_dir, cfg)
        cov = b.meta['frames']['coverage']
        c = b.meta['crop']
        off = c['frames_partly_off_source']
        print(f'{b.element_id}')
        print(f'  {len(b.frames)} frames  {b.n_shapes} shapes  {c["mode"]} crop '
              f'{c["size_src_px"]}px -> {args.size}x{args.size} (scale {c["scale"]:.3f})')
        print(f'  coverage {min(cov):.3f}..{max(cov):.3f}  mean {sum(cov)/len(cov):.3f}'
              + (f'  |  {off} frame(s) partly off-source' if off else ''))
        print(f'  {b.directory}')
        out.append(b.directory)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='roto', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('inventory', help='per-shot summary')
    p.add_argument('data_root')
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser('parse', help='.sfx -> roto_ir.json')
    p.add_argument('shot_dir')
    p.add_argument('-o', '--output')
    p.set_defaults(fn=cmd_parse)

    for name, fn in (('verify', cmd_verify), ('verify-all', cmd_verify_all)):
        p = sub.add_parser(name, help='render and score against the delivered mattes')
        p.add_argument('shot_dir' if name == 'verify' else 'data_root')
        p.add_argument('--scale', type=float, default=0.35)
        p.add_argument('--stride', type=int, default=5)
        p.add_argument('--viz-dir', default=None)
        if name == 'verify-all':
            p.add_argument('--report', default=None)
        p.set_defaults(fn=fn)

    p = sub.add_parser('elements', help='list clean-alpha training elements (no EXR)')
    p.add_argument('--data-root', default='data/extracted/test_data')
    p.add_argument('--all', action='store_true', help='include excluded elements')
    p.set_defaults(fn=cmd_elements)

    p = sub.add_parser('dataset', help='render clean alpha + spline program per element')
    p.add_argument('out_dir')
    p.add_argument('--data-root', default='data/extracted/test_data')
    p.add_argument('--element', action='append', default=[],
                   help='build just this element_id; repeatable, bypasses exclusions')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--supersample', type=int, default=4)
    p.add_argument('--stride', type=int, default=1,
                   help='emit every Nth frame (1 = all; larger for a smoke run)')
    p.set_defaults(fn=cmd_dataset)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
