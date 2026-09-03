"""Command line surface.

    roto inventory <data_root>                     what is in the archive
    roto parse     <shot_dir> [-o out.json]        .sfx -> readable JSON
    roto layers    [--all]                         roto layers available for training
    roto dataset   <out_dir>                       render mattes + spline programs

A *layer* is a top-level Silhouette layer: a group of shapes that renders to one matte. A
*shape* is one B-spline. Those are the units everything else is counted in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .sfx.json_ir import write_json_ir
from .sfx.read import read_sfx
from .shots import find_shot, find_shots

DEFAULT_DATA_ROOT = 'data/extracted/test_data'


def cmd_inventory(args: argparse.Namespace) -> None:
    for shot in find_shots(args.data_root):
        if shot.sfx is None:
            continue
        doc = read_sfx(shot.sfx, validate=False)
        st = doc.stats()
        print(f'{shot.name}')
        print(f'  {doc.width}x{doc.height}  start {doc.start_frame}  {doc.duration} frames  '
              f'Silhouette {doc.dialect}')
        print(f'  {len(doc.roots)} top-level layers ({", ".join(r.name for r in doc.roots)})')
        print(f'  {st["shapes"]} shapes, {st["points"]} control points, '
              f'{st["keys"]} keyframes')
        print(f'  {st["tracked_layers"]} tracked layers, shape types {st["shape_types"]}')


def cmd_parse(args: argparse.Namespace) -> Path:
    shot = find_shot(args.shot_dir)
    if shot.sfx is None:
        raise SystemExit(f'no .sfx found under {args.shot_dir}')
    doc = read_sfx(shot.sfx)
    out = Path(args.output) if args.output else Path(args.shot_dir) / 'roto_ir.json'
    write_json_ir(doc, out)
    st = doc.stats()
    print(f'{out}  ({st["shapes"]} shapes, {st["keys"]} keyframes, '
          f'{out.stat().st_size / 1e6:.1f} MB)')
    return out


def cmd_layers(args: argparse.Namespace) -> list:
    from .dataset import discover
    found = discover(args.data_root)
    shown = found if args.all else [e for e in found if e.is_target]
    for e in shown:
        st = e.stats
        mark = ' ' if e.is_target else 'x'
        print(f'{mark} {e.layer_id:<58} {st.shapes:>5} shapes {st.groups:>4} groups '
              f'{st.keys:>6} keys  {st.keys_per_live_frame:>5.2f} keys/live frame  '
              f'{", ".join(e.excluded_by)}')
    keep = [e for e in found if e.is_target]
    print(f'\n{len(keep)}/{len(found)} layers selected, across '
          f'{len({e.shot for e in keep})} shots '
          f'({sum(e.stats.shapes for e in keep)} shapes, '
          f'{sum(e.stats.keys for e in keep)} keyframes)')
    return shown


def cmd_dataset(args: argparse.Namespace) -> list[Path]:
    from .dataset import CropConfig, build, discover
    found = discover(args.data_root)
    if args.layer:
        wanted = set(args.layer)
        chosen = [e for e in found if e.layer_id in wanted]
        if not chosen:
            raise SystemExit(f'no layer matched {sorted(wanted)}')
    else:
        chosen = [e for e in found if e.is_target]

    cfg = CropConfig(size=args.size, supersample=args.supersample, stride=args.stride)
    out = []
    for e in chosen:
        b = build(e, args.data_root, args.out_dir, cfg)
        cov = b.meta['frames']['coverage']
        c = b.meta['crop']
        off = c['frames_partly_off_source']
        print(f'{b.layer_id}')
        print(f'  {len(b.frames)} frames  {b.n_shapes} shapes  {c["mode"]} crop '
              f'{c["size_src_px"]}px -> {args.size}x{args.size} (scale {c["scale"]:.3f})')
        print(f'  matte coverage {min(cov):.3f}..{max(cov):.3f}  '
              f'mean {sum(cov) / len(cov):.3f}'
              + (f'  |  {off} frame(s) partly off-source' if off else ''))
        out.append(b.directory)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='roto', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('inventory', help='per-shot summary')
    p.add_argument('data_root', nargs='?', default=DEFAULT_DATA_ROOT)
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser('parse', help='.sfx -> readable JSON')
    p.add_argument('shot_dir')
    p.add_argument('-o', '--output')
    p.set_defaults(fn=cmd_parse)

    p = sub.add_parser('layers', help='roto layers available for training')
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--all', action='store_true', help='include excluded layers')
    p.set_defaults(fn=cmd_layers)

    p = sub.add_parser('dataset', help='render mattes + spline programs')
    p.add_argument('out_dir')
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--layer', action='append', default=[],
                   help='build just this layer id; repeatable, bypasses exclusions')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--supersample', type=int, default=4)
    p.add_argument('--stride', type=int, default=1,
                   help='emit every Nth frame (1 = all; larger for a smoke run)')
    p.set_defaults(fn=cmd_dataset)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
