"""``python -m roto.v2 <command>`` -- v2's own entry point.

Separate from ``roto.cli`` so that v2 can be driven without importing anything v1 owns. See
the boundary note in :mod:`roto.v2`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import subset as subset_mod
from .dataset import build_dataset
from .ledger import RED, build_ledger, render


def _shots(args) -> tuple[str, ...] | None:
    if args.all:
        return None
    if args.shots:
        return tuple(args.shots)
    return {'tier1': subset_mod.TIER1,
            'tier2': subset_mod.TIER1 + subset_mod.TIER2}[args.tier]


def cmd_dataset(args) -> int:
    from .build import v2_crop_config
    m = build_dataset(args.data_root, args.out, _shots(args),
                      cfg=v2_crop_config(size=args.size, stride=args.stride),
                      qc_gates=not args.no_qc_gate)
    c = m['counts']
    print(f"{args.out}: {c['shots']} shots, {c['elements']} elements "
          f"({c['gold']} gold, {c['silver']} silver), {c['shapes']} shapes, "
          f"{c['frames']} element-frames")
    print(f"  {c['trained_elements']} trainable, {c['frames_held']} frames held, "
          f"held shots {json.loads((Path(args.out) / 'splits.json').read_text())['held_shots']}")
    if m['shots_failed_qc']:
        print(f"  QC failed: {m['shots_failed_qc']}")
    return 0


def cmd_ledger(args) -> int:
    rows = build_ledger(args.dataset)
    print(render(rows))
    if args.json:
        Path(args.json).write_text(json.dumps([r.as_dict() for r in rows], indent=2))
    return 1 if any(r.status == RED for r in rows) else 0


def cmd_subset(args) -> int:
    print(f'TIER1    {len(subset_mod.TIER1):2d}  {", ".join(subset_mod.TIER1)}')
    print(f'  held   {len(subset_mod.HOLDOUT_SHOTS):2d}  {", ".join(subset_mod.HOLDOUT_SHOTS)}')
    print(f'TIER2    {len(subset_mod.TIER2):2d}  {", ".join(subset_mod.TIER2)}')
    for label, d in (('DEFERRED', subset_mod.DEFERRED), ('EXCLUDED', subset_mod.EXCLUDED)):
        print(f'{label} {len(d):2d}')
        for k, v in d.items():
            print(f'  {k}  {v}')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='roto.v2', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('dataset', help='build a v2 dataset')
    d.add_argument('out', help='output root, e.g. datasets/v003')
    d.add_argument('--data-root', default='data/spline_dataset_08_25_26')
    d.add_argument('--tier', default='tier1', choices=('tier1', 'tier2'))
    d.add_argument('--shots', nargs='+', help='explicit shot names, overriding --tier')
    d.add_argument('--all', action='store_true', help='every shot under --data-root')
    d.add_argument('--size', type=int, default=256)
    d.add_argument('--stride', type=int, default=1)
    d.add_argument('--no-qc-gate', action='store_true',
                   help='record pairing QC failures instead of dropping the shot')
    d.set_defaults(fn=cmd_dataset)

    g = sub.add_parser('ledger', help='run the exactness ledger; exits non-zero on RED')
    g.add_argument('dataset', help='dataset root, e.g. datasets/v003')
    g.add_argument('--json', help='also write the rows here')
    g.set_defaults(fn=cmd_ledger)

    s = sub.add_parser('subset', help='print the shot selection and its reasons')
    s.set_defaults(fn=cmd_subset)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
