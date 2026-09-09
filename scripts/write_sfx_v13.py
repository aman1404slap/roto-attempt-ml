"""Write real Silhouette projects from a reconstruction, for opening in a seat.

The plan's S3: implement the writer, round-trip it through our own reader, and validate one
file in an actual Silhouette seat. The first two are code and are done (`roto.sfx.write`, and
the `read(write(IR))` row of `scripts/ledger.py`). This script is the third's input -- it turns
a scored checkpoint into files somebody can double-click.

**Two files per layer, because they answer different questions.**

`<layer>.reconstructed.sfx` is the artist's *own project* with that one layer's shapes replaced
by ours and every other byte of structure preserved -- the plate, the node graph, the pipes,
the session settings, the other roto layers. This is the one to open. If it renders differently
from the artist's, the difference is our shapes, because nothing else changed. It is also the
only form in which "renders identical" is a question a seat can answer cheaply: the artist's
own version of the same layer is one undo away.

`<layer>.standalone.sfx` is built from the IR alone -- no template, no plate. It exists to
prove the writer needs nothing borrowed, and it is the file the bit-exact round-trip row is
measured on. Do not judge the render from it; it has no source to render against.

Both are written from the **same reconstruction** the report scores, at the same operating
point, so the number in the report is the number in the file.

    python scripts/write_sfx_v13.py --run v002_final --smooth 9 --tol 0.5 --refit
    python scripts/write_sfx_v13.py --run v002_final --layer TVC_SHOTS_sh0260_BG01_v003_roto_v02__Layer_52
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.dataset import load_alpha                                     # noqa: E402
from roto.metrics import soft_iou                                       # noqa: E402
from roto.model.data import load_element                                # noqa: E402
from roto.model.reconstruct import (RebuildConfig, assemble, load_model,  # noqa: E402
                                    predict, score_doc)
from roto.model.smoothing import BOXCAR, KINDS                          # noqa: E402
from roto.render.raster import config_from_meta, render_union           # noqa: E402
from roto.sfx.read import read_sfx                                      # noqa: E402
from roto.sfx.write import rewrite_sfx, write_sfx                       # noqa: E402

RUN_DIRS = ('v1.3/runs', 'v1.2/runs', 'v1.1/runs')


def find_run(name: str) -> Path:
    for d in RUN_DIRS:
        p = Path(d) / name / 'model.pt'
        if p.exists():
            return p
    raise SystemExit(f'no checkpoint for run {name!r} under {", ".join(RUN_DIRS)}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='v002_final')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--out', default='v1.3/sfx')
    ap.add_argument('--layer', action='append', default=[],
                    help='layer id; repeatable. Default: every layer the run trained on')
    ap.add_argument('--tol', type=float, default=0.5)
    ap.add_argument('--smooth', type=int, default=9)
    ap.add_argument('--smooth-kind', default=BOXCAR, choices=list(KINDS))
    ap.add_argument('--refit', action='store_true', default=True)
    ap.add_argument('--key-bias', type=float, default=0.0)
    args = ap.parse_args()

    ckpt = find_run(args.run)
    net, ck = load_model(ckpt)
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    withheld = set(ck.get('withheld_layers', []))
    dataset = Path(args.dataset or ck.get('dataset') or '')
    if not dataset.name:
        raise SystemExit('checkpoint records no dataset; pass --dataset '
                         '-- see scripts/report_v13.py for why')
    cfg = RebuildConfig(tol_px=args.tol, smooth=args.smooth, smooth_kind=args.smooth_kind,
                        refit_values=args.refit, key_bias=args.key_bias)

    wanted = set(args.layer)
    dirs = [p for p in sorted(dataset.iterdir()) if (p / 'meta.json').exists()
            and (p.name in wanted if wanted else p.name not in withheld)]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    written = []

    for d in dirs:
        el = load_element(d)
        meta = json.loads((d / 'meta.json').read_text())
        crop_pts, pred_aff, key_prob = predict(net, el, sbase.get(d.name, 0),
                                               gbase.get(d.name, 0))
        rec = assemble(el, crop_pts, pred_aff, cfg, key_prob=key_prob)

        # 1. standalone, and re-read to confirm the file on disk is the document we scored.
        stand = write_sfx(rec.doc, out / f'{d.name}.standalone.sfx')
        back = read_sfx(stand)
        soft, _ = score_doc(el, back.layer([r.name for r in rec.doc.roots]),
                            [int(f) for f in el.frames])
        drift = float(abs(soft.mean() - rec.soft_iou.mean()))

        # 2. the artist's own project with this layer substituted. Layer name, not layer id:
        #    the id is <shot>__<layer> and the template knows only the layer.
        template = Path(meta['source']['sfx'])
        layer_name = meta['layer']['name']
        rw = None
        if template.exists():
            rw = rewrite_sfx(template, rec.doc, out / f'{d.name}.reconstructed.sfx',
                             layers=[layer_name])

        written.append({'layer_id': d.name, 'layer': layer_name,
                        'standalone': str(stand), 'reconstructed': str(rw) if rw else None,
                        'template': str(template),
                        'dialect': rec.doc.dialect or back.dialect,
                        'shapes': sum(1 for _ in rec.doc.shapes()),
                        'keys': sum(len(s.path) for _, s in rec.doc.shapes()),
                        'keys_artist': rec.keys_artist,
                        'soft_iou_scored': float(rec.soft_iou.mean()),
                        'soft_iou_after_file_round_trip': float(soft.mean()),
                        'round_trip_drift': drift,
                        'bytes_standalone': stand.stat().st_size,
                        'bytes_reconstructed': rw.stat().st_size if rw else None})
        print(f'{d.name[:44]:<46} {rec.keys_predicted:>6} keys '
              f'({rec.keys_predicted / max(1, rec.keys_artist):.2f}x artist)  '
              f'soft IoU {rec.soft_iou.mean():.4f} -> {soft.mean():.4f} through the file '
              f'(drift {drift:.2e})  {stand.stat().st_size / 1e6:.2f} MB', flush=True)

    manifest = {'run': args.run, 'checkpoint': str(ckpt), 'dataset': str(dataset),
                'rebuild': {k: getattr(cfg, k) for k in
                            ('tol_px', 'smooth', 'smooth_kind', 'refit_values', 'key_bias')},
                'worst_round_trip_drift': max((w['round_trip_drift'] for w in written),
                                              default=0.0),
                'files': written}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\n{len(written)} layer(s) written to {out}. Worst drift through the file: '
          f'{manifest["worst_round_trip_drift"]:.2e} soft IoU -- the file on disk is the '
          f'document the report scored.\n-> {out / "manifest.json"}')


if __name__ == '__main__':
    main()
