"""Phase 3: how well does curve simplification recover the artist's keyframes?

No model is involved. Each shape's control-point track is reconstructed densely from the
artist's own path keys, the DP is asked for the fewest knots reproducing it within a pixel
tolerance, and the result is scored against the keys the artist actually set. This is the
cheapest honest read on the half of the problem that is not solved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.ir import opacity_at, sample                       # noqa: E402
from roto.keys import f1, select                             # noqa: E402
from roto.sfx.json_ir import from_json_ir                    # noqa: E402


def shapes_of(doc):
    for ancestors, shape in doc.shapes():
        yield shape


def evaluate(element_dir: Path, tol_px: float, match_tol: int) -> dict:
    meta = json.loads((element_dir / 'meta.json').read_text())
    doc = from_json_ir(json.loads((element_dir / "target_ir.json").read_text()))
    px = meta['crop']['px_per_norm']
    frames = np.asarray(meta['frames']['index'], dtype=np.int32)

    rows = []
    for shape in shapes_of(doc):
        live = np.array([f for f in frames if opacity_at(shape, f) > 0.5], dtype=np.int32)
        if len(live) < 2:
            continue
        track = np.stack([np.asarray(sample(shape.path, f), dtype=np.float64)[:, 0, :]
                          for f in live]) * px
        truth = np.array(sorted({int(np.clip(k.frame, live[0], live[-1]))
                                 for k in shape.path}), dtype=np.int32)
        sel = select(track, live, tol_px)
        p, r, s = f1(sel.frames, truth, match_tol)
        rows.append({'n_live': len(live), 'n_truth': len(truth), 'n_pred': len(sel),
                     'precision': p, 'recall': r, 'f1': s, 'max_error_px': sel.max_error,
                     'exact': sel.exact})
    if not rows:
        return {}
    w = np.array([r['n_truth'] for r in rows], dtype=np.float64)
    w = w / w.sum()
    return {
        'element': element_dir.name,
        'shapes': len(rows),
        'artist_keys': int(sum(r['n_truth'] for r in rows)),
        'predicted_keys': int(sum(r['n_pred'] for r in rows)),
        'key_ratio': sum(r['n_pred'] for r in rows) / max(1, sum(r['n_truth'] for r in rows)),
        'precision': float(np.dot(w, [r['precision'] for r in rows])),
        'recall': float(np.dot(w, [r['recall'] for r in rows])),
        'f1': float(np.dot(w, [r['f1'] for r in rows])),
        'max_error_px': float(max(r['max_error_px'] for r in rows)),
        'all_exact': bool(all(r['exact'] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset', default='datasets/v001', nargs='?')
    ap.add_argument('--tol', type=float, action='append', default=[])
    ap.add_argument('--match-tol', type=int, default=1)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()
    tols = args.tol or [0.5, 1.0, 2.0, 4.0]

    dirs = sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists())
    out = []
    for tol in tols:
        rows = [r for r in (evaluate(d, tol, args.match_tol) for d in dirs) if r]
        tk = sum(r['artist_keys'] for r in rows)
        pk = sum(r['predicted_keys'] for r in rows)
        agg = {
            'tol_px': tol, 'elements': len(rows), 'artist_keys': tk, 'predicted_keys': pk,
            'key_ratio': pk / max(1, tk),
            'precision': float(np.average([r['precision'] for r in rows],
                                          weights=[r['artist_keys'] for r in rows])),
            'recall': float(np.average([r['recall'] for r in rows],
                                       weights=[r['artist_keys'] for r in rows])),
            'f1': float(np.average([r['f1'] for r in rows],
                                   weights=[r['artist_keys'] for r in rows])),
            'per_element': rows,
        }
        out.append(agg)
        print(f'tol {tol:>4.1f}px   keys {pk:>6} vs {tk:>6} artist (x{agg["key_ratio"]:.2f})   '
              f'P {agg["precision"]:.3f}  R {agg["recall"]:.3f}  F1 {agg["f1"]:.3f}')
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=2))
        print(f'\nreport: {args.report}')


if __name__ == '__main__':
    main()
