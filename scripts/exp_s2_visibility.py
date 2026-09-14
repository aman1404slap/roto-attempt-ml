"""Can the lifespan head see the thing it is being asked to predict?

``v2-s2-design-note.md`` asserts a ceiling on the lifespan head and does not measure it: a
layer's matte is the **union** of its shapes, so a shape that goes dark inside a pile of
overlapping ones changes the picture by nothing at all, and no head reading that picture can
know it happened. This script measures how often that is true.

The design note also establishes that S2's render gate needs roughly **99% lifespan cell
accuracy**, with omissions costing 3.2x intrusions. If a large fraction of the transitions are
invisible in the input, then that accuracy is not merely hard, it is unreachable -- and S2b's
result would be a fact about the task rather than about the head. That distinction is worth
knowing before the run is interpreted, not after.

**What is measured.** For every on/off transition in the dataset, at the boundary frame where
the shape is live:

* ``own``  -- the shape rendered alone
* ``A``    -- the element's union, as the artist has it
* ``B``    -- the same union with this one shape switched off

Since the union is a per-pixel max, ``A - B`` is exactly the area only this shape covers.
So::

    visibility = sum(A - B) / sum(own)

is the fraction of the shape's own ink that nothing else is already drawing. **1.0** means the
shape is fully exposed and its appearance is plain in the alpha; **0.0** means it is entirely
buried and switching it off changes no pixel. No threshold is chosen: the distribution is
reported and the reader picks.

This is an *upper* bound on what a head could detect, and a generous one -- it says the
evidence exists in the frame, not that a 256px encoder can resolve it.

    python scripts/exp_s2_visibility.py --dataset datasets/v003
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.ir import Key, RotoDoc                                       # noqa: E402
from roto.render.raster import config_from_meta, render_union          # noqa: E402
from roto.v2.build import element_dirs                                 # noqa: E402
from roto.v2.traindata import load_element                             # noqa: E402
from roto.sfx.json_ir import from_json_ir                              # noqa: E402

OFF = np.array([[0.0]])
ON = np.array([[100.0]])


def transitions(live: np.ndarray) -> list[tuple[int, int]]:
    """``(frame_index, shape_index)`` at every **interior** lifespan boundary, live side.

    The live side because that is the frame where the shape is drawn, and the question is
    whether drawing it is visible.

    **Interior only.** The start and end of the track are not transitions: a shape that is
    live on frame 0 never appears, it is simply there, and there is no boundary to place. This
    matters for the count as well as the principle -- treating the ends as boundaries reports
    10 for ``ts_020355__Green`` where there are 2, and would dilute the visibility
    distribution with frames that have no boundary in them. It is also the definition behind
    the design note's 509, so the two numbers stay the same quantity.
    """
    out = set()
    F, S = live.shape
    for si in range(S):
        col = live[:, si]
        for f in range(F):
            if not col[f]:
                continue
            if f > 0 and not col[f - 1]:
                out.add((f, si))
            if f < F - 1 and not col[f + 1]:
                out.add((f, si))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--out', default='runs/v2/s2_visibility.json')
    ap.add_argument('--progress', default=None)
    args = ap.parse_args()

    rows, per_el, t0 = [], [], time.time()
    for i, d in enumerate(element_dirs(args.dataset)):
        el = load_element(d, with_local=False)
        doc = from_json_ir(json.loads((Path(d) / 'target_ir.json').read_text()))
        shapes = [s for _, s in doc.shapes()]
        c = el.crop
        cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, None)

        def draw(frame: int) -> np.ndarray:
            x0, y0 = c['offsets'][int(frame)]
            box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
            return render_union(doc, int(frame), cfg, c['scale'], box)

        vis, cache = [], {}
        for fi, si in transitions(el.live):
            frame = int(el.frames[fi])
            shape = shapes[si]
            saved = shape.opacity
            if frame not in cache:
                cache[frame] = draw(frame)
            a = cache[frame]
            shape.opacity = [Key(frame, 'hold', OFF)]        # this shape only, off
            b = draw(frame)
            # Everything else off, so what renders is this shape alone. Both edits are
            # restored before the next transition, so the document never drifts.
            others = [(s, s.opacity) for s in shapes if s is not shape]
            for s, _ in others:
                s.opacity = [Key(frame, 'hold', OFF)]
            shape.opacity = [Key(frame, 'hold', ON)]
            own = draw(frame)
            shape.opacity = saved
            for s, o in others:
                s.opacity = o
            own_ink = float(own.sum())
            unique = float(np.clip(a - b, 0, None).sum())
            vis.append(unique / own_ink if own_ink > 0 else 0.0)
        v = np.asarray(vis) if vis else np.zeros(0)
        rows.extend(v.tolist())
        per_el.append({
            'element_id': el.element_id, 'in_train': bool(el.in_train),
            'shapes': int(el.n_shapes), 'transitions': int(len(v)),
            'visibility_mean': float(v.mean()) if len(v) else None,
            'visibility_median': float(np.median(v)) if len(v) else None,
            'buried_frac': float((v < 0.05).mean()) if len(v) else None,
        })
        msg = (f'[{i + 1}] {el.element_id}  {len(v)} transitions  '
               f'median visibility '
               f'{np.median(v):.3f}' if len(v) else f'[{i + 1}] {el.element_id}  none')
        print(f'{msg}  [{time.time() - t0:.0f}s]', flush=True)
        if args.progress:
            Path(args.progress).write_text(msg + '\n')

    tr = [r for r in per_el if r['in_train'] and r['transitions']]
    allv = np.concatenate([np.zeros(0)] + [np.asarray(rows)]) if rows else np.zeros(0)
    out = {'dataset': args.dataset, 'transitions': int(len(allv)),
           'visibility_mean': float(allv.mean()) if len(allv) else None,
           'quantiles': {q: float(np.quantile(allv, q / 100)) for q in (5, 25, 50, 75, 95)}
           if len(allv) else None,
           'buried_lt_0.05': float((allv < 0.05).mean()) if len(allv) else None,
           'buried_lt_0.20': float((allv < 0.20).mean()) if len(allv) else None,
           'exposed_gt_0.80': float((allv > 0.80).mean()) if len(allv) else None,
           'per_element': per_el, 'wall_clock_s': round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f'\n{out["transitions"]} lifespan transitions across {len(per_el)} elements')
    if len(allv):
        q = out['quantiles']
        print(f'  visibility (fraction of the shape\'s own ink nothing else draws)')
        print(f'    p5 {q[5]:.3f}   p25 {q[25]:.3f}   median {q[50]:.3f}   '
              f'p75 {q[75]:.3f}   p95 {q[95]:.3f}')
        print(f'  buried (< 0.05, switching it off changes almost no pixel): '
              f'{100 * out["buried_lt_0.05"]:.1f}%')
        print(f'  mostly hidden (< 0.20): {100 * out["buried_lt_0.20"]:.1f}%')
        print(f'  plainly visible (> 0.80): {100 * out["exposed_gt_0.80"]:.1f}%')
        print('\n  A buried transition is one no head reading the union alpha can detect, so '
              'this\n  is a ceiling on lifespan accuracy -- and a generous one, since it says '
              'the\n  evidence is in the frame, not that a 256px encoder can resolve it.')
    print(f'\nwrote {args.out}  ({out["wall_clock_s"]:.0f}s)')


if __name__ == '__main__':
    main()
