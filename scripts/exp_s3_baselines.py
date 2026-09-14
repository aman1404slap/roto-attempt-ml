"""What charter S4's S3 gate is worth, priced before the S3 design note is written.

S1 was closed by a gate a constant function could clear, discovered after the run. S2's gate
was priced first and held. This script does the same for S3, and it asks one question above
all others:

    **Charter S4 sets S3's gate at "soft IoU >= 0.95 on held-out layers".
    Can a single shape traced around the silhouette clear it?**

If it can, the gate does not test what S3 is for. S3 is the stage where the model invents the
*breakdown* -- how many shapes, which is which -- and soft IoU is **decomposition-blind by
construction**. That blindness is charter L2 working as intended when it stops us punishing a
valid alternative breakdown; it is a hole when it is the only thing in the gate. One shape
hugging the union outline is not roto, cannot be edited the way an artist needs, and would
score whatever the union scores.

Three families:

*Silhouette traces* (renders). Take the artist's own alpha, find its outline, resample it to a
plausible control-point count, rebuild it as real IR shapes and render it back. No model, no
decomposition, no understanding -- just the answer traced off the question.

* ``trace_1``  -- one closed shape around the largest blob
* ``trace_k``  -- one shape per connected blob, which is the smartest trivial answer available
* each at several point counts, because "trace it with 64 points" and "trace it with 16" are
  different claims about how cheaply the gate falls

*Burial* (renders). ``exp_s2_visibility.py`` measured how exposed a shape is at its lifespan
**boundaries** and found 61.4% invisible. S3 needs the same question over **every live cell**:
if the median shape contributes almost nothing to the picture, then any loss that matches
predicted curves to the artist's curves is scoring handwriting the picture cannot show, and
charter L2's "grade the render" is not a preference but the only option.

*Style statistics* (no renders). The distributions plan section 6 wants as distribution
penalties: shapes per element, points per shape, keys per live frame. Reported so the S3 note
can set them from the archive rather than from taste.

    python scripts/exp_s3_baselines.py --dataset datasets/v003
    python scripts/exp_s3_baselines.py --dataset datasets/v003 --skip-burial
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.geometry import crop_to_local                                # noqa: E402
from roto.ir import ADD, BSPLINE, Key, Layer, RotoDoc, Shape           # noqa: E402
from roto.metrics import soft_iou                                      # noqa: E402
from roto.render.raster import config_from_meta, render_union          # noqa: E402
from roto.sfx.json_ir import from_json_ir                              # noqa: E402
from roto.v2.build import element_dirs, load_alpha                     # noqa: E402
from roto.v2.traindata import load_element                             # noqa: E402

POINTS = (16, 32, 64)
"""Control points per traced shape. v003's artist shapes run 4-68, median 18, so 16 is an
ordinary artist count and 64 is at the top of what the archive ever uses -- the range says
whether the gate falls to a *plausible* trivial answer or only to an implausibly dense one."""

MIN_AREA_PX = 8.0
"""Contours smaller than this are noise in the alpha's anti-aliased edge, not blobs."""


def trace(alpha: np.ndarray, n_points: int, largest_only: bool) -> list[np.ndarray]:
    """Outline the silhouette as one or more closed rings of ``n_points`` crop-space points.

    ``cv2.findContours`` on the thresholded alpha, outer contours only -- a hole in the matte
    is not something a single traced ring can express, and pretending otherwise would flatter
    the baseline. Each contour is then resampled at uniform arc length, which is roughly how
    an artist distributes points and keeps the comparison fair.
    """
    mask = (alpha > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c[:, 0, :].astype(np.float64) for c in cnts
            if cv2.contourArea(c) >= MIN_AREA_PX and len(c) >= 4]
    if not cnts:
        return []
    if largest_only:
        cnts = [max(cnts, key=lambda c: cv2.contourArea(c.astype(np.float32)))]
    out = []
    for c in cnts:
        closed = np.vstack([c, c[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0:
            continue
        want = np.linspace(0.0, s[-1], n_points, endpoint=False)
        out.append(np.stack([np.interp(want, s, closed[:, 0]),
                             np.interp(want, s, closed[:, 1])], axis=-1))
    return out


def trace_doc(el, frame: int, rings: list[np.ndarray]) -> RotoDoc:
    """Traced crop-pixel rings -> a real IR document the project's own renderer can draw.

    Built as genuine ``Shape`` objects under one root rather than rasterised directly, so the
    number this produces comes off exactly the path every other number in the project comes
    off -- same renderer, same conventions, same scoring.
    """
    c = el.crop
    eye = np.eye(4)
    shapes = []
    for i, ring in enumerate(rings):
        local = crop_to_local(ring / c['out_px'], eye, width=c['width'], height=c['height'],
                              offset=c['offsets'][int(frame)], scale=c['scale'],
                              out_px=c['out_px'])
        shapes.append(Shape(name=f'trace_{i}', shape_type=BSPLINE, closed=True,
                            path=[Key(int(frame), 'linear',
                                      local.reshape(-1, 1, 2).astype(np.float64))]))
    return RotoDoc(c['width'], c['height'], len(el.frames),
                   [Layer(name='trace', children=shapes, blend=ADD)])


def score_trace(el, n_points: int, largest_only: bool) -> tuple[np.ndarray, list[int]]:
    """Soft IoU per frame for the traced answer, plus how many shapes it used."""
    c = el.crop
    cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, None)
    scores, counts = [], []
    for f in el.frames:
        truth = load_alpha(el.directory, int(f))
        rings = trace(truth, n_points, largest_only)
        counts.append(len(rings))
        if not rings:
            # Nothing on screen: drawing nothing is exactly right, and soft_iou says so.
            scores.append(1.0 if truth.max() <= 0.5 else 0.0)
            continue
        x0, y0 = c['offsets'][int(f)]
        box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
        pred = render_union(trace_doc(el, int(f), rings), int(f), cfg, c['scale'], box)
        if pred.shape != truth.shape:
            pred = pred[:truth.shape[0], :truth.shape[1]]
        scores.append(soft_iou(pred, truth))
    return np.asarray(scores), counts


def burial(el, doc: RotoDoc, max_cells: int = 400, seed: int = 0) -> np.ndarray:
    """How much of each shape's own ink nothing else draws, over live cells generally.

    ``exp_s2_visibility.py`` asks this at lifespan boundaries. S3 needs it everywhere: the
    question there is not "can the model see this shape appear" but "can it see this shape at
    all", which is what decides whether matching predicted curves to the artist's curves is
    scoring something the picture contains.

    Sampled rather than exhaustive -- 25,137 live cells at two renders each is not worth the
    wall clock when a few hundred settle the distribution.
    """
    c = el.crop
    cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, None)
    shapes = [s for _, s in doc.shapes()]
    rng = np.random.default_rng(seed)
    cells = np.argwhere(el.live)
    if not len(cells):
        return np.zeros(0)
    if len(cells) > max_cells:
        cells = cells[rng.choice(len(cells), max_cells, replace=False)]

    def draw(frame: int) -> np.ndarray:
        x0, y0 = c['offsets'][int(frame)]
        box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
        return render_union(doc, int(frame), cfg, c['scale'], box)

    off = np.array([[0.0]])
    on = np.array([[100.0]])
    cache, out = {}, []
    for fi, si in cells:
        frame = int(el.frames[fi])
        shape = shapes[si]
        saved = shape.opacity
        if frame not in cache:
            cache[frame] = draw(frame)
        a = cache[frame]
        shape.opacity = [Key(frame, 'hold', off)]
        b = draw(frame)
        others = [(s, s.opacity) for s in shapes if s is not shape]
        for s, _ in others:
            s.opacity = [Key(frame, 'hold', off)]
        shape.opacity = [Key(frame, 'hold', on)]
        own = draw(frame)
        shape.opacity = saved
        for s, o in others:
            s.opacity = o
        ink = float(own.sum())
        out.append(float(np.clip(a - b, 0, None).sum()) / ink if ink > 0 else 0.0)
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--out', default='runs/v2/s3_baselines.json')
    ap.add_argument('--skip-burial', action='store_true')
    ap.add_argument('--progress', default=None)
    args = ap.parse_args()

    rows, t0 = [], time.time()
    for i, d in enumerate(element_dirs(args.dataset)):
        el = load_element(d, with_local=False)
        meta = json.loads((Path(d) / 'meta.json').read_text())
        row = {
            'element_id': el.element_id, 'in_train': bool(el.in_train),
            'frames': int(len(el.frames)), 'shapes': int(el.n_shapes),
            'points_per_shape': el.n_points_per_shape.tolist(),
            'keys_per_live_frame': meta.get('tags', {}).get('keys_per_live_frame'),
            'on': int(el.live.any(axis=1).sum()),
        }
        on = el.live.any(axis=1)
        for n in POINTS:
            for tag, largest in (('trace_1', True), ('trace_k', False)):
                s, counts = score_trace(el, n, largest)
                row[f'{tag}_{n}'] = {
                    'mean': float(s.mean()),
                    'mean_onscreen': float(s[on].mean()) if on.any() else None,
                    'min': float(s.min()),
                    'shapes_used': float(np.mean([c for c, o in zip(counts, on) if o])
                                         if on.any() else 0.0),
                }
        if not args.skip_burial:
            doc = from_json_ir(json.loads((Path(d) / 'target_ir.json').read_text()))
            b = burial(el, doc)
            row['burial'] = {
                'cells': int(len(b)),
                'median': float(np.median(b)) if len(b) else None,
                'buried_lt_0.05': float((b < 0.05).mean()) if len(b) else None,
                'exposed_gt_0.80': float((b > 0.80).mean()) if len(b) else None,
                'samples': [round(float(x), 5) for x in b],
            }
        rows.append(row)
        msg = (f'[{i + 1}] {el.element_id}  trace_k@32 on-screen '
               f'{row["trace_k_32"]["mean_onscreen"]:.4f}  [{time.time() - t0:.0f}s]')
        print(msg, flush=True)
        if args.progress:
            Path(args.progress).write_text(msg + '\n')

    tr = [r for r in rows if r['in_train']]
    held = [r for r in rows if not r['in_train']]
    w = np.array([r['on'] for r in tr], float)

    def agg(group, key, n):
        g = [r for r in group if r[f'{key}_{n}']['mean_onscreen'] is not None]
        if not g:
            return None
        gw = np.array([r['on'] for r in g], float)
        v = np.array([r[f'{key}_{n}']['mean_onscreen'] for r in g])
        return {'mean_onscreen': float(np.average(v, weights=gw)),
                'worst_element': float(v.min()),
                'shapes_used': float(np.average(
                    [r[f'{key}_{n}']['shapes_used'] for r in g], weights=gw))}

    every_burial = np.concatenate([np.asarray(r['burial']['samples'])
                                   for r in tr if r.get('burial')]) if not args.skip_burial \
        else np.zeros(0)
    pts = [p for r in tr for p in r['points_per_shape']]
    out = {
        'dataset': args.dataset, 'elements': len(rows), 'trained': len(tr),
        'gate': 0.95,
        'trace': {f'{k}_{n}': {'trained': agg(tr, k, n), 'held_out_layers': agg(held, k, n)}
                  for k in ('trace_1', 'trace_k') for n in POINTS},
        'style': {
            'shapes_per_element': sorted(r['shapes'] for r in tr),
            'points_per_shape': {'min': int(min(pts)), 'median': float(np.median(pts)),
                                 'max': int(max(pts)), 'mean': float(np.mean(pts))},
            'keys_per_live_frame': [r['keys_per_live_frame'] for r in tr
                                    if r['keys_per_live_frame'] is not None],
        },
        'burial': ({'cells': int(len(every_burial)),
                    'median': float(np.median(every_burial)),
                    'quantiles': {q: float(np.quantile(every_burial, q / 100))
                                  for q in (5, 25, 50, 75, 95)},
                    'buried_lt_0.05': float((every_burial < 0.05).mean()),
                    'exposed_gt_0.80': float((every_burial > 0.80).mean())}
                   if len(every_burial) else None),
        'per_element': rows, 'wall_clock_s': round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f'\nSILHOUETTE TRACE vs charter S4\'s S3 gate (soft IoU >= 0.95)')
    print(f'  {"baseline":<14}{"shapes":>8}{"trained":>10}{"worst el":>10}{"held-out":>10}')
    for k in ('trace_1', 'trace_k'):
        for n in POINTS:
            a = out['trace'][f'{k}_{n}']['trained']
            h = out['trace'][f'{k}_{n}']['held_out_layers']
            if not a:
                continue
            print(f'  {k}@{n:<10}{a["shapes_used"]:>8.1f}{a["mean_onscreen"]:>10.4f}'
                  f'{a["worst_element"]:>10.4f}'
                  f'{(h["mean_onscreen"] if h else float("nan")):>10.4f}')
    print('  on-screen frames only. "shapes" is how many the trace used against the artist\'s')
    print(f'  {min(r["shapes"] for r in tr)}-{max(r["shapes"] for r in tr)}.')
    if out['burial']:
        b = out['burial']
        print(f'\nBURIAL over {b["cells"]} sampled live cells (not just boundaries)')
        q = b['quantiles']
        print(f'  p5 {q[5]:.3f}  p25 {q[25]:.3f}  median {q[50]:.3f}  p75 {q[75]:.3f}  '
              f'p95 {q[95]:.3f}')
        print(f'  buried (< 0.05): {100 * b["buried_lt_0.05"]:.1f}%   '
              f'plainly visible (> 0.80): {100 * b["exposed_gt_0.80"]:.1f}%')
    st = out['style']
    print(f'\nSTYLE TARGETS for plan section 6')
    print(f'  shapes per element  {st["shapes_per_element"][0]}-'
          f'{st["shapes_per_element"][-1]}, median '
          f'{int(np.median(st["shapes_per_element"]))}')
    print(f'  points per shape    {st["points_per_shape"]["min"]}-'
          f'{st["points_per_shape"]["max"]}, median '
          f'{st["points_per_shape"]["median"]:.0f}')
    k = st['keys_per_live_frame']
    if k:
        print(f'  keys per live frame {min(k):.3f}-{max(k):.3f}, median {np.median(k):.3f}')
    print(f'\nwrote {args.out}  ({out["wall_clock_s"]:.0f}s)')


if __name__ == '__main__':
    main()
