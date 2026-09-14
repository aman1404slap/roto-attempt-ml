"""What S2's two new predictions are worth, measured before any of S2 is built.

S1 was closed by a number that could have been measured first: a constant function passed its
acceptance gate. That happened because the gate was written against a metric nobody had
priced against the things requiring no skill. This script prices S2's gate the same way,
*before* the design note commits to it.

Charter S4's S2 asks the model to invent two things it is currently handed -- **lifespans**
(which frames a shape is on screen for) and **point counts** (how many control points it has)
-- and grades the result on charter S5's render gate: *within 0.01 of S1*. So the question
this script answers is not "can a head predict these", it is:

    **Is the render gate sensitive to these two quantities at all?**

If getting a lifespan trivially wrong costs less than 0.01 soft IoU, then S2's gate is passed
by a rule that does no work, and it measures nothing -- exactly S1's failure, found before the
run instead of after it.

Two families, both computed on the **artist's own** IR so the model is not in the picture:

*Structure statistics* (no render). What the point-count and lifespan labels actually look
like across ``datasets/v003``: the distributions, and what the trivial predictors of each
score. The point-count ones carry a caveat this script cannot measure away -- shape identity
is still given at S2, and the query table is per ``(element, shape)``, so a point count that
is constant across frames is memorisable exactly. The honest baseline for point count is
therefore 100%, and the number that matters is the render column below.

*Render sensitivity* (renders every element-frame, several times). The artist's own shapes,
re-rendered under a deliberately wrong structure, scored against the stored alpha with the
dataset's own conventions:

* ``exact`` -- untouched. **Must be 1.000000**, or nothing below it means anything. This is
  the ledger's artist-self-score row, repeated here so the variants have a proven zero.
* ``always_alive`` -- every shape on screen on every frame. The degenerate lifespan answer.
* ``coverage_alive`` -- every shape alive exactly when *any* shape of the element is. The
  smarter trivial rule: it needs the element's own on-screen span and nothing per shape.
* ``drop_1`` / ``drop_20pct`` -- point count reduced by one point, and by a fifth, dropping
  points at even spacing and leaving the survivors where the artist put them.

The two drop rows are a **floor, not a prediction**. A model that genuinely chose a smaller
point count would place those points well; decimation places them where a larger curve wanted
them. So the drop rows say what point count is worth when nothing compensates, which is the
pessimistic end of the range S2's decode sits in.

    python scripts/exp_s2_baselines.py --dataset datasets/v003
    python scripts/exp_s2_baselines.py --dataset datasets/v003 --stats-only   # no renders
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.ir import RotoDoc, opacity_at                                # noqa: E402
from roto.metrics import soft_iou                                      # noqa: E402
from roto.render.raster import config_from_meta, render_union          # noqa: E402
from roto.v2.build import element_dirs, load_alpha                     # noqa: E402
from roto.v2.traindata import load_element                             # noqa: E402


# ---------------------------------------------------------------------------- structure


def structure_stats(el) -> dict:
    """Point counts, coordinate widths and lifespans for one element. No render."""
    n_points = el.n_points_per_shape.astype(int)
    coords = el.coords_per_shape.astype(int)
    closed = el.closed_per_shape
    live = el.live                                     # (F, S) bool
    F, S = live.shape

    # How often a shape's on-screen state *changes*. A shape that is alive for its whole
    # track is a lifespan the head gets right by saying nothing; the work is in the others.
    transitions = int((live[1:] != live[:-1]).sum())
    always = int(live.all(axis=0).sum())
    never = int((~live.any(axis=0)).sum())
    # The element's own on-screen span: is *any* shape alive. This is what `coverage_alive`
    # below predicts with, and the gap between it and `live` is all a per-shape head can win.
    any_live = live.any(axis=1)                        # (F,)
    coverage_rule = np.broadcast_to(any_live[:, None], live.shape)
    agree = int((coverage_rule == live).sum())

    return {
        'element_id': el.element_id,
        'in_train': bool(el.in_train),
        'frames': int(F),
        'shapes': int(S),
        'points_min': int(n_points.min()), 'points_max': int(n_points.max()),
        'points_median': float(np.median(n_points)),
        'points_distinct': int(len(set(n_points.tolist()))),
        'points_mode': int(Counter(n_points.tolist()).most_common(1)[0][0]),
        'bezier_shapes': int((coords == 3).sum()),
        'closed_shapes': int(closed.sum()),
        'live_cells': int(live.sum()), 'cells': int(F * S),
        'live_rate': float(live.mean()),
        'shapes_always_live': always,
        'shapes_never_live': never,
        'lifespan_transitions': transitions,
        'coverage_rule_agreement': agree / (F * S),
        'frames_any_live': int(any_live.sum()),
        'n_points': n_points.tolist(),
    }


def point_count_baselines(rows: list[dict]) -> dict:
    """What the trivial point-count predictors score, and why one of them is not a baseline.

    Three predictors, none of which looks at the picture:

    * **global mode** -- one number for the whole dataset.
    * **per-element mode** -- one number per element. Available to the model for free: shape
      count and identity are still given at S2, so the query knows which element it is in.
    * **exact memorisation** -- the per ``(element, shape)`` query row. Point count does not
      vary across frames, so this row can carry it exactly and the picture need not be
      consulted. That is 100% by construction, which is why accuracy is not S2's metric.
    """
    every = [p for r in rows for p in r['n_points']]
    g_mode = Counter(every).most_common(1)[0][0]
    g_acc = sum(p == g_mode for p in every) / len(every)
    g_mae = float(np.mean([abs(p - g_mode) for p in every]))

    e_hits = e_abs = 0
    for r in rows:
        mode = Counter(r['n_points']).most_common(1)[0][0]
        e_hits += sum(p == mode for p in r['n_points'])
        e_abs += sum(abs(p - mode) for p in r['n_points'])
    return {
        'shapes': len(every),
        'distinct_counts': len(set(every)),
        'range': [min(every), max(every)],
        'median': float(np.median(every)),
        'histogram': dict(sorted(Counter(every).items())),
        'global_mode': {'value': int(g_mode), 'accuracy': g_acc, 'mae': g_mae},
        'per_element_mode': {'accuracy': e_hits / len(every), 'mae': e_abs / len(every)},
        'exact_memorisation': {
            'accuracy': 1.0,
            'note': 'per (element, shape) query row; point count is constant across frames, '
                    'so this needs no information from the alpha'},
    }


def lifespan_baselines(rows: list[dict]) -> dict:
    """What the trivial lifespan predictors score. Unlike point count, these are real.

    A lifespan varies frame to frame and the query row does not, so it cannot be memorised
    the way a point count can -- the head has to read the alpha. What it *can* be is
    unnecessary: if shapes are nearly always alive, or alive exactly when the element is on
    screen, then the two trivial rules below already answer it.
    """
    cells = sum(r['cells'] for r in rows)
    live = sum(r['live_cells'] for r in rows)
    tp = live                                   # always-alive: every live cell is a hit
    fp = cells - live                           # ...and every dead cell a false positive
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * 1.0 / max(1e-9, prec + 1.0)      # recall is exactly 1 by construction
    return {
        'cells': cells, 'live_cells': live, 'live_rate': live / cells,
        'live_rate_per_element': {r['element_id']: r['live_rate'] for r in rows},
        'live_rate_spread': [min(r['live_rate'] for r in rows),
                             max(r['live_rate'] for r in rows)],
        'shapes_always_live': sum(r['shapes_always_live'] for r in rows),
        'shapes_total': sum(r['shapes'] for r in rows),
        'transitions': sum(r['lifespan_transitions'] for r in rows),
        'always_alive': {'accuracy': live / cells, 'precision': prec, 'recall': 1.0,
                         'f1': f1},
        'coverage_rule': {
            'accuracy': sum(r['coverage_rule_agreement'] * r['cells'] for r in rows) / cells,
            'note': 'alive iff any shape of the element is alive; needs no per-shape signal'},
    }


# ---------------------------------------------------------------------------- renders


def strip_opacity(doc: RotoDoc) -> RotoDoc:
    """Every shape alive on every frame: drop the opacity track (``None`` means 1.0)."""
    out = deepcopy(doc)
    for _, s in out.shapes():
        s.opacity = None
    return out


def opacity_from_mask(doc: RotoDoc, el, mask: np.ndarray) -> RotoDoc:
    """Rewrite every shape's lifespan to ``mask`` ``(F, S)``.

    Written as a per-frame opacity track with hold interpolation, which is what the IR uses
    for a lifespan, so the render path is the ordinary one rather than a special case.
    """
    from roto.ir import Key
    out = deepcopy(doc)
    for si, (_, s) in enumerate(out.shapes()):
        s.opacity = [Key(int(f), 'hold', np.array([[100.0 if mask[i, si] else 0.0]]))
                     for i, f in enumerate(el.frames)]
    return out


def coverage_opacity(doc: RotoDoc, el) -> RotoDoc:
    """Every shape alive exactly when *any* shape of the element is."""
    return opacity_from_mask(doc, el, np.broadcast_to(el.live.any(axis=1)[:, None],
                                                      el.live.shape))


def noisy_lifespan(doc: RotoDoc, el, rate: float, direction: str, seed: int = 0) -> RotoDoc:
    """The artist's lifespans with ``rate`` of **all** cells flipped, in one direction.

    The rate is a fraction of every ``(frame, shape)`` cell, not of the ones eligible to
    flip, so a variant at 0.01 is a lifespan head at exactly **99% cell accuracy** and the
    render cost reads straight off as "what one point of lifespan accuracy is worth".

    ``direction`` separates the two mistakes, because a layer's matte is a **union** and they
    are not symmetric. A shape drawn where the artist drew none adds area nothing covers. A
    shape omitted from a pile of overlapping ones may cost nothing at all, because its
    neighbours already fill those pixels. Which of the two is expensive is what sets the
    decode threshold, and guessing it would be guessing the thing S2 turns on.

    Flips are independent per cell, so they arrive as one-frame flicker -- which is precisely
    the failure mode plan section 5's hysteresis exists to suppress. This measures what the
    hysteresis has to be worth.
    """
    rng = np.random.default_rng(seed)
    live = el.live.copy()
    n = int(round(rate * live.size))
    pool = np.flatnonzero(~live.ravel() if direction == 'fp' else live.ravel())
    if len(pool):
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
        flat = live.ravel().copy()
        flat[pick] = (direction == 'fp')
        live = flat.reshape(live.shape)
    return opacity_from_mask(doc, el, live)


def edge_lifespan(doc: RotoDoc, el, direction: str) -> RotoDoc:
    """Every lifespan boundary moved one frame, in one direction.

    The realistic mistake, where the flicker above is the pathological one. A per-frame alive
    logit that has learned *which* shapes come and go, and is a frame late or a frame early on
    when, produces exactly this -- and there are 509 on/off transitions in this dataset, so
    "off by one everywhere" is 1.3% of all cells. Whether that fits inside charter S5's
    0.01-wide render gate is the question S2's decode has to answer, and it is answerable now.

    ``dilate`` turns the frame either side of every live run on (pure false positive);
    ``erode`` turns the first and last frame of every live run off (pure false negative).
    """
    live = el.live
    if direction == 'dilate':
        out = live.copy()
        out[1:] |= live[:-1]
        out[:-1] |= live[1:]
    else:
        out = live.copy()
        out[1:] &= live[:-1]
        out[:-1] &= live[1:]
        out &= live
    return opacity_from_mask(doc, el, out)


def decimate(doc: RotoDoc, keep_fn) -> RotoDoc:
    """Drop control points from every shape, keeping ``keep_fn(P)`` of them, evenly spaced.

    The survivors stay exactly where the artist put them. That is what makes this a floor
    rather than a prediction: a model choosing a smaller count would move its points to
    compensate, and this does not.
    """
    out = deepcopy(doc)
    for _, s in out.shapes():
        P = s.n_points
        k = max(3, int(keep_fn(P)))
        if k >= P:
            continue
        idx = np.unique(np.round(np.linspace(0, P - 1, k)).astype(int))
        for key in s.path:
            key.value = np.ascontiguousarray(np.asarray(key.value)[idx])
    return out


def score_variant(el, doc: RotoDoc) -> np.ndarray:
    """Soft IoU per frame of ``doc`` against the element's stored alpha.

    Render config comes from the dataset's own record, exactly as ``reconstruct.score_doc``
    does -- a freshly built one reasserts class defaults for conventions this dataset
    measured, and scores the artist's own shapes below 1.0 for it.
    """
    c = el.crop
    cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, None)
    out = np.empty(len(el.frames), float)
    for i, f in enumerate(el.frames):
        x0, y0 = c['offsets'][int(f)]
        box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
        pred = render_union(doc, int(f), cfg, c['scale'], box)
        truth = load_alpha(el.directory, int(f))
        if pred.shape != truth.shape:
            pred = pred[:truth.shape[0], :truth.shape[1]]
        out[i] = soft_iou(pred, truth)
    return out


VARIANTS = ('exact', 'always_alive', 'coverage_alive', 'drop_1', 'drop_20pct')

NOISE_RATES = (0.01, 0.03, 0.10)
NOISE_VARIANTS = tuple(f'{d}_{int(r * 100):02d}pct'
                       for d in ('fp', 'fn') for r in NOISE_RATES) + (
    'edge_dilate_1', 'edge_erode_1')
"""A lifespan head at 99%, 97% and 90% cell accuracy, wrong in one direction at a time.

These are the rows that turn charter S5's ``render within 0.01 of S1`` from a sentence into a
requirement on the head: whichever rate first costs 0.01 soft IoU is the accuracy S2 has to
reach before it can pass, and it is written down before the head exists."""


def render_sensitivity(el, doc: RotoDoc, which=VARIANTS) -> dict:
    """Every variant of one element, scored on all frames and on on-screen frames only.

    Both readings, because Step 2c established that an average over frames where the layer
    is absent hands out free 1.0s in proportion to how often it is absent -- and the
    lifespan variants are precisely the ones that draw *something* on those frames.
    """
    docs = {
        'exact': doc,
        'always_alive': strip_opacity(doc),
        'coverage_alive': coverage_opacity(doc, el),
        'drop_1': decimate(doc, lambda p: p - 1),
        'drop_20pct': decimate(doc, lambda p: round(0.8 * p)),
        **{n: None for n in NOISE_VARIANTS},
    }
    for d in ('fp', 'fn'):
        for r in NOISE_RATES:
            name = f'{d}_{int(r * 100):02d}pct'
            if name in which:
                docs[name] = noisy_lifespan(doc, el, r, d)
    for name, d in (('edge_dilate_1', 'dilate'), ('edge_erode_1', 'erode')):
        if name in which:
            docs[name] = edge_lifespan(doc, el, d)
    on = el.live.any(axis=1)
    res = {}
    for name in which:
        s = score_variant(el, docs[name])
        res[name] = {
            'mean': float(s.mean()),
            'mean_onscreen': float(s[on].mean()) if on.any() else None,
            'min': float(s.min()),
            'frames_below_0.90': int((s < 0.90).sum()),
        }
    return res


# ---------------------------------------------------------------------------- driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--out', default='runs/v2/s2_baselines.json')
    ap.add_argument('--stats-only', action='store_true',
                    help='structure statistics only; skip the renders')
    ap.add_argument('--noise', action='store_true',
                    help='also render the lifespan-accuracy rows (fp/fn at 1%%, 3%%, 10%%)')
    ap.add_argument('--progress', default=None,
                    help='write a one-line progress file here as elements complete')
    args = ap.parse_args()

    which = VARIANTS + (NOISE_VARIANTS if args.noise else ())
    dirs = list(element_dirs(args.dataset))
    rows, sens = [], {}
    t0 = time.time()
    for i, d in enumerate(dirs):
        el = load_element(d, with_local=False)
        rows.append(structure_stats(el))
        if not args.stats_only:
            doc = _load_doc(d)
            sens[el.element_id] = render_sensitivity(el, doc, which)
        msg = (f'[{i + 1}/{len(dirs)}] {el.element_id}  '
               f'{rows[-1]["shapes"]} shapes  live {rows[-1]["live_rate"]:.3f}  '
               f'[{time.time() - t0:.0f}s]')
        print(msg, flush=True)
        if args.progress:
            Path(args.progress).write_text(msg + '\n')

    trained = [r for r in rows if r['in_train']]
    out = {
        'dataset': args.dataset,
        'elements': len(rows), 'trained_elements': len(trained),
        'point_count': point_count_baselines(trained),
        'point_count_all': point_count_baselines(rows),
        'lifespan': lifespan_baselines(trained),
        'lifespan_all': lifespan_baselines(rows),
        'per_element': rows,
        'render_sensitivity': sens or None,
        'wall_clock_s': round(time.time() - t0, 1),
    }
    if sens:
        out['render_sensitivity_trained'] = _aggregate(sens, trained, which)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    _print_report(out)
    print(f'\nwrote {args.out}  ({out["wall_clock_s"]:.0f}s)')


def _load_doc(d: Path) -> RotoDoc:
    from roto.sfx.json_ir import from_json_ir
    return from_json_ir(json.loads((Path(d) / 'target_ir.json').read_text()))


def _aggregate(sens: dict, trained: list[dict], which=VARIANTS) -> dict:
    """Frame-weighted means over the trained elements, plus the per-element floor.

    Charter L3: worst case beside the mean, never the mean alone.
    """
    ids = [r['element_id'] for r in trained]
    w = np.array([r['frames'] for r in trained], float)
    out = {}
    for name in which:
        vals = np.array([sens[i][name]['mean'] for i in ids])
        on = np.array([sens[i][name]['mean_onscreen'] or sens[i][name]['mean'] for i in ids])
        worst = int(np.argmin(on))
        out[name] = {
            'mean': float(np.average(vals, weights=w)),
            'mean_onscreen': float(np.average(on, weights=w)),
            'worst_element': ids[worst], 'worst_element_onscreen': float(on[worst]),
            'frames_below_0.90': int(sum(sens[i][name]['frames_below_0.90'] for i in ids)),
        }
    base = out['exact']['mean_onscreen']
    for name in which:
        out[name]['cost_vs_exact_onscreen'] = base - out[name]['mean_onscreen']
    return out


def _print_report(out: dict) -> None:
    pc, ls = out['point_count'], out['lifespan']
    print(f'\n{out["dataset"]}: {out["trained_elements"]} trained elements '
          f'of {out["elements"]}\n')
    print('POINT COUNT -- what the model is asked to invent at S2')
    print(f'  {pc["shapes"]} shapes, {pc["distinct_counts"]} distinct counts, '
          f'range {pc["range"][0]}-{pc["range"][1]}, median {pc["median"]:.0f}')
    print(f'  global mode {pc["global_mode"]["value"]:>3}      '
          f'accuracy {pc["global_mode"]["accuracy"]:.3f}  mae {pc["global_mode"]["mae"]:.2f}')
    print(f'  per-element mode       accuracy '
          f'{pc["per_element_mode"]["accuracy"]:.3f}  mae {pc["per_element_mode"]["mae"]:.2f}')
    print('  exact memorisation     accuracy 1.000  '
          '<- the query row can carry it; accuracy is not a metric here')
    print('\nLIFESPAN')
    print(f'  {ls["live_cells"]}/{ls["cells"]} cells live ({ls["live_rate"]:.3f}), '
          f'per element {ls["live_rate_spread"][0]:.3f}-{ls["live_rate_spread"][1]:.3f}')
    print(f'  {ls["shapes_always_live"]}/{ls["shapes_total"]} shapes are alive on every '
          f'frame; {ls["transitions"]} on/off transitions in the dataset')
    print(f'  "always alive"         accuracy {ls["always_alive"]["accuracy"]:.3f}  '
          f'F1 {ls["always_alive"]["f1"]:.3f}')
    print(f'  "alive iff element is" accuracy {ls["coverage_rule"]["accuracy"]:.3f}')
    rs = out.get('render_sensitivity_trained')
    if rs:
        print('\nRENDER SENSITIVITY -- artist shapes, structure deliberately wrong')
        print(f'  {"variant":<16}{"on-screen":>10}{"cost":>9}{"worst el":>10}{"<0.90":>8}')
        for name in rs:
            r = rs[name]
            print(f'  {name:<16}{r["mean_onscreen"]:>10.6f}'
                  f'{r["cost_vs_exact_onscreen"]:>9.4f}'
                  f'{r["worst_element_onscreen"]:>10.4f}{r["frames_below_0.90"]:>8}')
        print('  cost is soft IoU lost against the artist\'s own shapes. S2\'s gate is '
              '0.01 wide.')


if __name__ == '__main__':
    main()
