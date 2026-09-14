"""Aggregating per-element rows into the run table charter S5 reads.

Ported from ``roto.model.report`` at v2 Step 2. Pure arithmetic over row dicts -- it touches
neither the network nor the dataset -- but it is v2's because the table's *shape* is frozen by
``v2_implementation_plan.md`` S2 and v2 adds a column the archive had no split for: held-out
**shots**, reported apart from held-out frames and held-out elements.

Charter L3 governs how these are read: worst case, never averages alone. Every aggregate here
is accompanied by its per-element floor and its frames-below-threshold count.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def run_totals(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-element summaries (``Reconstruction.summary()``) into one row.

    ``rows`` must be non-empty. Key F1 and the key ratio are weighted by the *artist's* key
    count rather than by frames, because they are statements about keys.
    """
    if not rows:
        raise ValueError('no elements to aggregate')
    w = np.array([r['frames'] for r in rows], float)
    kw = np.array([r['keys_artist'] for r in rows], float)
    av = lambda k, wt=w: float(np.average([r[k] for r in rows], weights=wt))

    worst_element = min(rows, key=lambda r: r['mean_soft_iou'])
    worst_frame = min(rows, key=lambda r: r['min_soft_iou'])
    tot = {
        'elements': len(rows), 'frames': int(w.sum()),
        'mean_soft_iou': av('mean_soft_iou'), 'mean_iou': av('mean_iou'),
        # The three worst-case columns the handover asks for by name.
        'worst_element_soft_iou': worst_element['mean_soft_iou'],
        'worst_element': worst_element['element_id'],
        'worst_frame_soft_iou': worst_frame['min_soft_iou'],
        'worst_frame': worst_frame['worst_frame'],
        'worst_frame_element': worst_frame['element_id'],
        'point_err_px': av('point_err_px'),
        'p95_point_err_px': av('p95_point_err_px'),
        'p95_point_err_worst_element_px': max(r['p95_point_err_px'] for r in rows),
        # One control point, and not averaged into anything: the max is a max.
        'max_point_err_px': max(r.get('max_point_err_px', 0.0) for r in rows),
        'jitter_px': av('jitter_px'),
        'frames_below_0.95': int(sum(r['frames_below_0.95'] for r in rows)),
        'frames_below_0.90': int(sum(r['frames_below_0.90'] for r in rows)),
        'keys_predicted': int(sum(r['keys_predicted'] for r in rows)),
        'keys_artist': int(sum(r['keys_artist'] for r in rows)),
        'key_f1': av('key_f1', kw),
        # Quoted with the key F1 everywhere: see Reconstruction.key_f1_over_random.
        'baseline_pipeline_key_f1_random': (
            av('baseline_pipeline_key_f1_random', kw)
            if any('baseline_pipeline_key_f1_random' in r for r in rows) else 0.0),
        'key_f1_over_random': (av('key_f1_over_random', kw)
                               if any('key_f1_over_random' in r for r in rows) else 0.0),
        'key_f1_strict': (av('key_f1_strict', kw)
                          if any('key_f1_strict' in r for r in rows) else 0.0),
        'shapes_with_no_in_range_key': int(sum(r.get('shapes_with_no_in_range_key', 0)
                                               for r in rows)),
        # The key-timing head scored on its own, beside the keys the DP actually chose. Both,
        # because they can move in opposite directions and the difference says whether the
        # bias is set too low or is doing damage. Zero for a checkpoint with no key head.
        'head_key_f1': av('head_key_f1', kw) if any('head_key_f1' in r for r in rows) else 0.0,
        'head_key_precision': (av('head_key_precision', kw)
                               if any('head_key_precision' in r for r in rows) else 0.0),
        'head_key_recall': (av('head_key_recall', kw)
                            if any('head_key_recall' in r for r in rows) else 0.0),
        # Never aggregated without them: see Reconstruction.head_key_f1_over_best_baseline.
        'baseline_key_f1_all_live': (av('baseline_key_f1_all_live', kw)
                                     if any('baseline_key_f1_all_live' in r for r in rows)
                                     else 0.0),
        'baseline_key_f1_random': (av('baseline_key_f1_random', kw)
                                   if any('baseline_key_f1_random' in r for r in rows)
                                   else 0.0),
        'head_key_f1_over_best_baseline': (av('head_key_f1_over_best_baseline', kw)
                                           if any('head_key_f1_over_best_baseline' in r
                                                  for r in rows) else 0.0),
        'key_tol_min': min((r.get('key_tol_min', 0.0) for r in rows), default=0.0),
        'key_tol_max': max((r.get('key_tol_max', 0.0) for r in rows), default=0.0),
        'per_element': list(rows),
    }
    # The same aggregates over frames where the target draws something. Weighted by on-screen
    # frames, not by all frames: an element contributes in proportion to how much of it was
    # actually visible to score. See roto.v2.score for what this corrects and by how much.
    ow = np.array([r.get('frames_onscreen', r['frames']) for r in rows], float)
    if ow.sum() > 0 and any('mean_soft_iou_onscreen' in r for r in rows):
        have = [r for r in rows if 'mean_soft_iou_onscreen' in r]
        hw = np.array([r['frames_onscreen'] for r in have], float)
        worst_on = min(have, key=lambda r: r['mean_soft_iou_onscreen'])
        tot['mean_soft_iou_onscreen'] = float(np.average(
            [r['mean_soft_iou_onscreen'] for r in have], weights=hw))
        tot['worst_element_soft_iou_onscreen'] = worst_on['mean_soft_iou_onscreen']
        tot['worst_element_onscreen'] = worst_on['element_id']
        tot['frames_onscreen'] = int(hw.sum())
        tot['frames_offscreen'] = int(sum(r.get('frames_offscreen', 0) for r in rows))
        tot['frames_below_0.90_onscreen'] = int(
            sum(r.get('frames_below_0.90_onscreen', 0) for r in have))
    # Charter S4 stage S2. Weighted by *cells* rather than frames: a lifespan mistake is a
    # (frame, shape) event, so a 247-shape element carries 247 times the exposure of a
    # 1-shape one at the same length, and weighting by frames would hide that.
    if any('lifespan_cells' in r for r in rows):
        have = [r for r in rows if r.get('lifespan_cells')]
        if have:
            cw = np.array([r['lifespan_cells'] for r in have], float)
            cav = lambda k: float(np.average([r[k] for r in have], weights=cw))
            tot['lifespan_accuracy'] = cav('lifespan_accuracy')
            # Never summed into one error rate: they cost 1 : 3.2 at the render.
            tot['lifespan_fp_rate'] = cav('lifespan_fp_rate')
            tot['lifespan_fn_rate'] = cav('lifespan_fn_rate')
            tot['lifespan_cells'] = int(cw.sum())
            # The per-element floor, per charter L3. Ranked on the FN rate rather than on
            # accuracy, because that is the side the render gate is tightest against.
            worst_ls = max(have, key=lambda r: r['lifespan_fn_rate'])
            tot['lifespan_worst_element'] = worst_ls['element_id']
            tot['lifespan_worst_fn_rate'] = worst_ls['lifespan_fn_rate']
            tot['lifespan_transitions_pred'] = sum(r['lifespan_transitions_pred']
                                                   for r in have)
            tot['lifespan_transitions_artist'] = sum(r['lifespan_transitions_artist']
                                                     for r in have)
            # Split by the dataset's own frame holdout. The gap between these two is the
            # whole question for this head -- see roto.v2.score.
            for tag in ('held', 'train'):
                k = f'{tag}_lifespan_accuracy'
                got = [r for r in have if k in r]
                if got:
                    gw = np.array([r['lifespan_cells'] for r in got], float)
                    for m in ('accuracy', 'fn_rate', 'fp_rate'):
                        tot[f'{tag}_lifespan_{m}'] = float(np.average(
                            [r[f'{tag}_lifespan_{m}'] for r in got], weights=gw))
            if 'held_lifespan_accuracy' in tot:
                tot['lifespan_held_gap'] = (tot['train_lifespan_accuracy']
                                            - tot['held_lifespan_accuracy'])
            tot['lifespan_transition_ratio'] = (
                tot['lifespan_transitions_pred']
                / max(1, tot['lifespan_transitions_artist']))
    if any(r.get('point_count_exact') is not None for r in rows):
        have = [r for r in rows if r.get('point_count_exact') is not None]
        sw = np.array([r.get('shapes') or r['frames'] for r in have], float)
        tot['point_count_exact'] = float(np.average(
            [r['point_count_exact'] for r in have], weights=sw))
        tot['point_count_over'] = int(sum(r['point_count_over'] for r in have))
        tot['point_count_under'] = int(sum(r['point_count_under'] for r in have))
    tot['key_ratio'] = tot['keys_predicted'] / max(1, tot['keys_artist'])
    tot['frames_below_0.95_pct'] = 100.0 * tot['frames_below_0.95'] / max(1, tot['frames'])
    tot['frames_below_0.90_pct'] = 100.0 * tot['frames_below_0.90'] / max(1, tot['frames'])
    # The *worst element's* share, not the run's. A gate of "frames below 0.90 <= 1% per element"
    # is a per-element statement and a run-level percentage can pass it while one element fails
    # badly: 1% of 1,810 frames is 18, which one 77-frame element could supply entirely.
    worst_pct = max(rows, key=lambda r: r['frames_below_0.90'] / max(1, r['frames']))
    tot['worst_element_frames_below_0.90_pct'] = (
        100.0 * worst_pct['frames_below_0.90'] / max(1, worst_pct['frames']))
    tot['worst_element_frames_below_0.90'] = worst_pct['element_id']

    # The held-out split, where a run has one. Weighted by held frames, not by all frames:
    # the question is how the withheld frames scored, and the trained column is there only
    # to be subtracted from them.
    if any('held_soft_iou' in r for r in rows):
        hw = np.array([r.get('held_frames', 0) for r in rows], float)
        tot['held_soft_iou'] = float(np.average(
            [r.get('held_soft_iou', 0.0) for r in rows], weights=hw))
        tot['train_soft_iou'] = float(np.average(
            [r.get('train_soft_iou', 0.0) for r in rows], weights=hw))
        tot['held_gap'] = tot['train_soft_iou'] - tot['held_soft_iou']
        tot['held_frames'] = int(hw.sum())
    return tot


def by_training_status(rows: Sequence[dict[str, Any]],
                       withheld: Sequence[str]) -> dict[str, Any]:
    """Aggregate a run three ways: the elements it trained on, the ones it did not, and all.

    v1.3 is the first round where a run is scored on elements it was never allowed to train on,
    and one aggregate cannot carry both. The **trained** block is the headline and is what the
    acceptance gates read, because a gate like "every element >= 0.90" is a statement about
    reconstruction quality and an element with freshly initialised query rows is not a
    reconstruction -- it is a measurement of the encoder with the memorisation capacity set to
    zero (``reconstruct.untrained_queries``). The **held_elements** block is that measurement,
    kept apart so it can never dilute the headline or be mistaken for a generalisation claim.
    The **all_elements** block exists only so a v002 row can still be read against v1.1's
    thirteen-element tables.

    ``withheld`` comes from the *checkpoint*, not from the dataset: the dataset says which
    elements its split withholds, and a run may have opted out (``TrainConfig.use_split``), so
    only the checkpoint knows what a given run actually saw.
    """
    held = set(withheld)
    trained = [r for r in rows if r['element_id'] not in held]
    frozen = [r for r in rows if r['element_id'] in held]
    out = dict(run_totals(trained if trained else rows))
    out['withheld_elements'] = sorted(held)
    out['all_elements'] = run_totals(rows) if frozen else None
    out['held_elements'] = run_totals(frozen) if frozen else None
    return out


def spread(runs: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Per-metric min / max / mean / sd across repeat runs -- the noise floor.

    Every ladder in v1 and v1.1 is a single seed per configuration, which makes a +0.003 row
    and a lucky row the same row. What this returns is the number that lets the rest of the
    report say "larger than noise" without hedging: the observed range of a metric when
    *nothing* changes but the seed.

    ``sd`` is the sample standard deviation (ddof=1) and is only meaningful for three or more
    runs; with two, ``range`` is the honest statistic and is what the report quotes.
    """
    out: dict[str, Any] = {'n_runs': len(runs)}
    for k in keys:
        v = np.array([r[k] for r in runs if k in r], float)
        if not len(v):
            continue
        out[k] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
                  'range': float(v.max() - v.min()),
                  'sd': float(v.std(ddof=1)) if len(v) > 2 else None,
                  'values': [float(x) for x in v]}
    return out
