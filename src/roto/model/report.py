"""Per-layer reconstructions -> the row a report quotes.

Split out of the scoring scripts because v1.2 changes what a row *is*. v1 and v1.1 reported
means: soft IoU averaged over frames, point error averaged over live control points. The
handover's central argument is that a mean is the wrong summary for this product --

    "A shot with 190 perfect frames and one bad one is a rejected shot."

-- so every row here carries the tail beside the mean: the worst layer, the worst single
frame, the 95th percentile of point error, and how many frames fall below thresholds an
artist would notice. Nothing is recomputed from pixels; this only arranges what
``reconstruct.assemble`` already measured, so a table can never disagree with the run that
produced it.

Two aggregation rules, stated because they are choices rather than arithmetic:

* **Means are frame-weighted.** A 320-frame layer counts more than a 60-frame one, which is
  what "the average frame in this archive" means. v1 and v1.1 do the same, so the columns
  stay comparable row for row.
* **Tails are not averaged into anything.** The worst layer is a layer, and the worst frame
  is a frame -- both are reported with their identity, because "0.87" is a number to argue
  about and "0.87 on nfl_0200 green at frame 1042" is a thing to go and look at.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def run_totals(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-layer summaries (``Reconstruction.summary()``) into one row.

    ``rows`` must be non-empty. Key F1 and the key ratio are weighted by the *artist's* key
    count rather than by frames, because they are statements about keys.
    """
    if not rows:
        raise ValueError('no layers to aggregate')
    w = np.array([r['frames'] for r in rows], float)
    kw = np.array([r['keys_artist'] for r in rows], float)
    av = lambda k, wt=w: float(np.average([r[k] for r in rows], weights=wt))

    worst_layer = min(rows, key=lambda r: r['mean_soft_iou'])
    worst_frame = min(rows, key=lambda r: r['min_soft_iou'])
    tot = {
        'layers': len(rows), 'frames': int(w.sum()),
        'mean_soft_iou': av('mean_soft_iou'), 'mean_iou': av('mean_iou'),
        # The three worst-case columns the handover asks for by name.
        'worst_layer_soft_iou': worst_layer['mean_soft_iou'],
        'worst_layer': worst_layer['layer_id'],
        'worst_frame_soft_iou': worst_frame['min_soft_iou'],
        'worst_frame': worst_frame['worst_frame'],
        'worst_frame_layer': worst_frame['layer_id'],
        'point_err_px': av('point_err_px'),
        'p95_point_err_px': av('p95_point_err_px'),
        'p95_point_err_worst_layer_px': max(r['p95_point_err_px'] for r in rows),
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
        'per_layer': list(rows),
    }
    tot['key_ratio'] = tot['keys_predicted'] / max(1, tot['keys_artist'])
    tot['frames_below_0.95_pct'] = 100.0 * tot['frames_below_0.95'] / max(1, tot['frames'])
    tot['frames_below_0.90_pct'] = 100.0 * tot['frames_below_0.90'] / max(1, tot['frames'])
    # The *worst layer's* share, not the run's. A gate of "frames below 0.90 <= 1% per layer"
    # is a per-layer statement and a run-level percentage can pass it while one layer fails
    # badly: 1% of 1,810 frames is 18, which one 77-frame layer could supply entirely.
    worst_pct = max(rows, key=lambda r: r['frames_below_0.90'] / max(1, r['frames']))
    tot['worst_layer_frames_below_0.90_pct'] = (
        100.0 * worst_pct['frames_below_0.90'] / max(1, worst_pct['frames']))
    tot['worst_layer_frames_below_0.90'] = worst_pct['layer_id']

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
    """Aggregate a run three ways: the layers it trained on, the ones it did not, and all.

    v1.3 is the first round where a run is scored on layers it was never allowed to train on,
    and one aggregate cannot carry both. The **trained** block is the headline and is what the
    acceptance gates read, because a gate like "every layer >= 0.90" is a statement about
    reconstruction quality and a layer with freshly initialised query rows is not a
    reconstruction -- it is a measurement of the encoder with the memorisation capacity set to
    zero (``reconstruct.untrained_queries``). The **held_layers** block is that measurement,
    kept apart so it can never dilute the headline or be mistaken for a generalisation claim.
    The **all_layers** block exists only so a v002 row can still be read against v1.1's
    thirteen-layer tables.

    ``withheld`` comes from the *checkpoint*, not from the dataset: the dataset says which
    layers its split withholds, and a run may have opted out (``TrainConfig.use_split``), so
    only the checkpoint knows what a given run actually saw.
    """
    held = set(withheld)
    trained = [r for r in rows if r['layer_id'] not in held]
    frozen = [r for r in rows if r['layer_id'] in held]
    out = dict(run_totals(trained if trained else rows))
    out['withheld_layers'] = sorted(held)
    out['all_layers'] = run_totals(rows) if frozen else None
    out['held_layers'] = run_totals(frozen) if frozen else None
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
