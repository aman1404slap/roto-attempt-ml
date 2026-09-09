"""Choose the keyframe operating point for a v1.3 model, including the key-timing bias.

Same argument as `sweep_operating_point.py` and one more axis. The tolerance has to sit above
the model's own jitter rather than above the artist's precision, so it is re-chosen whenever
the model changes; and v1.3 adds a knob that did not exist -- how far the key-timing head may
tighten that tolerance locally (`RebuildConfig.key_bias`).

Five axes, and two of them are free in a way worth stating: `key_bias` and `key_thresh` are
*scoring-time* parameters, so sweeping them costs no training at all. The prediction is
computed once per layer and reused across every cell, because none of these axes change it.

    smoothing window   1 (off), 5, 9, 13
    filter             boxcar (v1) vs savgol (keeps the peaks artists key)
    tolerance          0.5 .. 4.0 crop px
    key values         from the filtered track (v1) vs refitted to the raw track
    key bias           0 (v1.2) .. 0.8       tighten where a key is expected
    key slack          0 (v1.2) .. 1.0       loosen where none is -- only these two need a
                                             checkpoint with a key head

**And "the bias raised key F1" is not on its own a result**, for two reasons that took two
passes to separate. The bias raises the key *count*, and a model that under-keys gains recall
-- and therefore F1 -- from any extra key, which a lower tolerance supplies with no head at
all. And matching the count *in aggregate* is still not enough: key density ranges 13x across
these layers, so a bias that merely moves keys towards the densely-keyed layer wins the
aggregate at a fixed total. So the control matches the key count **per layer**, and its delta
is the only number in this sweep that is about the head.

**Maximising rendered IoU alone is the wrong objective**, and v1.1 measured how wrong: its
unconstrained optimum lands at 0.61x the artist's key count with key F1 0.36, because rendered
IoU never asks *where* the keys are. So the sweep reports two answers -- the unconstrained
best, and the best whose key ratio stays inside `--key-ratio-range` -- and the constrained one
is the one to ship. With a key head there is a third question the sweep has to answer, and it
is the one this round exists for: does the bias buy key F1 at a cost in soft IoU an artist
could see, or not.

**The gate configuration needs its own sweep.** The plan's S4 gates the candidate with
predicted motion end to end, and that is not the same operating point: the keyframe search
runs on the *local* track, and inverting a predicted matrix rather than the artist's gives a
noisier one, which a tolerance chosen for the quieter track over-keys. Measured on
`v002_control`, reusing the teacher-forced winner at predicted motion puts the key count at
1.3x to 9x the artist's on individual layers. So `--motion predicted` re-runs the whole grid
through that path.

    python scripts/sweep_v13.py --run v002_keytime
    python scripts/sweep_v13.py --run v002_final --biases 0
    python scripts/sweep_v13.py --run v002_final --motion predicted --biases 0   # the gate
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.model.data import load_element                               # noqa: E402
from roto.model.reconstruct import (ARTIST, MOTION_SOURCES, PREDICTED,  # noqa: E402
                                    RebuildConfig, load_model, predict,
                                    predicted_shape_matrices, rebuild, score_doc,
                                    to_local, transform_matrices, with_transforms)
from roto.model.smoothing import BOXCAR, SAVGOL                        # noqa: E402

LAYERS = [
    'FAM_0060_L1_A0003C007_v001__blue',
    'nfl_0200_bg01_v001_compplate_roto_v001__blue',
    'TVC_SHOTS_sh0260_BG01_v003_roto_v02__Layer_52',
]
"""Three layers spanning the quality range -- the same three v1 and v1.1 swept, so the tables
are comparable row for row: a near-perfect small layer, a mid one, and the 592-shape worst
case. All three are layers every run trains on, so the sweep is not choosing an operating
point on the strength of a layer whose queries are untrained."""

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
    ap.add_argument('--run', required=True)
    ap.add_argument('--dataset', default=None, help="default: the checkpoint's own dataset")
    ap.add_argument('--out', default='v1.3/results')
    ap.add_argument('--stride', type=int, default=3,
                    help='score every Nth frame; the sweep compares cells, not headlines')
    ap.add_argument('--windows', type=int, nargs='+', default=[1, 5, 9, 13])
    ap.add_argument('--tols', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--biases', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.8],
                    help='key-timing bias: how far the head may tighten the tolerance where '
                         'it expects a key. Ignored for a checkpoint with no key head')
    ap.add_argument('--slacks', type=float, nargs='+', default=[0.0, 0.5, 1.0],
                    help='how far the head may loosen the tolerance where it expects none. '
                         'The half that can raise precision, which is what caps key F1')
    ap.add_argument('--key-thresh', type=float, default=0.5,
                    help="threshold for the head's own reported F1; not used by the DP")
    ap.add_argument('--motion', default=ARTIST, choices=list(MOTION_SOURCES),
                    help="whose transform track to sweep through. 'predicted' is the gate "
                         "configuration, and it is not the same sweep: the tolerance has to "
                         "sit above the model's own noise, and taking the artist's motion "
                         'track away adds noise to the local track the keys are chosen on. '
                         'Reusing the teacher-forced winner would place keys with a '
                         'tolerance chosen for a quieter track')
    ap.add_argument('--key-ratio-range', type=float, nargs=2, default=[0.75, 1.3],
                    metavar=('LO', 'HI'),
                    help='key economy an artist can still edit; the constrained optimum is '
                         'the best cell inside it')
    args = ap.parse_args()
    lo, hi = args.key_ratio_range

    ckpt = find_run(args.run)
    net, ck = load_model(ckpt)
    sbase, gbase = ck.get('shape_base', {}), ck.get('group_base', {})
    dataset = args.dataset or ck.get('dataset')
    if not dataset:
        raise SystemExit(f'{ckpt} records no dataset (it predates v1.3), so pass --dataset '
                         'explicitly -- see scripts/report_v13.py for why it is not guessed.')
    has_key = bool(getattr(net, 'key_head', False))
    biases = args.biases if has_key else [0.0]
    slacks = args.slacks if has_key else [0.0]
    if not has_key and (any(args.biases) or any(args.slacks)):
        print(f'{args.run} has no key head; the bias and slack axes are dropped rather than '
              'run at a uniform 0, which would be the same cell many times over.')
    # (0, 0) is the unbiased cell and appears once; every other pair is a real cell.
    pairs = sorted({(b, k) for b in biases for k in slacks})
    grid = list(itertools.product(args.windows, (BOXCAR, SAVGOL), args.tols, (False, True),
                                  pairs))

    rows: list[dict] = []
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    e2e = args.motion == PREDICTED
    out_path = out_dir / f'operating_point_{args.run}{"_e2e" if e2e else ""}.json'
    for name in LAYERS:
        d = Path(dataset) / name
        el = load_element(d)
        crop_pts, pred_aff, key_prob = predict(net, el, sbase.get(name, 0), gbase.get(name, 0))
        # Both halves of the round trip use the predicted matrix, or the error measured is the
        # mismatch between them rather than the head's -- see roto.model.reconstruct.
        mats = predicted_shape_matrices(el, pred_aff) if e2e else None
        doc_mats = transform_matrices(el, pred_aff) if e2e else None
        local = to_local(el, crop_pts, mats)               # once: independent of every axis
        frames = [int(f) for f in el.frames[::args.stride]]
        print(f'\n=== {name} ({len(el.frames)} frames, scoring {len(frames)}, '
              f'{args.motion} motion) ===')
        print(f'{"win":>4} {"filter":>7} {"tol":>5} {"refit":>6} {"bias":>5} {"slack":>6} '
              f'{"keys/artist":>12} {"F1":>6} {"headF1":>7} {"vsBase":>7} {"softIoU":>8}')
        for win, kind, tol, refit, (bias, slack) in grid:
            if win <= 1 and kind == SAVGOL:
                continue                                   # no window, no filter to choose
            cfg = RebuildConfig(tol_px=tol, smooth=win, smooth_kind=kind,
                                refit_values=refit, key_bias=bias, key_slack=slack,
                                key_thresh=args.key_thresh)
            doc, st = rebuild(el, local, cfg, key_prob)
            if e2e:
                doc = with_transforms(doc, el, doc_mats)
            soft, hard = score_doc(el, doc, frames)
            row = {'layer': name, 'window': win, 'filter': kind, 'tol_px': tol,
                   'refit': refit, 'key_bias': bias, 'key_slack': slack,
                   'motion': args.motion,
                   'keys_predicted': st['keys_predicted'],
                   'keys_artist': st['keys_artist'],
                   'key_ratio': st['keys_predicted'] / max(1, st['keys_artist']),
                   'key_f1': st['key_f1'],
                   'head_key_f1': st.get('head_key_f1', 0.0),
                   'head_key_f1_over_best_baseline':
                       st.get('head_key_f1_over_best_baseline', 0.0),
                   'tol_min': st['key_tol_min'], 'tol_max': st['key_tol_max'],
                   'mean_soft_iou': float(soft.mean()), 'mean_iou': float(hard.mean())}
            rows.append(row)
            print(f'{win:>4} {kind:>7} {tol:>5.1f} {str(refit):>6} {bias:>5.1f} '
                  f'{slack:>6.1f} {row["key_ratio"]:>11.2f}x {row["key_f1"]:>6.3f} '
                  f'{row["head_key_f1"]:>7.3f} '
                  f'{row["head_key_f1_over_best_baseline"]:>+7.3f} '
                  f'{row["mean_soft_iou"]:>8.4f}', flush=True)
            out_path.write_text(json.dumps(
                {'run': args.run, 'dataset': dataset, 'stride': args.stride,
                 'motion': args.motion, 'has_key_head': has_key, 'rows': rows}, indent=2))

    print(f'\n{len(rows)} cells in {time.time() - t0:.0f}s')
    by_cell: dict[tuple, list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r['window'], r['filter'], r['tol_px'], r['refit'],
                            r['key_bias'], r['key_slack']), []).append(r)
    agg = [{'cell': k,
            'mean_soft_iou': float(np.mean([x['mean_soft_iou'] for x in v])),
            'key_ratio': float(np.mean([x['key_ratio'] for x in v])),
            'key_f1': float(np.mean([x['key_f1'] for x in v])),
            'head_key_f1': float(np.mean([x['head_key_f1'] for x in v])),
            'head_over_baseline': float(np.mean(
                [x['head_key_f1_over_best_baseline'] for x in v]))}
           for k, v in by_cell.items() if len(v) == len(LAYERS)]
    agg.sort(key=lambda r: -r['mean_soft_iou'])
    best = agg[0]
    inside = [a for a in agg if lo <= a['key_ratio'] <= hi]
    constrained = max(inside, key=lambda r: r['mean_soft_iou']) if inside else None
    # And the cell that maximises key F1 inside the band -- the objective this round added.
    # It is reported separately from the IoU-best cell because the two need not coincide, and
    # if they do not, which to ship is a judgement about the deliverable rather than a max.
    best_f1 = max(inside, key=lambda r: r['key_f1']) if inside else None

    fmt = ('window=%s filter=%s tol=%s refit=%s bias=%s slack=%s -> %.4f at %.2fx keys, '
           'F1 %.3f (head %.3f)')
    args_of = lambda r: (*r['cell'], r['mean_soft_iou'], r['key_ratio'], r['key_f1'],
                         r['head_key_f1'])
    print('\nbest soft IoU (unconstrained)  ' + fmt % args_of(best))
    if constrained:
        print(f'best IoU with keys in [{lo:.2f}, {hi:.2f}]  ' + fmt % args_of(constrained))
        print(f'  the constraint costs '
              f'{constrained["mean_soft_iou"] - best["mean_soft_iou"]:+.4f} soft IoU and buys '
              f'{constrained["key_f1"] - best["key_f1"]:+.3f} key F1')
    if not inside:
        # Said out loud rather than omitted: a sweep whose grid cannot satisfy the constraint
        # must not look like a sweep that chose not to report it.
        print(f'no cell has a key ratio in [{lo:.2f}, {hi:.2f}] -- the grid ranges '
              f'{min(a["key_ratio"] for a in agg):.2f}x to '
              f'{max(a["key_ratio"] for a in agg):.2f}x. Widen --tols downward: a tighter '
              f'tolerance places more keys.')
    if best_f1:
        print(f'best key F1 with keys in [{lo:.2f}, {hi:.2f}]  ' + fmt % args_of(best_f1))
        if constrained:
            print(f'  against the IoU-best constrained cell: '
                  f'{best_f1["key_f1"] - constrained["key_f1"]:+.3f} key F1 for '
                  f'{best_f1["mean_soft_iou"] - constrained["mean_soft_iou"]:+.4f} soft IoU')
    matched = None
    if has_key:
        # ---- the control that decides whether the head is worth anything --------------
        # "Bias raises key F1" is not a result on its own, for two reasons, and the second one
        # only showed up when the first was controlled for.
        #
        # 1. The bias raises the *key count*, and when a model under-keys, more keys raises
        #    recall and therefore F1 for free -- which a lower tolerance supplies with no head
        #    at all. So the comparison must be at a matched key count.
        #
        # 2. Matching the count **in aggregate** is not enough. Key density ranges 13x across
        #    these layers, so F1 is far cheaper to earn on a dense layer than a sparse one, and
        #    a bias that merely *redistributes* keys towards the dense layer wins the aggregate
        #    while holding the total fixed. Measured: on the 12k probe the best aggregate-
        #    matched pair read +0.063, of which +0.135 came from `Layer_52` at **0.84x keys
        #    against the rival's 0.61x** -- not matched at all -- while the two sparse layers
        #    contributed +0.037 and +0.017 at slightly *fewer* keys than their rivals.
        #
        # So the match is per layer: every biased cell against the unbiased cell that places
        # the same number of keys **on that layer**, and the deltas are then averaged. Only the
        # layers that find a rival count, and how many did is reported, because a control that
        # silently drops the layers it cannot match is not a control.
        KEY_MATCH = 0.04
        layers = sorted({r['layer'] for r in rows})
        by_layer_cell = {(r['layer'], (r['window'], r['filter'], r['tol_px'], r['refit'],
                                       r['key_bias'], r['key_slack'])): r for r in rows}
        unb_by_layer = {L: [r for r in rows
                            if r['layer'] == L and not r['key_bias'] and not r['key_slack']]
                        for L in layers}
        cells = sorted({(r['window'], r['filter'], r['tol_px'], r['refit'], r['key_bias'],
                         r['key_slack']) for r in rows if r['key_bias'] or r['key_slack']})
        pairs_out = []
        for cell in cells:
            per, matched_layers = [], []
            for L in layers:
                a = by_layer_cell.get((L, cell))
                if a is None:
                    continue
                near = [u for u in unb_by_layer[L]
                        if abs(u['key_ratio'] - a['key_ratio']) <= KEY_MATCH]
                if not near:
                    continue
                rival = max(near, key=lambda u: u['key_f1'])
                per.append({'layer': L, 'keys': a['key_ratio'],
                            'rival_keys': rival['key_ratio'],
                            'key_f1': a['key_f1'], 'rival_key_f1': rival['key_f1'],
                            'delta_key_f1': a['key_f1'] - rival['key_f1'],
                            'rival_cell': rival['window'], 'rival_tol': rival['tol_px']})
                matched_layers.append(L)
            if len(per) < 2:                 # one layer is an anecdote, not a control
                continue
            pairs_out.append({
                'cell': cell, 'layers_matched': len(per), 'layers_total': len(layers),
                'mean_delta_key_f1': float(np.mean([r['delta_key_f1'] for r in per])),
                'worst_delta_key_f1': min(r['delta_key_f1'] for r in per),
                'per_layer': per})
        if pairs_out:
            pairs_out.sort(key=lambda r: -r['mean_delta_key_f1'])
            matched = {'key_match_tolerance': KEY_MATCH, 'matched_per_layer': True,
                       'cells': pairs_out,
                       'best_mean_delta_key_f1': pairs_out[0]['mean_delta_key_f1'],
                       'best_worst_delta_key_f1': pairs_out[0]['worst_delta_key_f1']}
            print(f'\nthe control: each biased cell against the unbiased cell placing the same '
                  f'number of keys\nON THAT LAYER (within {KEY_MATCH:.2f}x). Matching the count '
                  f'only in aggregate lets a bias win by\nmoving keys to the densely-keyed '
                  f'layer, where F1 is cheap.')
            print(f'  {"bias":>5} {"slack":>6} {"tol":>5} {"filter":>7} {"layers":>7} '
                  f'{"mean d F1":>10} {"worst d F1":>11}')
            for r in pairs_out[:6]:
                print(f'  {r["cell"][4]:>5.1f} {r["cell"][5]:>6.1f} {r["cell"][2]:>5.2f} '
                      f'{r["cell"][1]:>7} {r["layers_matched"]:>3}/{r["layers_total"]:<3} '
                      f'{r["mean_delta_key_f1"]:>+10.3f} {r["worst_delta_key_f1"]:>+11.3f}')
            top = pairs_out[0]
            print(f'  best cell, per layer:')
            for r in top['per_layer']:
                print(f'    {r["layer"].split("__")[-1][:26]:<28} {r["keys"]:.2f}x vs '
                      f'{r["rival_keys"]:.2f}x   F1 {r["key_f1"]:.3f} vs '
                      f'{r["rival_key_f1"]:.3f}   {r["delta_key_f1"]:+.3f}')
        else:
            print(f'\nno unbiased cell lands within {KEY_MATCH:.2f}x of a biased one on two or '
                  'more layers, so the\nmatched-key-count control cannot be run on this grid. '
                  'Widen --tols.')

        # What the knobs are worth, holding everything else at the constrained winner. Read
        # this *after* the control above: it does not hold the key count fixed.
        base = constrained or best
        w, f, t, rf = base['cell'][:4]
        line = [a for a in agg if a['cell'][:4] == (w, f, t, rf)]
        line.sort(key=lambda r: (r['cell'][4], r['cell'][5]))
        print('\nthe key-timing knobs, holding the winning cell fixed otherwise:')
        for a in line:
            print(f'  bias {a["cell"][4]:>4.1f} slack {a["cell"][5]:>4.1f} -> soft IoU '
                  f'{a["mean_soft_iou"]:.4f}  keys {a["key_ratio"]:.2f}x  '
                  f'key F1 {a["key_f1"]:.3f}  head F1 {a["head_key_f1"]:.3f} '
                  f'({a["head_over_baseline"]:+.3f} vs the best trivial baseline)')

    out_path.write_text(json.dumps(
        {'run': args.run, 'dataset': dataset, 'stride': args.stride,
         'motion': args.motion, 'has_key_head': has_key,
         'key_ratio_range': [lo, hi], 'rows': rows,
         'aggregate': agg, 'best': best, 'best_constrained': constrained,
         'best_constrained_key_f1': best_f1,
         'matched_key_count_control': matched}, indent=2))
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
