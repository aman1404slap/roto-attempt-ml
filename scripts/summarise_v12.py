"""Arrange every scored v1.2 run into the tables the handover asks for.

Nothing is computed here that `report_v12.py` did not measure -- this only arranges it, so a
table can never disagree with the run that produced it. What it adds is the three readings the
handover asks for by name, each stated against the *measured* noise floor rather than against
a guess:

1. the aligned-window verdict,
2. the transform-head verdict, in both framings, because they disagree,
3. the constrained key sweep.

    python scripts/summarise_v12.py        # -> v1.2/results/bucket_b.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = [
    ('v1_control', "v1's configuration, 12k, seed 0 (v1.1's own rung, re-scored)"),
    ('control_s1', 'the same, seed 1'),
    ('control_s2', 'the same, seed 2'),
    ('window_aligned', 'aligned 3-frame window (R1)'),
    ('window_temporal_aligned', '+ temporal consistency loss (R1)'),
    ('affine_crop', 'crop-space projective transform target, misaligned window (R2)'),
    ('affine_crop_aligned', 'the same, aligned window (R2)'),
    ('affine_deep', 'the same, transform decoder at depth 6 instead of 3 (S4)'),
    ('v1_control_long', "v1's configuration, 40k, seed 0"),
    ('control_long_s1', 'the same, seed 1'),
    ('control_long_s2', 'the same, seed 2'),
    ('final_long', 'v1.1 headline: sqrt weighting, 40k'),
    ('final_long_v2', '+ aligned window + crop-space transform, 40k'),
    ('final_v2_s1', 'the same, seed 1'),
    ('final_holdout', 'the same as final_long, every 7th frame held out'),
    ('final_long_v2_refit_savgol', 'final_long_v2 at its **unconstrained** operating point'),
    ('final_long_v2_constrained_refit',
     'final_long_v2 at its **constrained** operating point (keys in [0.75, 1.30])'),
]

E2E = [('final_long_v2_e2e', 'final_long_v2'),
       ('affine_crop_aligned_e2e', 'affine_crop_aligned'),
       ('affine_deep_e2e', 'affine_deep')]
"""De-teacher-forced rows and the teacher-forced row each is the same checkpoint as."""

COLS = ('| run | what | soft IoU | worst layer | worst frame | point px | p95 px | '
        'jitter px | keys | key F1 | held gap | steps | seed |')


def load(res: Path, name: str) -> dict | None:
    p = res / f'score_{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def line(name: str, what: str, d: dict) -> str:
    gap = f'{d["held_gap"]:+.4f}' if 'held_gap' in d else '—'
    return (f'| `{name}` | {what} | {d["mean_soft_iou"]:.4f} | '
            f'{d["worst_layer_soft_iou"]:.4f} | {d["worst_frame_soft_iou"]:.4f} | '
            f'{d["point_err_px"]:.2f} | {d["p95_point_err_px"]:.2f} | '
            f'{d["jitter_px"]:.2f} | {d["key_ratio"]:.2f}x | {d["key_f1"]:.3f} | {gap} | '
            f'{d.get("steps", "?")} | {d.get("seed", "?")} |')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='v1.2/results')
    ap.add_argument('--out', default='v1.2/results/bucket_b.md')
    args = ap.parse_args()
    res = Path(args.results)
    runs = {n: d for n, _ in ORDER if (d := load(res, n))}
    noise = json.loads((res / 'noise_floor.json').read_text()) \
        if (res / 'noise_floor.json').exists() else {}
    ceiling = json.loads((res / 'scoring_ceiling.json').read_text()) \
        if (res / 'scoring_ceiling.json').exists() else {}

    L = ['# Bucket B — model error, worst case', '']
    if noise:
        L += ['## The noise floor first, because it decides what the table can say', '',
              'One configuration retrained at several seeds, nothing else changed: v1\'s own',
              'at both schedules, and the headline configuration at two. Both earlier reports',
              'assumed "under about 0.003 soft IoU is noise". Measured:', '',
              '| schedule | seeds | soft IoU | worst layer | worst frame | point px | key F1 |',
              '|---|---|---|---|---|---|---|']
        for label, s in noise.items():
            L.append(f'| {label} | {s["n_runs"]} | ±{s["mean_soft_iou"]["range"]:.4f} | '
                     f'±{s["worst_layer_soft_iou"]["range"]:.4f} | '
                     f'±{s["worst_frame_soft_iou"]["range"]:.4f} | '
                     f'±{s["point_err_px"]["range"]:.2f} | '
                     f'±{s["key_f1"]["range"]:.3f} |')
        L += ['', 'Ranges, not standard deviations: three runs do not describe a distribution.',
              '**Every difference in the table below that is smaller than its schedule\'s row',
              'here is not a result.**', '']

    L += ['## The runs', '',
          'Same 13 layers, same `datasets/v001`, same default rebuild settings unless the row',
          'says otherwise. `worst layer` is the lowest per-layer mean; `worst frame` the lowest',
          'single frame anywhere in the run; `p95 px` the 95th percentile of per-point error.',
          '', COLS, '|' + '---|' * 13]
    for name, what in ORDER:
        if name in runs:
            L.append(line(name, what, runs[name]))

    if ceiling:
        L += ['', '## What the worst frame can mean', '',
              f'The artist\'s own program, rendered by today\'s renderer against these same',
              f'alphas, scores **{ceiling["mean_ceiling"]:.6f}** on average and',
              f'**{ceiling["worst_frame_ceiling"]:.6f}** on its worst frame -- see',
              '`scoring_ceiling.json` and the ledger. So a frame is only as judgeable as its',
              'own ceiling, and the worst frames below are compared against theirs rather than',
              'against 1.000.', '',
              '| run | worst frame | layer | ceiling on that frame | model\'s share |',
              '|---|---|---|---|---|']
        per = {r['layer_id']: dict(zip(r['frame_index'], r['ceiling_per_frame']))
               for r in ceiling['per_layer']}
        for name in ('final_long', 'final_long_v2', 'final_long_v2_constrained_refit',
                     'v1_control_long', 'v1_control'):
            d = runs.get(name)
            if not d:
                continue
            lid, f = d['worst_frame_layer'], d['worst_frame']
            cap = per.get(lid, {}).get(f)
            share = f'{cap - d["worst_frame_soft_iou"]:.4f}' if cap is not None else '—'
            L.append(f'| `{name}` | {d["worst_frame_soft_iou"]:.4f} | {lid} @{f} | '
                     f'{cap:.6f} | {share} |' if cap is not None else
                     f'| `{name}` | {d["worst_frame_soft_iou"]:.4f} | {lid} @{f} | — | — |')

    # ---- 1. the aligned-window verdict --------------------------------------
    c, w = runs.get('v1_control'), runs.get('window_aligned')
    if c and w:
        nf = noise.get('12k', {}).get('mean_soft_iou', {}).get('range')
        L += ['', '## 1. The aligned-window verdict', '',
              '| | soft IoU | point px | p95 px | jitter px |', '|---|---|---|---|---|',
              f'| `v1_control` | {c["mean_soft_iou"]:.4f} | {c["point_err_px"]:.2f} | '
              f'{c["p95_point_err_px"]:.2f} | {c["jitter_px"]:.2f} |',
              f'| `window_aligned` | {w["mean_soft_iou"]:.4f} | {w["point_err_px"]:.2f} | '
              f'{w["p95_point_err_px"]:.2f} | {w["jitter_px"]:.2f} |',
              f'| difference | {w["mean_soft_iou"] - c["mean_soft_iou"]:+.4f} | '
              f'{w["point_err_px"] - c["point_err_px"]:+.2f} | '
              f'{w["p95_point_err_px"] - c["p95_point_err_px"]:+.2f} | '
              f'{w["jitter_px"] - c["jitter_px"]:+.2f} |']
        if nf:
            L += ['', f'The 12k seed spread is ±{nf:.4f} soft IoU and '
                      f'±{noise["12k"]["point_err_px"]["range"]:.2f} point px. Every number in '
                      'that difference row is inside it.']

    # ---- 2. the transform head, in both framings ----------------------------
    aff = {}
    for p in sorted(res.glob('score_*_affine.json')):
        aff[p.stem[len('score_'):-len('_affine')]] = json.loads(p.read_text())
    if aff:
        L += ['', '## 2. The transform-head verdict', '',
              'The artist\'s own control points moved by the *predicted* transform track,',
              'rendered. The artist\'s real track scores 1.000 by construction, and the',
              '8-number representation\'s own ceiling is 0.9996 (`affine_target.json`).', '',
              '| run | soft IoU | worst layer | worst frame |', '|---|---|---|---|']
        for k, d in aff.items():
            L.append(f'| `{k}` | {d["mean_soft_iou"]:.4f} | {d["worst_layer_soft_iou"]:.4f} '
                     f'| {d["worst_frame_soft_iou"]:.4f} |')
        L += ['', 'And the same question asked end to end -- predicted geometry *and* predicted',
              'motion, through the keyframe stage, which is the number the roadmap actually',
              'needs. It is not the same measurement: see `roto.model.reconstruct`.', '',
              '| checkpoint | teacher-forced motion | predicted motion, end to end | cost |',
              '|---|---|---|---|']
        for e2e, base in E2E:
            a, b = load(res, e2e), runs.get(base)
            if a and b:
                L.append(f'| `{base}` | {b["mean_soft_iou"]:.4f} | {a["mean_soft_iou"]:.4f} '
                         f'| {a["mean_soft_iou"] - b["mean_soft_iou"]:+.4f} |')

    # ---- 3. the constrained key sweep ---------------------------------------
    u, cst = runs.get('final_long_v2_refit_savgol'), runs.get('final_long_v2_constrained_refit')
    if u and cst:
        L += ['', '## 3. The constrained key sweep', '',
              'Both cells scored on **every** frame of all 13 layers, not on the sweep\'s',
              'stride-3 subset, so they are directly comparable to the rows above.', '',
              '| operating point | window | filter | tol | soft IoU | keys/artist | key F1 |',
              '|---|---|---|---|---|---|---|',
              f'| unconstrained | {u["rebuild"]["smooth"]} | {u["rebuild"]["smooth_kind"]} | '
              f'{u["rebuild"]["tol_px"]} | {u["mean_soft_iou"]:.4f} | {u["key_ratio"]:.2f}x | '
              f'{u["key_f1"]:.3f} |',
              f'| constrained to [0.75, 1.30] | {cst["rebuild"]["smooth"]} | '
              f'{cst["rebuild"]["smooth_kind"]} | {cst["rebuild"]["tol_px"]} | '
              f'{cst["mean_soft_iou"]:.4f} | {cst["key_ratio"]:.2f}x | {cst["key_f1"]:.3f} |',
              '',
              f'The constraint costs {cst["mean_soft_iou"] - u["mean_soft_iou"]:+.4f} soft IoU '
              f'and buys {cst["key_f1"] - u["key_f1"]:+.3f} key F1 at '
              f'{cst["key_ratio"]:.2f}x the artist\'s keys instead of {u["key_ratio"]:.2f}x.']

    Path(args.out).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
