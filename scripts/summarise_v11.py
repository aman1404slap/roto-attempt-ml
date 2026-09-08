"""Collect every scored run into the tables the v1.1 report is built from.

Reads whatever ``score_*.json`` files exist under the results directory plus v1's own
published numbers, and writes ``ladder.md``. Nothing is computed here that was not measured
by ``report_v11.py`` -- this only arranges it, so the table can never disagree with the runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ['v1_published', 'v1_rebaselined', 'v1_control_runsampling', 'v1_control',
         'window', 'window_temporal', 'window_aligned', 'window_temporal_aligned',
         'window_temporal_curve', 'full', 'full_sqrt_weight', 'affine_crop',
         'affine_crop_aligned', 'affine_crop_light', 'v1_control_long', 'final_long',
         'final_long_v2', 'final_holdout']

WHAT = {
    'v1_published': 'v1 as shipped, with v1\'s own metrics',
    'v1_rebaselined': 'same checkpoint, corrected metrics (S6, S7, supersample)',
    'v1_control_runsampling': 'v1 config, contiguous-run sampling',
    'v1_control': 'v1 config retrained (random sampling)',
    'window': '+ 3-frame temporal window, **misaligned** (S1a)',
    'window_temporal': '+ temporal consistency loss, misaligned window (S1b)',
    'window_aligned': 'the window, neighbours warped into the anchor crop (R1)',
    'window_temporal_aligned': '+ temporal consistency loss, aligned window (R1)',
    'window_temporal_curve': '+ curve-space loss (S3)',
    'full': '+ query self-attention (S9)',
    'full_sqrt_weight': 'full, sqrt(frames*shapes) layer weighting (S12)',
    'affine_crop': 'full_sqrt_weight + crop-space projective transform target (R2)',
    'affine_crop_aligned': 'the same, with the aligned window (R2)',
    'affine_crop_light': 'the same, transform term at weight 0.25 instead of 1.0 (R2)',
    'v1_control_long': "v1 config, 40k steps -- the schedule's own contribution",
    'final_long': 'v1.1 config (sqrt weighting), 40k steps',
    'final_long_v2': 'final_long + aligned window + crop-space transform, 40k steps',
    'final_holdout': 'the same, every 7th frame held out (S4)',
}
"""``S``n cites the v1 review's numbering; ``R``n the v1.1 review's."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='v1.1/results')
    ap.add_argument('--out', default='v1.1/results/ladder.md')
    args = ap.parse_args()
    res = Path(args.results)

    runs: dict[str, dict] = {}
    for f in sorted(res.glob('score_*.json')):
        d = json.loads(f.read_text())
        key = f.stem[len('score_'):]
        runs[key] = d

    v1 = Path('v1/results/reconstruction.json')
    if v1.exists():
        d = json.loads(v1.read_text())
        runs['v1_published'] = d

    lines = ['# v1.1 measurement ladder', '',
             'Every row is a separate training run scored by `scripts/report_v11.py`, except',
             'the two v1 rows, which score v1\'s own checkpoint. One thing changes per row.',
             '',
             '| run | what changed | soft IoU | IoU | point px | jitter px | keys/artist | key F1 |',
             '|---|---|---|---|---|---|---|---|']
    for name in ORDER:
        r = runs.get(name)
        if not r:
            continue
        j = r.get('jitter_px')
        lines.append(
            f'| `{name}` | {WHAT.get(name, "")} | {r["mean_soft_iou"]:.4f} | '
            f'{r["mean_iou"]:.4f} | {r["point_err_px"]:.2f} | '
            f'{j:.2f} | ' if j is not None else
            f'| `{name}` | {WHAT.get(name, "")} | {r["mean_soft_iou"]:.4f} | '
            f'{r["mean_iou"]:.4f} | {r["point_err_px"]:.2f} | - | ')
        lines[-1] += f'{r["key_ratio"]:.2f}x | {r["key_f1"]:.3f} |'

    affine = sorted(k for k in runs if k.endswith('_affine'))
    if affine:
        lines += ['', '## The transform head, scored on its own', '',
                  'The artist\'s own control points moved by the *predicted* transform track,',
                  'rendered. The artist\'s real track scores 1.000 by construction. A target',
                  'the representation cannot express caps this before the model is consulted:',
                  '`results/affine_target.json` measures that ceiling at **0.9930** for v1\'s',
                  '6-number affine form and **0.9996** for v1.1\'s 8-number projective one.',
                  '',
                  '| run | transform target | soft IoU |', '|---|---|---|']
        for k in affine:
            r = runs[k]
            dof = (r.get('train_config') or {}).get('affine_space', 'doc')
            label = ('crop-space projective (8)' if dof == 'crop'
                     else 'document-space affine (6)')
            lines.append(f'| `{k}` | {label} | {r["mean_soft_iou"]:.4f} |')

    extra = [k for k in sorted(runs) if k not in ORDER and k not in affine]
    if extra:
        lines += ['', '## Rebuild variants (same checkpoint, different keyframe stage)', '',
                  '| variant | soft IoU | keys/artist | key F1 |', '|---|---|---|---|']
        for k in extra:
            r = runs[k]
            lines.append(f'| `{k}` | {r["mean_soft_iou"]:.4f} | {r["key_ratio"]:.2f}x | '
                         f'{r["key_f1"]:.3f} |')

    held = [(k, r) for k, r in runs.items() if 'held_soft_iou' in r]
    if held:
        lines += ['', '## Held-out frames', '',
                  '| run | trained frames | held-out frames | gap |', '|---|---|---|---|']
        for k, r in held:
            gap = r['train_soft_iou'] - r['held_soft_iou']
            lines.append(f'| `{k}` | {r["train_soft_iou"]:.4f} | {r["held_soft_iou"]:.4f} '
                         f'| {gap:+.4f} |')

    Path(args.out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
