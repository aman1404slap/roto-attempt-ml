"""What a key-timing metric is worth, measured against the things that require no skill.

This is the measurement that caught an acceptance gate, and the reason it exists is that key
F1 -- with a one-frame matching tolerance, one-to-one matched, weighted by the artist's own key
count -- looks like a strict metric and is not. It is close to saturated by keying *often*, and
this archive makes that cheap in a specific way:

**Key density ranges 13x across the layers.** 0.037 of live (frame, shape) cells on `FAM green`
against 0.474 on `TVC Layer_52`, archive mean 0.113. And the two densest layers carry 40% of
the archive's 22,645 keys, so the aggregate is dominated by the layers where a good score is
cheapest to earn.

Three baselines, none of which knows anything about timing:

* **all live** -- fire on every live frame. The over-keying degenerate case, and the one an
  unweighted loss converges to the complement of.
* **random, same count** -- the same number of keys as the thing being measured, placed at
  random over the same live frames. This is the baseline for *placement*: it holds the count
  fixed, so anything above it is knowing where.
* **the artist's own keys** -- must score exactly 1.000, or the metric is misaligned. The
  must-come-out-perfect check, applied to a metric rather than to a render.

Measured on `v002_keytime_probe` (12k, one global positive weight), the key-timing head scored
**0.803** on `Layer_52` -- past the plan's >= 0.50 gate -- against **0.797 for firing on every
live frame**. It had learned each layer's base rate and nothing inside any of them. On
`FAM blue`, density 0.057, it stopped firing at all and scored 0.017 against a 0.111 baseline.

    python scripts/exp_key_baselines.py --run v002_keytime
    python scripts/exp_key_baselines.py --run v002_control     # the pipeline only, no head
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_v13 import find_run                                        # noqa: E402
from roto.model.data import load_element                               # noqa: E402
from roto.model.reconstruct import (RebuildConfig, load_model, predict,  # noqa: E402
                                    rebuild, to_local)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='v002_keytime')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--key-thresh', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    net, ck = load_model(find_run(args.run))
    sb, gb = ck.get('shape_base', {}), ck.get('group_base', {})
    dataset = Path(args.dataset or ck.get('dataset') or '')
    if not dataset.name:
        raise SystemExit('checkpoint records no dataset; pass --dataset '
                         '-- see scripts/report_v13.py for why')
    withheld = set(ck.get('withheld_layers', []))
    has_key = bool(getattr(net, 'key_head', False))
    cfg = RebuildConfig(key_thresh=args.key_thresh)
    rng = np.random.default_rng(args.seed)

    rows = []
    print(f'{"layer":<24} {"density":>8} | {"the pipeline":>25} | '
          f'{"the head at its threshold":>34}')
    print(f'{"":<24} {"k/live":>8} | {"keys":>6} {"F1":>6} {"rand":>6} {"skill":>6} | '
          f'{"fires":>6} {"F1":>6} {"allLv":>6} {"rand":>6} {"skill":>6}')
    for d in sorted(p for p in dataset.iterdir() if (p / 'meta.json').exists()):
        el = load_element(d)
        if el.layer_id in withheld or el.layer_id not in sb:
            continue                     # untrained query rows: a different measurement
        crop, _, kp = predict(net, el, sb[el.layer_id], gb.get(el.layer_id, 0))
        _, st = rebuild(el, to_local(el, crop), cfg, kp)
        density = float((el.key_mask & el.live).sum() / max(1, el.live.sum()))
        fires = (float(((kp > args.key_thresh) & el.live).sum() / max(1, el.live.sum()))
                 if has_key else float('nan'))
        row = {'layer_id': el.layer_id, 'key_density': density,
               'shapes': el.n_shapes, 'artist_keys': int((el.key_mask & el.live).sum()),
               'key_ratio': st['keys_predicted'] / max(1, st['keys_artist']),
               'key_f1': st['key_f1'], 'key_f1_strict': st.get('key_f1_strict', 0.0),
               'pipeline_random': st.get('baseline_pipeline_key_f1_random', 0.0),
               'pipeline_skill': st.get('key_f1_over_random', 0.0),
               'head_fires_fraction': fires,
               'head_key_f1': st.get('head_key_f1', 0.0),
               'head_all_live': st.get('baseline_key_f1_all_live', 0.0),
               'head_random': st.get('baseline_key_f1_random', 0.0),
               'head_skill': st.get('head_key_f1_over_best_baseline', 0.0)}
        rows.append(row)
        head = (f'{fires:>6.2f} {row["head_key_f1"]:>6.3f} {row["head_all_live"]:>6.3f} '
                f'{row["head_random"]:>6.3f} {row["head_skill"]:>+6.3f}'
                if has_key else f'{"—":>6} {"—":>6} {"—":>6} {"—":>6} {"—":>6}')
        print(f'{el.layer_id.split("__")[-1][:24]:<24} {density:>8.3f} | '
              f'{row["key_ratio"]:>5.2f}x {row["key_f1"]:>6.3f} '
              f'{row["pipeline_random"]:>6.3f} {row["pipeline_skill"]:>+6.3f} | {head}',
              flush=True)

    kw = np.array([r['artist_keys'] for r in rows], float)
    kw = kw / kw.sum()
    av = lambda k: float(np.dot(kw, [r[k] for r in rows]))
    totals = {'run': args.run, 'dataset': str(dataset), 'has_key_head': has_key,
              'key_thresh': args.key_thresh, 'layers': len(rows),
              'key_f1': av('key_f1'), 'key_f1_strict': av('key_f1_strict'),
              'pipeline_random': av('pipeline_random'),
              'pipeline_skill': av('pipeline_skill'),
              'head_key_f1': av('head_key_f1'), 'head_all_live': av('head_all_live'),
              'head_random': av('head_random'), 'head_skill': av('head_skill'),
              'density_min': min(r['key_density'] for r in rows),
              'density_max': max(r['key_density'] for r in rows),
              'per_layer': rows}
    out = Path(args.out or f'v1.3/results/key_baselines_{args.run}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(totals, indent=2))

    print(f'\nweighted by the artist\'s key count, over {len(rows)} trained layers:')
    print(f'  the pipeline:  key F1 {totals["key_f1"]:.3f}  '
          f'(strict {totals["key_f1_strict"]:.3f})   the same keys at random '
          f'{totals["pipeline_random"]:.3f}   -> {totals["pipeline_skill"]:+.3f} of placement')
    if has_key:
        print(f'  the head:      key F1 {totals["head_key_f1"]:.3f}   all live '
              f'{totals["head_all_live"]:.3f}   random {totals["head_random"]:.3f}   '
              f'-> {totals["head_skill"]:+.3f} over the better of them')
    print(f'  key density spans {totals["density_min"]:.3f} to {totals["density_max"]:.3f} '
          f'({totals["density_max"] / totals["density_min"]:.0f}x) across these layers, which '
          f'is why\n  none of these figures is readable on its own.\n-> {out}')


if __name__ == '__main__':
    main()
