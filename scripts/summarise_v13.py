"""Arrange every scored v1.3 run into the tables the plan asks for.

Nothing is computed here that `report_v13.py` did not measure -- this only arranges it, so a
table can never disagree with the run that produced it. What it adds is the four readings the
plan's S2 and S4 ask for by name, each stated against the *measured* noise floor rather than
against a guess:

1. the re-baseline: what the rebuilt dataset is worth, on the layers every run trains on,
2. the key-timing head, in both framings -- the head alone and the pipeline it feeds,
3. the price of the split: the frame holdout, and the query table,
4. the gates, and which one names the next rung.

    python scripts/summarise_v13.py            # -> v1.3/results/results.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = [
    ('v002_control', "v1's configuration, 40k, seed 0 — **the new baseline**"),
    ('v002_control_s1', 'the same, seed 1'),
    ('v002_final', "v1.1's headline configuration, 40k, seed 0 — the shipping candidate"),
    ('v002_final_s1', 'the same, seed 1'),
    ('v002_final_all', 'the same with the split off: all 13 layers, all 1810 frames'),
    ('v002_keytime_probe', '+ the key-timing head, 12k (a sanity rung, not a quality one)'),
    ('v002_keytime_probe2', 'the same with the positive weight balanced per layer'),
    ('v002_keytime', '+ the key-timing head, 40k, seed 0'),
    ('v002_keytime_s1', 'the same, seed 1'),
]

SHIPPING = [
    ('v002_control_ship_refit', "v1's configuration at its own best editable cell"),
    ('v002_final_ship_refit', "v1.1's, at its own — window 1, i.e. **no smoothing**"),
    ('v002_final_s1_ship_refit', 'the same, seed 1'),
    ('v002_final_all_ship_refit', 'the same with the split off'),
    ('v002_keytime_ship_refit_kb0.6', '+ the key head as a tolerance bias, seed 0'),
    ('v002_keytime_s1_ship_refit_kb0.6', 'the same, seed 1'),
    ('v002_keytime_maxf1_savgol_kb0.6_ks0.5',
     '+ the cell that maximises key F1 inside the band — **the shipping row**'),
    ('v002_keytime_s1_maxf1_savgol_kb0.6_ks0.5', 'the same, seed 1'),
]
"""Each run at the operating point *its own sweep* chose, inside the key-economy band. These
are the rows a delivery would use; the table above is each run at v1's default settings, which
is what makes the runs comparable to each other rather than best."""

GATE_ROWS = [
    ('v002_final_gate_e2e_refit', 'the bias-free candidate'),
    ('v002_final_s1_gate_e2e_refit', 'the same, seed 1'),
    ('v002_keytime_gate_e2e_refit_kb0.6_ks0.5', '**the candidate**'),
    ('v002_keytime_s1_gate_e2e_refit_kb0.6_ks0.5', 'the same, seed 1'),
]
"""The gate configuration: the constrained operating point *and* predicted motion end to end,
which is what the plan's S4 gates. Neither the headline nor the shipping row."""

V001_BACKFILL = [
    ('final_long_s1', "v1.1's `final_long` at seed 1, on `datasets/v001` — the backfill"),
]

COLS = ('| run | what | soft IoU | worst layer | worst frame | pt px | p95 px | jitter | '
        'keys | key F1 | strict | vs random | head F1 | held gap | <0.90 | seed |')


def load(res: Path, name: str, suffix: str = '') -> dict | None:
    p = res / f'score_{name}{suffix}.json'
    return json.loads(p.read_text()) if p.exists() else None


def line(name: str, what: str, d: dict) -> str:
    gap = f'{d["held_gap"]:+.4f}' if 'held_gap' in d else '—'
    head = f'{d["head_key_f1"]:.3f}' if d.get('head_key_f1') else '—'
    # A *missing* measurement must not render as a zero: "+0.000" reads as "exactly at
    # baseline", which is a result, and "not measured" is not. Scores written before the
    # baselines existed have to be re-scored, and the table has to say so rather than
    # quietly averaging them in as zeros.
    skill = (f'{d["key_f1_over_random"]:+.3f}' if 'key_f1_over_random' in d else '_n/m_')
    strict = f'{d["key_f1_strict"]:.3f}' if 'key_f1_strict' in d else '_n/m_'
    return (f'| `{name}` | {what} | {d["mean_soft_iou"]:.4f} | '
            f'{d["worst_layer_soft_iou"]:.4f} | {d["worst_frame_soft_iou"]:.4f} | '
            f'{d["point_err_px"]:.2f} | {d["p95_point_err_px"]:.2f} | '
            f'{d["jitter_px"]:.2f} | {d["key_ratio"]:.2f}x | {d["key_f1"]:.3f} | '
            f'{strict} | {skill} | {head} | '
            f'{gap} | {d["frames_below_0.90"]}/{d["frames"]} | {d.get("seed", "?")} |')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='v1.3/results')
    ap.add_argument('--out', default='v1.3/results/results.md')
    args = ap.parse_args()
    res = Path(args.results)

    runs = {n: d for n, _ in ORDER + V001_BACKFILL if (d := load(res, n))}
    cpath = res / 'scoring_ceiling.json'
    ceiling = json.loads(cpath.read_text()) if cpath.exists() else None
    rb = (json.loads((res / 'rebaseline.json').read_text())
          if (res / 'rebaseline.json').exists() else None)
    gates = (json.loads((res / 'gates.json').read_text())
             if (res / 'gates.json').exists() else None)
    noise = (json.loads((res / 'noise_floor.json').read_text())
             if (res / 'noise_floor.json').exists() else {})

    L = ['# v1.3 results', '',
         'Generated by `scripts/summarise_v13.py` from the scored runs. Every number here was',
         'measured by `scripts/report_v13.py` and is only arranged here.', '']

    # ---- 0. the ground ---------------------------------------------------------
    if ceiling:
        q = ceiling.get('worst_frame_ceiling_quantised')
        L += ['## The ground everything stands on', '',
              "The artist's own program, rendered by today's renderer against `datasets/v002`'s",
              'own alphas. This is the number every model number is measured against, and on',
              '`datasets/v001` it was **0.999529 mean / 0.992632 on its worst frame / 247 of',
              '1,810 frames below 0.999**.', '',
              '| | raw float | through the dataset\'s own uint16 round trip |',
              '|---|---|---|',
              f'| mean | {ceiling["mean_ceiling"]:.9f} | '
              f'**{ceiling["mean_ceiling_quantised"]:.9f}** |',
              f'| worst frame | {ceiling["worst_frame_ceiling"]:.9f} | '
              f'**{q:.9f}** |' if q else '| worst frame | — | — |',
              f'| frames short of it | {ceiling["frames_below_0.999"]}/{ceiling["frames"]} '
              f'below 0.999 | {ceiling.get("frames_below_1_quantised", "?")}'
              f'/{ceiling["frames"]} below 1.0 |', '',
              f'The raw-float residual is **exactly** 16-bit PNG quantisation and nothing else:',
              f'the worst pixel disagrees by {ceiling["max_abs_pixel"]:.3e} against a',
              f'half-quantum of {ceiling["half_quantum"]:.3e}. Put the render through the same',
              'uint16 round trip the dataset writes and it is bit-identical on every frame of',
              'every layer.', '']

    # ---- 1. the table ----------------------------------------------------------
    if noise:
        L += ['## The noise floor, because it decides what the table can say', '',
              '| configuration | seeds | soft IoU | worst layer | worst frame | pt px | key F1 |',
              '|---|---|---|---|---|---|---|']
        for label, s in noise.items():
            L.append(f'| {label} | {s["n_runs"]} | ±{s["mean_soft_iou"]["range"]:.4f} | '
                     f'±{s["worst_layer_soft_iou"]["range"]:.4f} | '
                     f'±{s["worst_frame_soft_iou"]["range"]:.4f} | '
                     f'±{s["point_err_px"]["range"]:.2f} | ±{s["key_f1"]["range"]:.3f} |')
        L += ['', 'Ranges, not standard deviations: two seeds do not describe a distribution.',
              '**A difference in the table below that is smaller than its row here is not a',
              'result.**', '']

    L += ['## The runs', '',
          '`datasets/v002`, and the mean is over the **11 layers each run trains on** — the two',
          'the dataset withholds are reported separately below, because a layer with untrained',
          'query rows is a measurement of the encoder rather than a reconstruction. `<0.90` is',
          'frames below 0.90 across the run. `vs random` is the key F1 minus what the *same',
          'number of keys* placed at random over the same live frames would have scored —',
          'quoted because key F1 with a one-frame tolerance is close to saturated by keying',
          'often, and the two densest-keyed layers carry 40% of this archive\'s keys.',
          '`strict` excludes the 7.91% of artist keys that sit outside their own shape\'s live',
          'range, which the established definition clips onto the live boundary where the DP',
          'always forces a knot and so matches for free. `head F1` is the key-timing head',
          'scored on its own, where a run has one.',
          '', COLS, '|' + '---|' * 16]
    for name, what in ORDER:
        if name in runs:
            L.append(line(name, what, runs[name]))
    ship = {n: d for n, _ in SHIPPING if (d := load(res, n))}
    if ship:
        L += ['', '### At each run\'s own shipping operating point', '',
              'The table above scores every run at v1\'s default keyframe settings, which is',
              'what makes the runs comparable. This one scores each at the cell *its own sweep*',
              'chose as the best rendered soft IoU whose key count an artist could still edit.',
              'Teacher-forced motion.', '', COLS, '|' + '---|' * 16]
        for name, what in SHIPPING:
            if name in ship:
                L.append(line(name, what, ship[name]))

    gates_rows = {n: d for n, _ in GATE_ROWS if (d := load(res, n))}
    if gates_rows:
        L += ['', '### At the gate configuration', '',
              'The constrained operating point **and** predicted motion end to end — no',
              'teacher-forced transform track. This is what the plan\'s §4 gates, and it is a',
              'harder configuration than any headline number on purpose: a delivery has neither',
              'the artist\'s motion track nor permission to place keys wherever rendered IoU',
              'likes them.', '', COLS, '|' + '---|' * 16]
        for name, what in GATE_ROWS:
            if name in gates_rows:
                L.append(line(name, what, gates_rows[name]))

    if any(n in runs for n, _ in V001_BACKFILL):
        L += ['', 'And the one rung that belongs on the old dataset, because its whole purpose',
              'is to finish a v1.1 comparison:', '', COLS, '|' + '---|' * 16]
        for name, what in V001_BACKFILL:
            if name in runs:
                L.append(line(name, what, runs[name]))

    # ---- 2. the re-baseline ----------------------------------------------------
    if rb:
        L += ['', '## 1. The re-baseline: what the rebuild was worth', '',
              f'Every past row re-aggregated on the same {len(rb["common_layers"])} layers, from',
              'the per-layer numbers those runs already stored — nothing re-rendered and nothing',
              'retrained. The `as published` column is what each report quoted over its own layer',
              'set and is here only so a row can be found in an old document; the `common` column',
              'is the one that compares.', '',
              '| run | dataset | as published | on the common layers | vs the new baseline | '
              'pt px | p95 px | key F1 | <0.90 |', '|' + '---|' * 9]
        anchor = next((r for r in rb['rows'] if r['run'] == rb['anchor']), None)
        base = anchor['common']['mean_soft_iou'] if anchor else None
        for r in rb['rows']:
            c = r['common']
            pub = r['as_published'].get('mean_soft_iou')
            delta = f'{c["mean_soft_iou"] - base:+.4f}' if base is not None else '—'
            L.append(f'| `{r["run"]}` | {Path(str(r["dataset"])).name} | '
                     f'{pub:.4f} | **{c["mean_soft_iou"]:.4f}** | {delta} | '
                     f'{c["point_err_px"]:.2f} | {c["p95_point_err_px"]:.2f} | '
                     f'{c["key_f1"]:.3f} | {c["frames_below_0.90"]} |')

    # ---- 3. the price of the split ---------------------------------------------
    split_rows = [(n, runs[n]) for n, _ in ORDER if n in runs and runs[n].get('held_layers')]
    if split_rows:
        L += ['', '## 2. The price of the split', '',
              'Two independent splits, both fixed when the dataset was built.', '',
              '**The frame holdout** — every 7th frame, withheld from training and scored',
              'separately. It is a column on every v002 run rather than a separate rung, which',
              'is what "defined at build time" buys.', '',
              '| run | trained frames | held frames | gap |', '|---|---|---|---|']
        for n, d in split_rows:
            if 'held_gap' in d:
                L.append(f'| `{n}` | {d["train_soft_iou"]:.4f} | {d["held_soft_iou"]:.4f} | '
                         f'{d["held_gap"]:+.4f} |')
        L += ['', '**The layer holdout** — two layers withheld from training entirely. This is',
              '*not* a generalisation claim: shape queries are per (layer, shape), so a layer',
              'that never trains has no query rows and is given freshly initialised ones. What it',
              'measures is how much of the headline number lives in the query table rather than',
              'in the encoder — which is the number v2\'s dynamic queries have to beat.', '',
              '| run | trained layers | the 2 withheld layers | point error there |',
              '|---|---|---|---|']
        for n, d in split_rows:
            h = d['held_layers']
            L.append(f'| `{n}` | {d["mean_soft_iou"]:.4f} | **{h["mean_soft_iou"]:.4f}** | '
                     f'{h["point_err_px"]:.1f} px |')
        first = split_rows[0][1]
        L += ['', f'Withheld: {", ".join("`" + x + "`" for x in first["withheld_layers"])}.']

    # ---- 4. the key-timing head ------------------------------------------------
    key_runs = [(n, d) for n, d in runs.items() if d.get('head_key_f1')]
    if key_runs:
        L += ['', '## 3. The key-timing head', '',
              'Two framings, and they answer different questions. **The head alone** is its own',
              'key set at its own threshold, scored against the artist\'s with a one-frame',
              'tolerance. **The pipeline** is the keys the DP actually chose, with the head\'s',
              'probability biasing its tolerance locally. The second is the deliverable; the',
              'first says whether a disappointing second is the head\'s fault or the bias\'s.', '',
              '| run | bias | the head alone (P / R / F1) | vs the best trivial baseline | '
              'the pipeline\'s key F1 | keys | soft IoU |', '|' + '---|' * 7]
        for n, d in sorted(key_runs):
            vs = (f'**{d["head_key_f1_over_best_baseline"]:+.3f}**'
                  if 'head_key_f1_over_best_baseline' in d else '_not measured_')
            L.append(f'| `{n}` | {d["rebuild"]["key_bias"]:.2f} | '
                     f'{d["head_key_precision"]:.2f} / {d["head_key_recall"]:.2f} / '
                     f'{d["head_key_f1"]:.3f} | {vs} | '
                     f'{d["key_f1"]:.3f} | '
                     f'{d["key_ratio"]:.2f}x | {d["mean_soft_iou"]:.4f} |')
        L += ['', 'The `vs baseline` column is the only one of these that is a statement about',
              'the head: firing on every live frame, or on a random subset of the same size,',
              'already scores what the raw figure shows. Key density ranges 13x across these',
              'layers, so a head that learns nothing but each layer\'s base rate scores well.',
              '', 'For context, the best key F1 recorded before this round was **0.408**, from',
              'v1.2\'s constrained operating point with no retraining and no new signal —',
              'quoted, like every key F1 before v1.3, without a baseline beside it.']

    # ---- 5. the gates ----------------------------------------------------------
    if gates:
        L += ['', '## 4. The acceptance gates', '',
              f'`{gates["candidate"]}`, {gates["seeds_found"]} seed(s), at the gate',
              f'configuration (`{gates["configuration"]}`): the constrained operating point',
              'with predicted motion end to end. Not the headline configuration — the',
              'plan gates a *delivery*, which has neither the artist\'s motion track nor',
              'permission to place keys wherever rendered IoU likes them.', '',
              '| | gate | measured | requirement |', '|---|---|---|---|']
        for r in gates['rows']:
            mark = {'PASS': '**PASS**', 'FAIL': '**FAIL**', 'PENDING': '_pending_'}[r['status']]
            L.append(f'| {mark} | {r["gate"]} | `{r["measured"]}` | {r["requirement"]} |')
        L += ['', f'{gates["passed"]} passed, {gates["failed"]} failed, '
                  f'{gates["pending"]} pending.']

    Path(args.out).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
