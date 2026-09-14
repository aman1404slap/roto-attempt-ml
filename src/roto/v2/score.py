"""Score a v2 checkpoint into the run table charter S5 reads.

One row per element, then three aggregates: what the run **trained on** (the headline, and what
the gates read), what it was **withheld from**, and everything. Charter L3 governs the reading --
worst case, never averages alone -- so every aggregate carries its per-element floor and its
frames-below-threshold count.

**A withheld element gets freshly initialised query rows, not another element's.** Shape queries
are per ``(element, shape)``, so an element the run never saw has no rows in the table and there
is no honest default. Reusing another element's rows -- which any ``dict.get(name, 0)`` does
silently -- measures nothing at all: a vector trained to mean "shape 7 of that element" applied
to a different element's shape 7. Allocating fresh rows measures something specific instead:
**what the encoder alone produces, with the memorisation capacity set to zero**. That is the
number charter S4's S3 has to beat, and :func:`roto.v2.reconstruct.untrained_queries` is where
it comes from.

**Soft IoU is reported twice, and the second one is the honest one.** 215 of this dataset's
1,412 frames render *nothing* -- the layer is not on screen at all. On those frames the artist
drew nothing, the model draws nothing, and ``metrics.soft_iou`` returns exactly 1.0. That is the
right convention for scoring one frame (predicting nothing where there is nothing *is* correct)
and the wrong one for an average, because it hands out a free mark in proportion to how often a
layer is absent -- which varies enormously: ``ts_020028__hair`` is absent on 79 of 94 frames and
``ts_021351__Green`` on none of 42. Measured on the S0 anchor the inflation is **+0.0094**,
twelve times the seed spread, and per element it reaches 0.846 vs 0.660.

So every soft-IoU aggregate carries an ``_onscreen`` twin computed over the frames where the
target actually draws something. No threshold is involved and none is needed: coverage is either
**exactly zero** or at least 2.35% of the crop, with nothing in between.

**The dataset is never guessed.** A checkpoint records what it trained on; scoring it against
different alphas would read as a quality loss rather than as the mistake it is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .build import element_dirs
from .reconstruct import (RebuildConfig, assemble, load_model, predict, untrained_queries)
from .report import by_training_status
from .splits import load_splits
from .traindata import load_element


def score_run(checkpoint: str | Path, dataset: str | Path | None = None,
              cfg: RebuildConfig | None = None, seed: int = 0) -> dict[str, Any]:
    """Every element of ``dataset`` through ``checkpoint``, plus the aggregates."""
    net, ck = load_model(checkpoint)
    sbase = dict(ck.get('shape_base', {}))
    gbase = dict(ck.get('group_base', {}))
    withheld = list(ck.get('withheld_elements', []))
    splits = ck.get('splits', {})
    root = Path(dataset or ck.get('dataset') or '')
    if not root.name:
        raise SystemExit(f'{checkpoint} records no dataset; pass one explicitly rather than '
                         'scoring it against alphas it was not trained on')
    cfg = cfg or RebuildConfig()

    held_shots = set((load_splits(root) or {}).get('held_shots', []))
    rows, frozen_ids = [], []
    for d in element_dirs(root):
        el = load_element(d)
        # An element is frozen if the checkpoint says so **or** if it simply has no query
        # rows -- an element added to the dataset after the run, say. Both cases are scored
        # the same way (fresh rows), so both must be *reported* the same way. Splitting the
        # table on the checkpoint's list alone lets the second case land in the headline
        # block, where one element reconstructing from noise drags the worst-element floor to
        # near zero and the failure looks like a model result.
        # **Charter S4 stage S3 changes what "withheld" can mean.** With a per-element query
        # table an unseen element has no rows and there is no honest default, so it is scored
        # with freshly initialised ones and reported apart -- a floor on the encoder, never a
        # generalisation claim. With a *shared slot bank* that problem disappears: the bank is
        # trained and applies to any element, so a held-out shot is scored with the real
        # model and its number is a real generalisation number. That is most of what S3 is
        # for, and collapsing the two cases would throw it away.
        shared = getattr(net, 'query_mode', 'table') == 'slots'
        frozen = (not shared) and (el.element_id in set(withheld)
                                   or el.element_id not in sbase)
        if frozen:
            frozen_ids.append(el.element_id)
        sb, gb = (untrained_queries(net, el, seed=seed) if frozen
                  else (sbase.get(el.element_id, 0), gbase.get(el.element_id, 0)))
        p = predict(net, el, sb, gb)
        rec = assemble(el, p.points, p.affine, cfg, key_prob=p.key_prob,
                       alive_prob=p.alive_prob, point_count=p.count)
        s = rec.summary()
        # With shared slots an element can be *scored* properly and still never have been
        # trained on, so the two facts stop being the same fact and the table needs both.
        s['in_train'] = el.element_id not in set(withheld) and el.element_id in sbase
        s['scored_with_real_queries'] = not frozen
        if p.slots_alive is not None:
            # The invented shape count, against the artist's. The style constraint the S3
            # gate amendment turns into a hard requirement: a model that renders well with
            # 1.2 shapes has found the traced-silhouette hole, not done the task.
            s['slots_alive'] = int(p.slots_alive)
            s['shape_count_ratio'] = p.slots_alive / max(1, el.n_shapes)
        # v2's own column: which held elements are held because their whole *shot* is. A shot
        # holdout is the only one of the three splits that is a generalisation claim, so it has
        # to be separable from an element holdout in the row rather than inferred from the name.
        s['shot'] = el.element_id.split('__')[0]
        s['held_shot'] = s['shot'] in held_shots
        s['soft_iou_per_frame'] = [round(float(x), 6) for x in rec.soft_iou]
        # On-screen frames: the target renders something. Exactly zero vs >= 2.35% coverage,
        # so this is a fact about the frame rather than a threshold somebody picked.
        cov = np.asarray(json.loads((d / 'meta.json').read_text())['frames']['coverage'])
        on = cov > 0.0
        s['frames_onscreen'] = int(on.sum())
        s['frames_offscreen'] = int((~on).sum())
        if on.any():
            s['mean_soft_iou_onscreen'] = float(rec.soft_iou[on].mean())
            s['min_soft_iou_onscreen'] = float(rec.soft_iou[on].min())
            s['frames_below_0.90_onscreen'] = int((rec.soft_iou[on] < 0.90).sum())
        rec_split = splits.get(el.element_id, {})
        held_pos = np.array(rec_split.get('held', []), int)
        if len(held_pos) and not frozen:
            train_pos = np.array(rec_split['train'], int)
            s['held_soft_iou'] = float(rec.soft_iou[held_pos].mean())
            s['train_soft_iou'] = float(rec.soft_iou[train_pos].mean())
            s['held_frames'] = int(len(held_pos))
            # The same split applied to the lifespan head, which is the only thing that
            # separates a head that learned to *see* a lifespan from one that memorised what
            # each training frame looks like. Worth asking loudly here: 61.4% of this
            # dataset's lifespan boundaries are invisible in the union alpha
            # (scripts/exp_s2_visibility.py), so a near-perfect training accuracy has a
            # cheaper explanation than perception.
            if rec.alive_pred is not None:
                from .reconstruct import lifespan_stats
                for tag, pos in (('held', held_pos), ('train', train_pos)):
                    st = lifespan_stats(rec.alive_pred[pos], el.live[pos])
                    s[f'{tag}_lifespan_accuracy'] = st['lifespan_accuracy']
                    s[f'{tag}_lifespan_fn_rate'] = st['lifespan_fn_rate']
                    s[f'{tag}_lifespan_fp_rate'] = st['lifespan_fp_rate']
        rows.append(s)

    # Split on **what the run trained on**, not on how the element happened to be scored.
    # Through S2 those were the same fact: an element withheld from training had no query rows,
    # so it was necessarily scored with fresh ones. S3's shared slot bank separates them -- a
    # held-out shot is now scored with the real model -- and splitting on the old field put
    # four untrained elements straight into the headline block, where they dragged the mean
    # from 0.96 to 0.80 and read as a model collapse.
    not_trained = [r['element_id'] for r in rows if not r['in_train']]
    out = by_training_status(rows, not_trained)
    out['rows'] = rows
    out['checkpoint'] = str(checkpoint)
    out['dataset'] = str(root)
    out['held_shots'] = sorted(held_shots)
    out['held_shot_elements'] = sorted(r['element_id'] for r in rows if r['held_shot'])
    out['frozen_elements'] = sorted(frozen_ids)
    if sorted(frozen_ids) != sorted(withheld):
        out['frozen_disagreement'] = {
            'checkpoint_says': sorted(withheld), 'actually_frozen': sorted(frozen_ids)}
    return out


def frozen_table(runs: Sequence[dict[str, Any]], label: str = 'v2 S0 baseline') -> str:
    """The table whose format ``v2_implementation_plan.md`` S2 freezes.

    Charter L4 governs how it is read: two seeds minimum, and a difference smaller than the
    spread between them is weather. The spread column is printed for every row so that is
    checkable rather than asserted, and a one-seed table says so in its header instead of
    quietly looking like a mean.

    Three blocks, because one aggregate cannot carry them (see :func:`roto.v2.report
    .by_training_status`). The **trained** block is the headline and what the gates read. The
    **held-out shot** block is the only generalisation claim in the table. The
    **encoder-only** block is what the withheld elements score with freshly initialised query
    rows -- not a generalisation number, but the floor charter S4's S3 has to beat.
    """
    def col(key: str, block: str | None = None):
        vals = []
        for r in runs:
            src = r if block is None else (r.get(block) or {})
            if key in src and src[key] is not None:
                vals.append(float(src[key]))
        if not vals:
            return None, None
        return float(np.mean(vals)), float(max(vals) - min(vals))

    n = len(runs)
    head = (f'{n} seeds, mean and spread' if n > 1
            else 'ONE SEED -- not a mean; charter L4 wants two before this is quotable')
    lines = [f'{label} -- {runs[0]["dataset"]} -- {head}', '',
             f'{"":38s} {"value":>10s} {"spread":>9s}']

    def section(title, rows, block=None):
        lines.append('')
        lines.append(title)
        lines.append('-' * 59)
        for label, key, fmt in rows:
            m, s = col(key, block)
            if m is None:
                lines.append(f'{label:38s} {"--":>10s}')
            else:
                lines.append(f'{label:38s} {m:>10.{fmt}f} {s:>9.{fmt}f}')

    section('TRAINED ELEMENTS (headline; the gates read this)', [
        ('soft IoU, mean', 'mean_soft_iou', 4),
        ('  ...on-screen frames only', 'mean_soft_iou_onscreen', 4),
        ('soft IoU, worst element', 'worst_element_soft_iou', 4),
        ('  ...on-screen frames only', 'worst_element_soft_iou_onscreen', 4),
        ('soft IoU, worst frame', 'worst_frame_soft_iou', 4),
        ('frames < 0.90, count', 'frames_below_0.90', 1),
        ('  ...on-screen frames only', 'frames_below_0.90_onscreen', 1),
        ('frames < 0.90, worst element %', 'worst_element_frames_below_0.90_pct', 2),
        ('point error, mean px', 'point_err_px', 3),
        ('point error, p95 px', 'p95_point_err_px', 3),
        ('point error, max px', 'max_point_err_px', 2),
        ('jitter px', 'jitter_px', 3),
        ('key ratio (x artist)', 'key_ratio', 3),
        ('key F1 (constrained)', 'key_f1', 4),
        ('key F1 over random baseline', 'key_f1_over_random', 4),
        ('held-frame gap (train - held)', 'held_gap', 4),
    ])
    # Charter S4 stage S2. Printed only when a run carries them, so an S0 or S1 table is
    # unchanged row for row and the frozen format is extended rather than rewritten.
    if any((r.get('trained') or r).get('lifespan_cells') for r in runs):
        section('LIFESPAN (S2) -- predicted, not given', [
            ('cell accuracy', 'lifespan_accuracy', 4),
            # Never one error rate: an omission costs 3.2x an intrusion at the render, so
            # two heads at one accuracy can sit either side of the gate.
            ('  false positive rate (drew it)', 'lifespan_fp_rate', 4),
            ('  false negative rate (missed it)', 'lifespan_fn_rate', 4),
            ('worst element, FN rate', 'lifespan_worst_fn_rate', 4),
            # The line that separates learning from memorising. 61.4% of this dataset's
            # lifespan boundaries are invisible in the union alpha, so a high trained-frame
            # accuracy needs this column beside it before it means anything.
            ('  accuracy, trained frames', 'train_lifespan_accuracy', 4),
            ('  accuracy, held-out frames', 'held_lifespan_accuracy', 4),
            ('  held gap (train - held)', 'lifespan_held_gap', 4),
            # The economy column, and it reads like key economy: a head that flickers makes
            # far more transitions than the artist and can still score well on accuracy.
            ('transition ratio (x artist)', 'lifespan_transition_ratio', 3),
        ])
    if any((r.get('trained') or r).get('point_count_exact') is not None for r in runs):
        section('POINT COUNT (S2) -- REPORTED, NOT GATED', [
            ('exact match', 'point_count_exact', 4),
            # Only one direction is dangerous: a count above the artist's reads point-head
            # slots the point term never supervised. See RebuildConfig.point_count.
            ('  predicted over (unsupervised slots)', 'point_count_over', 1),
            ('  predicted under', 'point_count_under', 1),
        ])
    shared = any(r.get('scored_with_real_queries') and not r['in_train']
                 for r in (runs[0].get('rows') or []))
    section('HELD-OUT ELEMENTS -- ' + ('scored with the SHARED SLOT BANK: a real '
                                       'generalisation number' if shared
                                       else 'encoder only, query table zeroed'), [
        ('soft IoU, mean', 'mean_soft_iou', 4),
        ('point error, mean px', 'point_err_px', 3),
        # The decisive test for the point-count head, and the only honest one available at
        # S2: these elements have freshly initialised query rows, so there is no memory to
        # read the count out of and it has to come from the encoder. Against the trained
        # block's figure, the difference *is* the memorisation.
        ('point count exact (no memory to read)', 'point_count_exact', 4),
        ('lifespan cell accuracy', 'lifespan_accuracy', 4),
    ], block='held_elements')

    on, off = col('frames_onscreen'), col('frames_offscreen')
    if on[0] is not None:
        lines.append('')
        lines.append(f'frames scored: {on[0]:.0f} on-screen, {off[0]:.0f} with the layer absent '
                     f'({100 * off[0] / (on[0] + off[0]):.1f}%).')
        lines.append('  An absent frame scores exactly 1.0 (nothing drawn, nothing expected),')
        lines.append('  which is right per frame and inflates any average over frames.')
    lines.append('')
    lines.append(f'held-out shots: {runs[0]["held_shots"]}')
    lines.append(f'  elements: {", ".join(runs[0]["held_shot_elements"])}')
    if shared:
        lines.append('  These were never trained on, and under S3\'s shared slot bank they are')
        lines.append('  scored with the real model -- the first true generalisation number here.')
    else:
        lines.append('  These have no query rows, so their number measures the encoder alone --')
        lines.append('  a floor for charter S4 stage S3, not a generalisation claim.')
    return '\n'.join(lines)
