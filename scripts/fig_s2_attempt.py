"""One picture per S2 training attempt, for the attempt log.

The point of this script is accountability rather than diagnosis. Every rung and every retry
costs about 45 minutes of GPU, and a log that records only the ones that worked cannot justify
the ones that did not. So each attempt gets a card: what changed, what it cost, what the
numbers did against the attempt before it, and what the pictures look like.

The card has three parts:

* **the change** -- one line, plus the run's own recorded configuration delta
* **the numbers** -- the gate row and the S2 rows, each against the previous attempt, with the
  two-seed spread beside them so a difference smaller than the spread reads as weather
  (charter L4) rather than as progress
* **the pictures** -- for the elements that matter, the artist's matte, ours, and the
  difference between them. Chosen worst-first per charter L3, because the mean is the number
  that flatters and the floor is the number that gets a shot sent back.

    python scripts/fig_s2_attempt.py --run s2a --against s1
    python scripts/fig_s2_attempt.py --run s2b --against s2a --lifespan predicted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.v2.build import load_alpha                                   # noqa: E402
from roto.v2.reconstruct import (ARTIST, PREDICTED, RebuildConfig,     # noqa: E402
                                 assemble, load_model, predict, score_doc)
from roto.v2.traindata import load_element                             # noqa: E402

ROWS = [
    ('soft IoU, on-screen', 'mean_soft_iou_onscreen', 4, True),
    ('worst element, on-screen', 'worst_element_soft_iou_onscreen', 4, True),
    ('frames < 0.90', 'frames_below_0.90', 0, False),
    ('point error, mean px', 'point_err_px', 3, False),
    ('point error, p95 px', 'p95_point_err_px', 3, False),
    ('key ratio (x artist)', 'key_ratio', 3, None),
    ('key F1 over random', 'key_f1_over_random', 4, True),
    ('held-frame gap', 'held_gap', 4, False),
    ('lifespan accuracy', 'lifespan_accuracy', 4, True),
    ('  false positives', 'lifespan_fp_rate', 4, False),
    ('  false negatives (3.2x cost)', 'lifespan_fn_rate', 4, False),
    # The pair that separates learning from memorising. 61% of this dataset's lifespan
    # boundaries are invisible in the union alpha, so a high trained-frame accuracy has a
    # cheaper explanation than perception and must never be shown without its held twin.
    ('  on frames it trained on', 'train_lifespan_accuracy', 4, True),
    ('  on frames it never saw', 'held_lifespan_accuracy', 4, True),
    ('  the gap between those', 'lifespan_held_gap', 4, False),
    ('transition ratio (x artist)', 'lifespan_transition_ratio', 3, None),
    ('point count exact', 'point_count_exact', 4, True),
]
"""``(label, key, decimals, higher_is_better)``. ``None`` means neither -- a ratio whose target
is 1.0, where both directions are wrong and an arrow would lie about which."""


def load(name: str) -> list[dict] | None:
    p = Path('runs/v2') / f'{name}_summary.json'
    return json.loads(p.read_text()) if p.exists() else None


def agg(runs: list[dict], key: str):
    vals = [float(r[key]) for r in runs if r.get(key) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(max(vals) - min(vals))


def visible_rows(runs):
    """Which rows this attempt actually has. S2A carries eight, S2B sixteen -- the header has
    to grow with them or the table runs over the pictures."""
    return [r for r in ROWS if agg(runs, r[1])[0] is not None]


def numbers_panel(ax, runs, prev, prev_name, title='', subtitle=''):
    """Title, cost line and the number table, all in one axes so they share a left edge."""
    ax.axis('off')
    n = len(visible_rows(runs))
    dy = 1.0 / (n + 4.2)          # +4.2: title, subtitle, header row and the footnote
    y = 1.0
    if title:
        ax.text(0.0, y, title, fontsize=13, va='top', weight='bold')
        y -= dy * 0.95
        ax.text(0.0, y, subtitle, fontsize=8.5, va='top', color='#666')
        y -= dy * 1.15
    ax.text(0.0, y, f'{"":30s}{"now":>10s}{"spread":>9s}{"was":>10s}{"delta":>10s}',
            family='monospace', fontsize=8.5, va='top', color='#444')
    for label, key, dec, higher in ROWS:
        m, s = agg(runs, key)
        if m is None:
            continue
        y -= dy
        was, _ = agg(prev, key) if prev else (None, None)
        d = '' if was is None else f'{m - was:+.{dec}f}'
        colour = '#222'
        if was is not None and higher is not None and abs(m - was) > (s or 0):
            # Coloured only when the move is larger than the seed spread. Inside the spread
            # it is weather, and colouring it would be the exact mistake charter L4 forbids.
            good = (m > was) if higher else (m < was)
            colour = '#1a7f37' if good else '#b3261e'
        ax.text(0.0, y, f'{label:30s}{m:>10.{dec}f}{(s or 0):>9.{dec}f}'
                        f'{"" if was is None else f"{was:>10.{dec}f}"}{d:>10s}',
                family='monospace', fontsize=8.5, va='top', color=colour)
    y -= dy
    if prev:
        ax.text(0.0, y, f'"was" = {prev_name}.  Coloured only where the move exceeds the '
                        'seed spread; inside it is weather.',
                fontsize=7.5, va='top', color='#666', style='italic')


def picture_row(axes, el, rec, frames):
    """Artist matte, ours, and the difference, for one element."""
    doc = rec.doc
    ours, _ = score_doc(el, doc, frames)
    for col, f in enumerate(frames):
        truth = load_alpha(el.directory, int(f))
        pred = None
        # Re-render this one frame through the same path the score used.
        from roto.render.raster import config_from_meta, render_union
        c = el.crop
        cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, None)
        x0, y0 = c['offsets'][int(f)]
        box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
        pred = render_union(doc, int(f), cfg, c['scale'], box)
        if pred.shape != truth.shape:
            pred = pred[:truth.shape[0], :truth.shape[1]]
        # Red where we drew and should not have, blue where we missed. The two colours are
        # the two error kinds the whole S2 design is about, and they cost 1 : 3.2.
        rgb = np.zeros(truth.shape + (3,), np.float32)
        rgb[..., 0] = np.clip(pred - truth, 0, 1)
        rgb[..., 2] = np.clip(truth - pred, 0, 1)
        rgb += (np.minimum(pred, truth) * 0.55)[..., None]
        ax = axes[col]
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        i = int(np.where(el.frames == f)[0][0])
        ax.set_title(f'frame {int(f)}   {rec.soft_iou[i]:.3f}', fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#ccc')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--against', default=None, help='previous attempt to show deltas against')
    ap.add_argument('--change', default='', help='one line: what changed in this attempt')
    ap.add_argument('--seed', type=int, default=1, help='which seed the pictures come from')
    ap.add_argument('--elements', type=int, default=3, help='how many elements to picture')
    ap.add_argument('--held', action='store_true',
                    help='picture the elements the run never trained on instead of the worst '
                         'trained ones. From S3 on these are scored with the real model, so '
                         'this is the generalisation picture rather than a floor')
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--dataset', default='datasets/v003')
    ap.add_argument('--lifespan', choices=(ARTIST, PREDICTED), default=ARTIST)
    ap.add_argument('--point-count', choices=(ARTIST, PREDICTED), default=ARTIST)
    ap.add_argument('--alive-on', type=float, default=0.5)
    ap.add_argument('--alive-off', type=float, default=0.2)
    ap.add_argument('--alive-min-gap', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs = load(args.run)
    if not runs:
        raise SystemExit(f'no runs/v2/{args.run}_summary.json -- score the run first with '
                         f'scripts/score_v2.py {args.run}')
    prev = load(args.against) if args.against else None

    # Worst-first, per charter L3: the floor is the number that gets a shot sent back.
    rows = [r for r in runs[0]['rows'] if r['in_train'] is not args.held]
    worst = sorted(rows, key=lambda r: r.get('mean_soft_iou_onscreen',
                                             r['mean_soft_iou']))[:args.elements]

    cfg = RebuildConfig(lifespan=args.lifespan, point_count=args.point_count,
                        alive_on=args.alive_on, alive_off=args.alive_off,
                        alive_min_gap=args.alive_min_gap)
    net, ck = load_model(Path('runs/v2') / f'{args.run}_seed{args.seed}' / 'model.pt')
    sbase, gbase = dict(ck.get('shape_base', {})), dict(ck.get('group_base', {}))

    n_el = len(worst)
    # The header is sized from the rows it will actually print, in inches, so the line
    # spacing stays constant whether an attempt has eight rows or sixteen.
    head_in = 0.42 + 0.163 * (len(visible_rows(runs)) + 4)
    fig = plt.figure(figsize=(12, head_in + 2.4 * n_el))
    gs = fig.add_gridspec(1 + n_el, args.frames,
                          height_ratios=[head_in] + [2.4] * n_el, hspace=0.26, wspace=0.05)
    head = fig.add_subplot(gs[0, :])

    train = json.loads((Path('runs/v2') / f'{args.run}_seed{args.seed}'
                        / 'train_log.json').read_text())
    mins = train['wall_clock_s'] / 60
    decode = ('' if args.lifespan == ARTIST and args.point_count == ARTIST
              else f'   decode: lifespan={args.lifespan}, point_count={args.point_count}'
                   + (f', on/off/gap {args.alive_on}/{args.alive_off}/{args.alive_min_gap}'
                      if args.lifespan == PREDICTED else ''))
    numbers_panel(head, runs, prev, args.against or '',
                  title=f'{args.run.upper()}   {args.change}',
                  subtitle=(f'{len(runs)} seed(s) x {train["steps"]} steps, '
                            f'{mins:.0f} min each on {train.get("device_name") or "cpu"}'
                            f'{decode}'))

    for r, row in enumerate(worst):
        el = load_element(Path(args.dataset) / row['element_id'])
        p = predict(net, el, sbase.get(el.element_id, 0), gbase.get(el.element_id, 0))
        rec = assemble(el, p.points, p.affine, cfg, key_prob=p.key_prob,
                       alive_prob=p.alive_prob, point_count=p.count)
        # Worst frames of the worst elements, on-screen only -- an absent frame scores a free
        # 1.0 and would picture as an empty square.
        on = np.where(el.live.any(axis=1))[0]
        order = on[np.argsort(rec.soft_iou[on])][:args.frames]
        frames = [int(el.frames[i]) for i in sorted(order)]
        axes = [fig.add_subplot(gs[1 + r, c]) for c in range(args.frames)]
        picture_row(axes, el, rec, frames)
        axes[0].set_ylabel(f'{row["element_id"]}\n{el.n_shapes} shapes',
                           fontsize=8, rotation=0, ha='right', va='center', labelpad=8)

    fig.text(0.02, 0.004, 'grey = we and the artist agree   '
                           'RED = we drew what she did not   '
                           'BLUE = we missed what she drew   '
                           '(worst frames of the worst elements)',
             fontsize=8, color='#444')
    out = Path(args.out or f'luthra-understands/s2-attempt-{args.run}.jpg')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
