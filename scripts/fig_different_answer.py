"""One picture for the question "could a different-but-valid breakdown be scored as wrong?"

The answer is no, and the reason is that soft IoU never sees the shapes -- it compares two
finished pictures pixel by pixel. This card is the proof, using a real answer that is about as
differently-carved as one could be: ``exp_s3_baselines.py``'s silhouette trace, which throws
away the breakdown entirely and draws one ring around each blob.

Five panels per element:

* **the artist's shapes** -- every shape outlined in its own colour, so the whole
  breakdown is visible at once and the count reads straight off the picture
* **the traced answer's shapes** -- the same treatment, and it is a single ring
* **her cutout** and **its cutout** -- what each of those two breakdowns renders to
* **the difference** -- red where the trace drew what she did not, blue where it missed, grey
  where they agree, with the soft IoU that pair actually scores

The point of the card is the first two panels next to the last one: 247 shapes against 1, and
a score of 0.947. If the metric punished a different decomposition, that number is impossible.

    python scripts/fig_different_answer.py
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

from roto.ir import Key                                                # noqa: E402
from roto.metrics import soft_iou                                      # noqa: E402
from roto.render.raster import config_from_meta, render_union          # noqa: E402
from roto.sfx.json_ir import from_json_ir                              # noqa: E402
from roto.v2.build import load_alpha                                   # noqa: E402
from roto.v2.traindata import load_element                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_s3_baselines import trace, trace_doc                          # noqa: E402

ELEMENTS = [
    ('datasets/v003/ts_021351__Green', 'the extreme: 247 shapes against 1'),
    ('datasets/v003/ts_021182__Lady_and_Man_matte', 'and on a shot held out of training'),
    ('datasets/v003/ts_020028__hair', 'a layer where the trace scores 0.97'),
    ('datasets/v003/ts_020355__Red', 'and one where it scores 0.98'),
]
"""``(element dir, the line under its row)``. Chosen for a wide spread of artist shape counts
against a trace that always collapses to 1-3, because the claim is about the *gap*."""

N_POINTS = 64
"""Matches ``trace_k@64``, the baseline row the S3 design note quotes and the gate was
re-priced against. Using anything else here would illustrate a number nobody has."""

OFF = np.array([[0.0]])
ON = np.array([[100.0]])


def busiest_frame(el) -> int:
    """The frame with the most shapes on screen -- the same reference ``canonical_order`` uses.

    A row drawn on a frame where half the layer is off screen understates the breakdown, which
    is the one thing this card exists to show.
    """
    live_count = el.live.sum(axis=1)
    return int(el.frames[int(np.argmax(live_count))]) if live_count.max() else int(el.frames[0])


def render_at(doc, el, frame: int) -> np.ndarray:
    """Render ``doc`` at ``frame`` through the element's own crop and render conventions."""
    c = el.crop
    cfg = config_from_meta(el.render or {'supersample': c.get('supersample', 2)}, None)
    x0, y0 = c['offsets'][int(frame)]
    box = (x0, y0, c['out_px'] / c['scale'], c['out_px'] / c['scale'])
    return render_union(doc, int(frame), cfg, c['scale'], box)


def mosaic(doc, el, frame: int, shape, scale: int = 3) -> tuple[np.ndarray, int]:
    """Outline every shape of ``doc`` in its own colour at one frame, over a faint union fill.

    Outlines rather than fills, and this is the whole design of the panel. Filling each shape
    in turn paints the later ones over the earlier ones, so a layer of 247 heavily overlapping
    shapes shows about six -- which understates the breakdown, the one thing the card exists
    to show. Outlined, all 247 curves are visible at once and the count reads off the picture.

    Every shape is rendered *alone* -- all opacities down, one back up -- rather than inferred
    from the union, because the union is exactly the thing that destroys this information.
    Drawn at ``scale``x so a 1 px stroke stays legible when 200 of them overlap.

    Returns the RGB image and how many shapes actually drew anything.
    """
    import cv2
    shapes = [s for _, s in doc.shapes()]
    saved = [s.opacity for s in shapes]
    h, w = shape
    H, W = h * scale, w * scale
    union = np.clip(render_at(doc, el, frame)[:h, :w], 0, 1)
    canvas = np.repeat(cv2.resize((union * 44).astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_LINEAR)[..., None], 3, axis=2)
    cmap = matplotlib.colormaps['turbo']
    order = np.random.default_rng(0).permutation(len(shapes))
    drawn = 0
    for s in shapes:
        s.opacity = [Key(int(frame), 'hold', OFF)]
    try:
        for i, s in enumerate(shapes):
            s.opacity = [Key(int(frame), 'hold', ON)]
            a = render_at(doc, el, frame)
            s.opacity = [Key(int(frame), 'hold', OFF)]
            if a.shape != (h, w):
                a = a[:h, :w]
            if a.max() <= 0.01:
                continue
            cnts, _ = cv2.findContours((a > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            cnts = [c * scale for c in cnts if len(c) >= 3]
            if not cnts:
                continue
            drawn += 1
            # Clamped away from both ends of turbo: its extremes are near-black and vanish.
            t = 0.12 + 0.83 * (order[i] + 0.5) / len(shapes)
            col = tuple(int(255 * v) for v in cmap(t)[:3])
            cv2.polylines(canvas, cnts, True, col, 1, cv2.LINE_AA)
    finally:
        for s, o in zip(shapes, saved):
            s.opacity = o
    return canvas.astype(np.float32) / 255.0, drawn


def bbox(truth: np.ndarray, pad: float = 0.09) -> tuple[int, int, int, int]:
    """Square crop around everything on screen, so a small element is not a speck in a panel.

    Applied identically to all five panels of a row, at each one's own resolution, so the
    comparison the row makes is never between two differently-framed pictures.
    """
    ys, xs = np.where(truth > 0.02)
    h, w = truth.shape
    if not len(ys):
        return 0, 0, h, w
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    side = min(int(side * (1 + 2 * pad)), min(h, w))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return y0, x0, side, side


def crop(img: np.ndarray, box, scale: int = 1) -> np.ndarray:
    y0, x0, hh, ww = (v * scale for v in box)
    return img[y0:y0 + hh, x0:x0 + ww]


def difference(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The project's standard card colouring: red drew-too-much, blue missed, grey agreed."""
    rgb = np.zeros(truth.shape + (3,), np.float32)
    rgb[..., 0] = np.clip(pred - truth, 0, 1)
    rgb[..., 2] = np.clip(truth - pred, 0, 1)
    rgb += (np.minimum(pred, truth) * 0.55)[..., None]
    return np.clip(rgb, 0, 1)


def panel(ax, img, title: str, subtitle: str = '', colour: str = '#222') -> None:
    # Single-channel panels are mattes, not data: greyscale, not a colour map.
    ax.imshow(img, cmap='gray', vmin=0, vmax=1) if img.ndim == 2 else ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, weight='bold', color=colour, pad=2)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8, color='#555', labelpad=3)
    for sp in ax.spines.values():
        sp.set_edgecolor('#ccc')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='luthra-understands/different-answer-possibility.jpg')
    args = ap.parse_args()

    rows = []
    for d, caption in ELEMENTS:
        el = load_element(Path(d), with_local=False)
        doc = from_json_ir(json.loads((Path(d) / 'target_ir.json').read_text()))
        frame = busiest_frame(el)
        truth = load_alpha(Path(d), frame)

        rings = trace(truth, N_POINTS, largest_only=False)
        tdoc = trace_doc(el, frame, rings)
        pred = render_at(tdoc, el, frame)
        if pred.shape != truth.shape:
            pred = pred[:truth.shape[0], :truth.shape[1]]

        art_rgb, n_art = mosaic(doc, el, frame, truth.shape)
        tr_rgb, n_tr = mosaic(tdoc, el, frame, truth.shape)
        box = bbox(truth)
        rows.append({
            'id': el.element_id, 'caption': caption, 'frame': frame,
            'art_rgb': crop(art_rgb, box, 3), 'tr_rgb': crop(tr_rgb, box, 3),
            'truth': crop(truth, box), 'pred': crop(pred, box),
            'diff': crop(difference(pred, truth), box),
            'n_art': n_art, 'n_tr': n_tr, 'n_layer': int(el.n_shapes),
            'iou': soft_iou(pred, truth), 'held': not el.in_train,
        })
        print(f'{el.element_id:<34} frame {frame:>3}  {n_art:>3} of {el.n_shapes} shapes '
              f'-> {n_tr}  soft IoU {rows[-1]["iou"]:.4f}', flush=True)

    n = len(rows)
    fig = plt.figure(figsize=(15.0, 2.05 + 2.6 * n))
    gs = fig.add_gridspec(n + 1, 5, height_ratios=[0.62] + [1] * n,
                          hspace=0.30, wspace=0.06, left=0.055, right=0.985,
                          top=0.985, bottom=0.045)

    head = fig.add_subplot(gs[0, :])
    head.axis('off')
    head.text(0.0, 1.0, 'Could a different-but-valid breakdown be marked wrong?',
              fontsize=16, weight='bold', va='top')
    head.text(0.0, 0.60,
              'No — and this is the proof. Every row is the same layer carved two completely '
              'different ways. Left: every shape she drew, outlined in its own colour. Next: a '
              'silhouette trace that throws the\nbreakdown away and draws one ring per blob '
              '— no model, no understanding. They are not remotely the same answer. The two '
              'cutouts they render to are nearly identical, and soft IoU\nsays so, because it '
              'compares finished pictures pixel by pixel and never sees the shapes at all. '
              'That blindness is deliberate: it is what stops us punishing valid roto that '
              'differs from hers.',
              fontsize=10.2, va='top', color='#333', linespacing=1.55)

    for r, row in enumerate(rows, start=1):
        axes = [fig.add_subplot(gs[r, c]) for c in range(5)]
        # Live-on-this-frame in the title, layer total in the caption: the panel can only show
        # the shapes that are on screen, and quoting the layer total alone would oversell it.
        total = '' if row['n_art'] == row['n_layer'] else f' · {row["n_layer"]} in the layer'
        panel(axes[0], row['art_rgb'], f'the artist: {row["n_art"]} shapes',
              f'{row["id"]}, frame {row["frame"]}{total}')
        panel(axes[1], row['tr_rgb'],
              f'the trace: {row["n_tr"]} shape' + ('s' if row['n_tr'] != 1 else ''),
              'outline only — nothing inside it')
        panel(axes[2], row['truth'], 'her cutout', 'what those shapes render to',
              colour='#555')
        panel(axes[3], row['pred'], 'its cutout', 'what the trace renders to', colour='#555')
        panel(axes[4], row['diff'], f'soft IoU  {row["iou"]:.4f}', row['caption'],
              colour='#b00' if row['iou'] < 0.9 else '#080')

    fig.text(0.055, 0.012,
             'grey = the two agree   RED = the trace drew what she did not   '
             'BLUE = the trace missed what she drew        '
             'Two answers can look nothing alike and still score 0.95+. '
             'That is the metric working as designed.',
             fontsize=8.6, color='#444')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor='white')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
