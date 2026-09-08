"""Three-panel comparison figures: artist splines, the alpha the model saw, what it rebuilt.

The panels are deliberately in that order, because it is the order the pipeline runs in and
because the middle panel is the only thing the model is given. Anyone reading the figure
should be able to cover the outer two and see exactly how little information the middle one
carries: a filled silhouette, no curves, no control points, no timing.

Splines are drawn as outlines rather than fills. A filled reconstruction next to a filled
target hides precisely the errors worth seeing -- a control point in the wrong place moves an
edge by a few pixels, which is invisible under fill and obvious under a stroked outline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use('Agg')   # headless: WSL sets DISPLAY, and the Tk backend then fails outright

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..dataset import load_alpha
from ..ir import RotoDoc, opacity_at
from ..render.raster import RenderConfig, shape_polyline

ARTIST_COLOUR = '#4dd0e1'
MODEL_COLOUR = '#ffb74d'
BEFORE_COLOUR = '#ef5350'
BACKGROUND = '#111318'


def polylines(doc: RotoDoc, frame: int, crop: dict, cfg: RenderConfig | None = None
              ) -> list[np.ndarray]:
    """Every live shape at ``frame`` as an (n, 2) polyline in crop pixels.

    Uses the renderer's own ``shape_polyline`` and its own normalised-to-pixel mapping, so a
    drawn outline lands exactly where the rasteriser would have put the edge. Re-deriving the
    mapping here is the obvious shortcut and would make the figure quietly disagree with the
    IoU printed beside it.
    """
    cfg = cfg or RenderConfig(supersample=1)
    x0, y0 = crop['offsets'][int(frame)]
    scale = crop['scale']
    ox = (crop['width'] / 2.0 - x0) * scale
    oy = (crop['height'] / 2.0 - y0) * scale
    hn = crop['height'] * scale

    out = []
    for ancestors, shape in doc.shapes():
        if opacity_at(shape, frame) <= 0.0:
            continue
        poly = shape_polyline(shape, frame, ancestors, cfg)
        if poly is None or len(poly) < 2:
            continue
        xy = np.empty_like(poly)
        xy[:, 0] = poly[:, 0] * hn + ox
        xy[:, 1] = poly[:, 1] * hn + oy
        if shape.closed:
            xy = np.vstack([xy, xy[:1]])
        out.append(xy)
    return out


def _panel(ax, title: str, subtitle: str = '') -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#3a3f4b')
    ax.set_title(title, color='#e8eaed', fontsize=9, pad=6, loc='left')
    if subtitle:
        ax.text(0.99, 1.02, subtitle, transform=ax.transAxes, ha='right', va='bottom',
                color='#9aa0a6', fontsize=8)


def element_figure(layer_dir: str | Path, artist: RotoDoc, model: RotoDoc, frame: int,
                   crop: dict, soft: float, hard: float, out_px: int,
                   layer_id: str, extra: str = '') -> Figure:
    alpha = load_alpha(layer_dir, frame)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.03, wspace=0.06)

    for ax in axes:
        ax.set_facecolor(BACKGROUND)
        ax.set_xlim(0, out_px); ax.set_ylim(out_px, 0)
        ax.set_aspect('equal')

    for xy in polylines(artist, frame, crop):
        axes[0].plot(xy[:, 0], xy[:, 1], color=ARTIST_COLOUR, lw=0.7)
    _panel(axes[0], 'Artist splines', 'ground truth, from the .sfx')

    axes[1].imshow(alpha, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    _panel(axes[1], 'Clean alpha', 'the only thing the model is given')

    for xy in polylines(model, frame, crop):
        axes[2].plot(xy[:, 0], xy[:, 1], color=MODEL_COLOUR, lw=0.7)
    _panel(axes[2], 'Reconstructed splines',
           f'soft IoU {soft:.4f}   IoU {hard:.4f}')

    fig.suptitle(f'{layer_id}    frame {frame}{"    " + extra if extra else ""}',
                 color='#e8eaed', fontsize=10, x=0.02, ha='left', y=0.955)
    return fig


def comparison_figure(rows: Sequence[dict], out_path: str | Path, title: str) -> Path:
    """One page, one row per layer: artist, the alpha, and *two* reconstructions side by side.

    The figure the v1.1 review asks the report to lead with, and the reason is that the
    aggregate hides the result. Eight easy layers dilute two hard ones, so the headline moved
    by +0.051 while the two worst rows moved by +0.039 and +0.038 -- and only on the two hard
    rows is the difference something an artist would call a different quality of work rather
    than a different number. Putting the same frame of the same layer under both models is the
    only presentation in which that is visible instead of argued.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14.5, 3.5 * n), facecolor=BACKGROUND,
                             squeeze=False)
    fig.subplots_adjust(left=0.012, right=0.988, top=1 - 0.75 / (3.5 * n + 1),
                        bottom=0.01, wspace=0.04, hspace=0.30)

    for r, row in enumerate(rows):
        alpha = load_alpha(row['dir'], row['frame'])
        for c in range(4):
            ax = axes[r][c]
            ax.set_facecolor(BACKGROUND)
            ax.set_xlim(0, row['out_px']); ax.set_ylim(row['out_px'], 0)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#3a3f4b')
        for xy in polylines(row['artist'], row['frame'], row['crop']):
            axes[r][0].plot(xy[:, 0], xy[:, 1], color=ARTIST_COLOUR, lw=0.6)
        axes[r][1].imshow(alpha, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        for xy in polylines(row['before'], row['frame'], row['crop']):
            axes[r][2].plot(xy[:, 0], xy[:, 1], color=BEFORE_COLOUR, lw=0.6)
        for xy in polylines(row['after'], row['frame'], row['crop']):
            axes[r][3].plot(xy[:, 0], xy[:, 1], color=MODEL_COLOUR, lw=0.6)

        # Two lines per title. A layer id is 34 characters and the panel is 3.6 inches
        # wide, so a single-line "id · frame · shapes" overruns the panel and collides
        # with the next one's title -- matplotlib does not clip titles to the axes.
        axes[r][0].set_title(f'artist splines\n{row["layer_id"]}',
                             color=ARTIST_COLOUR, fontsize=8, loc='left', pad=4)
        axes[r][1].set_title(f'clean alpha — all the model is given\n'
                             f'frame {row["frame"]}  ·  {row["n_shapes"]} shapes',
                             color='#9aa0a6', fontsize=8, loc='left', pad=4)
        axes[r][2].set_title(f'{row["before_label"]}\nsoft IoU {row["before_soft"]:.4f}',
                             color=BEFORE_COLOUR, fontsize=8, loc='left', pad=4)
        axes[r][3].set_title(f'{row["after_label"]}\nsoft IoU {row["after_soft"]:.4f}'
                             f'   ({row["after_soft"] - row["before_soft"]:+.4f})',
                             color=MODEL_COLOUR, fontsize=8, loc='left', pad=4)

    fig.suptitle(title, color='#e8eaed', fontsize=12, x=0.012, ha='left',
                 y=1 - 0.2 / (3.5 * n + 1))
    out = Path(out_path)
    fig.savefig(out, dpi=140, facecolor=BACKGROUND)
    plt.close(fig)
    return out


def contact_sheet(rows: Sequence[dict], out_path: str | Path, title: str) -> Path:
    """One page: each row is an layer, three panels wide."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n), facecolor=BACKGROUND,
                             squeeze=False)
    fig.subplots_adjust(left=0.015, right=0.985, top=1 - 0.5 / (3.5 * n + 1),
                        bottom=0.01, wspace=0.04, hspace=0.22)

    for r, row in enumerate(rows):
        alpha = load_alpha(row['dir'], row['frame'])
        for c in range(3):
            ax = axes[r][c]
            ax.set_facecolor(BACKGROUND)
            ax.set_xlim(0, row['out_px']); ax.set_ylim(row['out_px'], 0)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#3a3f4b')
        for xy in polylines(row['artist'], row['frame'], row['crop']):
            axes[r][0].plot(xy[:, 0], xy[:, 1], color=ARTIST_COLOUR, lw=0.6)
        axes[r][1].imshow(alpha, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        for xy in polylines(row['model'], row['frame'], row['crop']):
            axes[r][2].plot(xy[:, 0], xy[:, 1], color=MODEL_COLOUR, lw=0.6)

        axes[r][0].set_title(f'{row["layer_id"]}  ·  frame {row["frame"]}',
                             color='#e8eaed', fontsize=8, loc='left', pad=4)
        axes[r][1].set_title('clean alpha (model input)', color='#9aa0a6',
                             fontsize=8, loc='left', pad=4)
        axes[r][2].set_title(f'reconstructed  ·  soft IoU {row["soft"]:.4f}',
                             color='#9aa0a6', fontsize=8, loc='left', pad=4)

    fig.suptitle(title, color='#e8eaed', fontsize=12, x=0.015, ha='left',
                 y=1 - 0.16 / (3.5 * n + 1))
    out = Path(out_path)
    fig.savefig(out, dpi=140, facecolor=BACKGROUND)
    plt.close(fig)
    return out
