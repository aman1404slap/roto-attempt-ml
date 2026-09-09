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


def worst_frame_figure(rows: Sequence[dict], out_path: str | Path, title: str) -> Path:
    """One row per layer: the alpha, the worst frame as an overlay, a typical frame as an
    overlay, and the layer's whole per-frame trace with its ceiling.

    v1.1's figures pick a seeded random frame, which is the right default for a contact sheet
    and the wrong one for the question the handover asks. A mean of 0.97 with a frame at 0.41
    is a rejected shot, and the only way to know whether that frame is a real failure or a
    thin-coverage artefact is to look at it -- next to a frame of the same layer that works,
    and next to the trace that says whether it is one frame or a third of the track.

    Artist and model are drawn *on the same axes* here rather than side by side. On a frame
    that failed, what matters is which contour went where, and that is a comparison the eye
    cannot make across two panels.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.6 * n), facecolor=BACKGROUND,
                             squeeze=False)
    fig.subplots_adjust(left=0.015, right=0.985, top=1 - 0.8 / (3.6 * n + 1),
                        bottom=0.05, wspace=0.10, hspace=0.32)

    for r, row in enumerate(rows):
        for c in range(3):
            ax = axes[r][c]
            ax.set_facecolor(BACKGROUND)
            ax.set_xlim(0, row['out_px']); ax.set_ylim(row['out_px'], 0)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#3a3f4b')

        axes[r][0].imshow(load_alpha(row['dir'], row['worst_frame']), cmap='gray',
                          vmin=0, vmax=1, interpolation='nearest')
        axes[r][0].set_title(f'clean alpha — all the model is given\n{row["layer_id"]}',
                             color='#9aa0a6', fontsize=8, loc='left', pad=4)

        for col, (frame, soft, tag) in enumerate(
                [(row['worst_frame'], row['worst_soft'], 'worst frame'),
                 (row['typical_frame'], row['typical_soft'], 'median frame')], start=1):
            ax = axes[r][col]
            for xy in polylines(row['artist'], frame, row['crop']):
                ax.plot(xy[:, 0], xy[:, 1], color=ARTIST_COLOUR, lw=0.6, alpha=0.9)
            for xy in polylines(row['model'], frame, row['crop']):
                ax.plot(xy[:, 0], xy[:, 1], color=MODEL_COLOUR, lw=0.6, alpha=0.9)
            ax.set_title(f'{tag} {frame} — artist over model\nsoft IoU {soft:.4f}'
                         f'   coverage {row[f"{tag.split()[0]}_coverage"] * 100:.1f}%',
                         color='#e8eaed', fontsize=8, loc='left', pad=4)

        ax = axes[r][3]
        ax.set_facecolor(BACKGROUND)
        ax.plot(row['frames'], row['soft_per_frame'], color=MODEL_COLOUR, lw=0.8)
        if row.get('ceiling_per_frame') is not None:
            ax.plot(row['frames'], row['ceiling_per_frame'], color=ARTIST_COLOUR, lw=0.8,
                    alpha=0.7)
        ax.axvline(row['worst_frame'], color='#e06c75', lw=0.8, ls=':')
        ax.axhline(0.95, color='#9aa0a6', lw=0.6, ls='--')
        ax.set_ylim(min(0.9, float(np.min(row['soft_per_frame'])) - 0.01), 1.005)
        ax.tick_params(colors='#9aa0a6', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#3a3f4b')
        below = int((np.asarray(row['soft_per_frame']) < 0.95).sum())
        ax.set_title(f'per-frame soft IoU  ·  model, and the artist\'s own ceiling\n'
                     f'{below} of {len(row["frames"])} frames below 0.95',
                     color='#9aa0a6', fontsize=8, loc='left', pad=4)

    fig.suptitle(title, color='#e8eaed', fontsize=12, x=0.015, ha='left',
                 y=1 - 0.22 / (3.6 * n + 1))
    out = Path(out_path)
    fig.savefig(out, dpi=140, facecolor=BACKGROUND)
    plt.close(fig)
    return out
