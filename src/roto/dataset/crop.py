"""Fixed-size crop window that tracks the layer across the frame.

The obvious choice -- one static crop covering the layer's union bbox over all frames --
fails badly on anything that travels: ``nfl_0200``'s person-and-chair is a ~500x380px object
crossing the entire 2880px frame, so its union bbox is larger than the frame and the object
ends up occupying 1% of the crop. The opposite extreme, a crop refitted per frame, would make
the *scale* time-varying and point distances incomparable between frames.

So the window size, and therefore the scale, are constant, and only the translation varies
per frame. The offsets are data the model is given, not motion it has to infer; the layer's
real motion stays where it belongs, in the layer transform track.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..ir import RotoDoc
from ..render.raster import RenderConfig, content_bbox, render_union


@dataclass(slots=True)
class CropConfig:
    size: int = 512
    """Output alpha is ``size`` x ``size``."""
    supersample: int = 4
    """Higher than the render default of 2 because a crop is cheap: 512 at ss=4 is 2048."""
    margin: float = 0.08
    """Padding around the layer's bbox, as a fraction of its longer side."""
    mode: str = 'tracking'
    """``'tracking'``: constant window size, per-frame offset. ``'static'``: one fixed box."""
    min_crop_px: int = 64
    bbox_scale: float = 0.25
    """Scale for the cheap bbox survey pass, not for the emitted alpha."""
    stride: int = 1
    """Emit every ``stride``th frame. 1 for a real build; larger for a smoke run."""


@dataclass(slots=True)
class CropPlan:
    """Constant window size plus a per-frame top-left offset, both in source pixels."""
    size: int
    offsets: dict[int, tuple[int, int]]
    mode: str

    def box(self, frame: int) -> tuple[int, int, int, int]:
        x0, y0 = self.offsets[frame]
        return (x0, y0, self.size, self.size)


def frame_bboxes(doc: RotoDoc, frames: Sequence[int], cfg: CropConfig,
                 render_cfg: RenderConfig) -> dict[int, tuple[float, float, float, float]]:
    """Per-frame content bbox in source pixels, from a cheap low-res survey render."""
    out = {}
    for f in frames:
        bb = content_bbox(render_union(doc, f, render_cfg, cfg.bbox_scale))
        if bb is not None:
            out[f] = tuple(v / cfg.bbox_scale for v in bb)
    if not out:
        raise ValueError('layer renders empty on every frame')
    return out


def crop_plan(doc: RotoDoc, frames: Sequence[int], cfg: CropConfig,
              render_cfg: RenderConfig) -> CropPlan:
    """Window size from the largest single-frame bbox; offset per frame from its centre.

    Sizing on the largest *single-frame* bbox rather than the union across frames is the
    whole point: a translating layer's union spans its entire path, while any one frame
    spans only the layer.
    """
    boxes = frame_bboxes(doc, frames, cfg, render_cfg)
    if cfg.mode == 'static':
        lo_x = min(b[0] for b in boxes.values())
        lo_y = min(b[1] for b in boxes.values())
        hi_x = max(b[0] + b[2] for b in boxes.values())
        hi_y = max(b[1] + b[3] for b in boxes.values())
        side = max(hi_x - lo_x, hi_y - lo_y, float(cfg.min_crop_px)) * (1 + 2 * cfg.margin)
        size = int(round(side))
        cx, cy = (lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0
        fixed = (int(round(cx - size / 2.0)), int(round(cy - size / 2.0)))
        return CropPlan(size, {f: fixed for f in frames}, 'static')

    side = max(max(max(b[2], b[3]) for b in boxes.values()), float(cfg.min_crop_px))
    size = int(round(side * (1.0 + 2.0 * cfg.margin)))

    # Frames where the layer renders empty inherit the last known offset, so the window
    # does not jump for a shape that briefly disappears.
    offsets: dict[int, tuple[int, int]] = {}
    last = None
    for f in frames:
        bb = boxes.get(f)
        if bb is None:
            offsets[f] = last if last is not None else (0, 0)
            continue
        cx, cy = bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0
        # Deliberately not clamped to the frame: keeping the layer centred matters more
        # than staying in bounds, and outside the source simply renders empty.
        last = (int(round(cx - size / 2.0)), int(round(cy - size / 2.0)))
        offsets[f] = last
    first = next(f for f in frames if f in boxes)
    for f in frames:                                  # backfill leading empty frames
        if f == first:
            break
        offsets[f] = offsets[first]
    return CropPlan(size, offsets, 'tracking')
