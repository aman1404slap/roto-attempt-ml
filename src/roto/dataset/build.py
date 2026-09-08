"""Render one roto layer into a training sample: matte in, spline program out.

Each sample directory holds, for one top-level Silhouette layer:

    alpha/<frame>.png    the layer's matte, 16-bit, rendered from the artist's shapes
    target_ir.json       the shapes themselves -- native B-splines, keyframes, transforms
    tensors.npz          the same thing as arrays
    meta.json            crop transform, provenance, per-shape index
    preview.png          first / middle / last frame, for eyeballing

The matte is **our own render of the artist's shapes**, so the input and the answer agree
exactly and the model is never asked to account for anything the artist did not draw. The
frame range comes from the document.

**Control points stay in native local normalised coordinates.** It is tempting to bake the
crop transform into them so that a position loss comes out in pixels, but control points live
*under* the layer transform, and pre-baking a translation into them is meaningless and invites
the wrong composition. ``meta.json`` carries ``px_per_norm`` for a pixel-unit loss and spells
out the full composition formula.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..shots import find_shot
from .layers import layer_doc
from ..render.raster import RenderConfig, measured_conventions, render_union
from ..sfx.json_ir import to_json_ir
from ..sfx.read import read_sfx
from .arrays import derive_tensors
from .crop import CropConfig, CropPlan, crop_plan
from .manifest import RotoLayer, MANIFEST_VERSION, resolve

DATASET_VERSION = 2


@dataclass(slots=True)
class BuiltElement:
    layer_id: str
    directory: Path
    frames: list[int]
    crop: CropPlan
    n_shapes: int
    meta: dict[str, Any] = field(default_factory=dict)


def render_config_for(cfg: CropConfig) -> RenderConfig:
    """The renderer settings one ``CropConfig`` asks for. See ``CropConfig.conventions``."""
    if cfg.conventions == 'measured':
        return measured_conventions(cfg.supersample)
    if cfg.conventions == 'v1':
        return RenderConfig(supersample=cfg.supersample)
    raise ValueError(f'unknown conventions {cfg.conventions!r}, want v1 or measured')


def build(layer: RotoLayer, data_root: str | Path, out_root: str | Path,
          cfg: CropConfig | None = None) -> BuiltElement:
    cfg = cfg or CropConfig()
    render_cfg = render_config_for(cfg)
    shot = find_shot(Path(data_root) / layer.shot)
    if shot.sfx is None:
        raise FileNotFoundError(f'no .sfx under {shot.path}')
    doc = read_sfx(shot.sfx)
    ref = resolve(doc, layer)
    sub = layer_doc(doc, ref)

    frames = list(range(0, doc.duration, max(1, cfg.stride)))
    plan = crop_plan(sub, frames, cfg, RenderConfig(supersample=2))
    scale = cfg.size / plan.size

    out = Path(out_root) / layer.layer_id
    (out / 'alpha').mkdir(parents=True, exist_ok=True)

    keep, off_frame = [], 0
    for f in frames:
        box = plan.box(f)
        alpha = render_union(sub, f, render_cfg, scale, box)
        cv2.imwrite(str(out / 'alpha' / f'{f:05d}.png'),
                    np.round(np.clip(alpha, 0, 1) * 65535).astype(np.uint16))
        keep.append((f, float(alpha.mean())))
        if not (box[0] >= 0 and box[1] >= 0 and box[0] + box[2] <= doc.width
                and box[1] + box[3] <= doc.height):
            off_frame += 1

    tensors, index = derive_tensors(sub, frames)
    np.savez_compressed(out / 'tensors.npz', **tensors)
    (out / 'target_ir.json').write_text(json.dumps(to_json_ir(sub)))

    meta = {
        'dataset_version': DATASET_VERSION,
        'manifest_version': MANIFEST_VERSION,
        'layer': {**layer.as_dict(), 'uuid': ref.layer.uuid},
        'source': {
            'sfx': str(shot.sfx), 'width': doc.width, 'height': doc.height,
            'start_frame': doc.start_frame, 'duration': doc.duration,
            'frame_rate': doc.frame_rate, 'dialect': doc.dialect,
        },
        'frames': {
            'index': [f for f, _ in keep],
            'sfx': [f + doc.start_frame for f, _ in keep],
            'stride': cfg.stride,
            'coverage': [round(c, 6) for _, c in keep],
        },
        'crop': {
            'mode': plan.mode,
            'size_src_px': plan.size,
            'out_px': [cfg.size, cfg.size],
            'scale': scale,
            'px_per_norm': doc.height * scale,
            'offsets': {str(f): list(plan.offsets[f]) for f, _ in keep},
            'frames_partly_off_source': off_frame,
            'norm_to_crop_px': {
                'formula': 'crop_x = (nx * H + W/2 - x0[t]) * scale ; '
                           'crop_y = (ny * H + H/2 - y0[t]) * scale',
                'note': 'nx, ny are LOCAL normalised coords; to get on-screen position '
                        'first apply the composed layer matrix as a row vector '
                        '(p @ layer_matrix[t]), then this formula. (x0[t], y0[t]) is '
                        'offsets[str(frame)] -- the window translates, the scale does not.',
                'H': doc.height, 'W': doc.width, 'scale': scale,
            },
        },
        'render': {
            'input_alpha': 'rendered from the artist splines (clean alpha), 16-bit PNG',
            'supersample': cfg.supersample,
            'samples_per_seg': render_cfg.samples_per_seg,
            'conventions': cfg.conventions,
            'stroke_px_at_0.0309': render_cfg.stroke_px(0.030864, doc.height),
            'fill_open_zero_width': render_cfg.fill_open_zero_width,
            'open_end_rule': render_cfg.open_end_rule,
            'union_rule': 'per-pixel max over member layers',
        },
        'shapes': index,
        'stats': sub.stats(),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    _preview(out, [f for f, _ in keep])
    return BuiltElement(layer.layer_id, out, [f for f, _ in keep], plan, len(index), meta)


def _preview(out: Path, frames: Sequence[int]) -> None:
    """First / middle / last alpha side by side -- catches gross framing errors by eye."""
    if not frames:
        return
    picks = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
    tiles = [cv2.imread(str(out / 'alpha' / f'{f:05d}.png'), cv2.IMREAD_UNCHANGED)
             for f in picks]
    tiles = [t for t in tiles if t is not None]
    if tiles:
        cv2.imwrite(str(out / 'preview.png'), np.hstack(tiles))


def load_meta(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / 'meta.json').read_text())


def load_alpha(directory: str | Path, frame: int) -> np.ndarray:
    """One layer alpha frame as float32 in [0,1]."""
    path = Path(directory) / 'alpha' / f'{frame:05d}.png'
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 65535.0
