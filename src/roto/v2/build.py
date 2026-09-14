"""Render one v2 element into a training sample: matte in, spline program out.

Each sample directory holds, for one top-level Silhouette layer::

    alpha/<frame>.png    the layer's matte, 16-bit, rendered from the artist's shapes
    target_ir.json       the shapes themselves -- native B-splines, keyframes, transforms
    tensors.npz          the same thing as arrays
    meta.json            crop transform, provenance, pairing grade, per-shape index
    preview.png          first / middle / last frame, for eyeballing

**The matte is our own render of the artist's shapes**, so the input and the answer agree
exactly and the model is never asked to account for anything the artist did not draw (charter
S3; tracker D1). The delivered EXRs never enter a sample -- their verdict rides along as
``meta['pairing']`` and nothing else.

Control points stay in native local normalised coordinates. Baking the crop transform into
them would be meaningless -- control points live *under* the layer transform -- so
``meta.json`` carries ``px_per_norm`` and spells out the composition formula instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..dataset.arrays import derive_tensors
from ..dataset.crop import CropConfig, CropPlan, crop_plan
from ..dataset.layers import layer_doc
from ..render.raster import RenderConfig, conventions, render_union
from ..sfx.json_ir import to_json_ir
from ..sfx.read import read_sfx
from .ingest import Shot
from .manifest import MANIFEST_VERSION, Element, resolve
from .qc import Pairing

DATASET_VERSION = 4
"""4 is ``datasets/v003``: the 50-shot delivery, v2's element identity and split record.

3 was ``datasets/v002`` and 2 ``datasets/v001``, both built from the 6-shot archive by
``roto.dataset.build``. Bumped so that a mixed read is an error rather than a quiet
re-baseline -- the elements are differently named and differently split, and no number reads
across the boundary without being re-anchored.
"""


def v2_crop_config(size: int = 256, stride: int = 1) -> CropConfig:
    """The crop and render settings charter S3 specifies, in one place.

    ``conventions='measured'`` is the charter's "flip the two measured conventions" -- no
    zero-width fill, hairline stroke gain with a 1 px floor, duplicate open endpoints. The
    clamped Catmull-Rom interpolation law is not a flag; it lives in ``roto.ir.sample`` and
    applies unconditionally.
    """
    return CropConfig(size=size, supersample=4, conventions='measured', stride=stride)


@dataclass(slots=True)
class BuiltElement:
    element_id: str
    directory: Path
    frames: list[int]
    crop: CropPlan
    n_shapes: int
    meta: dict[str, Any] = field(default_factory=dict)


def build(element: Element, shot: Shot, out_root: str | Path,
          cfg: CropConfig | None = None, pairing: Pairing | None = None) -> BuiltElement:
    cfg = cfg or v2_crop_config()
    render_cfg = conventions(cfg.conventions, cfg.supersample)
    doc = read_sfx(shot.sfx)
    ref = resolve(doc, element)
    sub = layer_doc(doc, ref)

    frames = list(range(0, doc.duration, max(1, cfg.stride)))
    plan = crop_plan(sub, frames, cfg, RenderConfig(supersample=2))
    scale = cfg.size / plan.size

    out = Path(out_root) / element.element_id
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
        'element': element.as_dict(),
        'pairing': pairing.as_dict() if pairing else None,
        'source': {
            'sfx': str(shot.sfx), 'sfx_reason': shot.reason,
            'sfx_candidates': shot.candidates,
            'width': doc.width, 'height': doc.height,
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
            'clip_per_shape': render_cfg.clip_per_shape,
            'union_rule': 'per-pixel max over member layers',
            'interp_law': 'clamped Catmull-Rom endpoints, one law per key (roto.ir.sample)',
        },
        'shapes': index,
        'stats': sub.stats(),
        'tags': element.tags,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    _preview(out, [f for f, _ in keep])
    return BuiltElement(element.element_id, out, [f for f, _ in keep], plan, len(index), meta)


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
    """One element alpha frame as float32 in [0,1]."""
    path = Path(directory) / 'alpha' / f'{frame:05d}.png'
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 65535.0


def element_dirs(root: str | Path) -> list[Path]:
    """Every built element directory under a dataset root, in name order."""
    return sorted(d for d in Path(root).iterdir() if (d / 'meta.json').exists())
