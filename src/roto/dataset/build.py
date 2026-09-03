"""Build one element into a training example: clean alpha in, spline program out.

**The input alpha is our own render of the artist's splines.** Not the delivered EXR. The
answer key generates its own exam question, so input and target agree exactly and the model
is never asked to explain a vendor's compositing quirk alongside the artist's craft. The
EXRs stay what they always were: the evidence that the renderer is right (``roto verify``).

One consequence is easy to miss and is the reason this file no longer imports anything
EXR-shaped. The old builder took its frame range from the delivered EXR directory listing --
``sorted(shot.mattes[element.matte])`` -- so an element with no certified matte channel could
not be built at all. The frame range now comes from the document, which is where it always
belonged, and every layer in the archive is buildable whether or not a matte was delivered
for it.

**Points stay in native local normalised coordinates.** It is tempting to bake the crop
transform into the point arrays so a position loss comes out in pixels, but local control
points live *under* the layer transform, and pre-baking a translation into them is
meaningless and invites the wrong composition. ``meta.json`` carries ``px_per_norm``, so a
pixel-unit loss is ``|dp| * px_per_norm``, and the full on-screen position is
``crop_affine(local @ layer_matrix)`` -- spelled out in the meta itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..data.shots import find_shot
from ..eval.match import element_doc
from ..render.raster import RenderConfig, render_union
from ..sfx.json_ir import to_json_ir
from ..sfx.read import read_sfx
from .arrays import derive_tensors
from .crop import CropConfig, CropPlan, crop_plan
from .manifest import Element, MANIFEST_VERSION, resolve

DATASET_VERSION = 2


@dataclass(slots=True)
class BuiltElement:
    element_id: str
    directory: Path
    frames: list[int]
    crop: CropPlan
    n_shapes: int
    meta: dict[str, Any] = field(default_factory=dict)


def build(element: Element, data_root: str | Path, out_root: str | Path,
          cfg: CropConfig | None = None) -> BuiltElement:
    cfg = cfg or CropConfig()
    render_cfg = RenderConfig(supersample=cfg.supersample)
    shot = find_shot(Path(data_root) / element.shot)
    if shot.sfx is None:
        raise FileNotFoundError(f'no .sfx under {shot.path}')
    doc = read_sfx(shot.sfx)
    members = resolve(doc, element)
    sub = element_doc(doc, members)

    # Frame range from the document, not from a delivered matte directory.
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
        'members': [{'label': c.label, 'uuid': c.layer.uuid} for c in members],
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
            'norm_to_packet_px': {
                'formula': 'packet_x = (nx * H + W/2 - x0[t]) * scale ; '
                           'packet_y = (ny * H + H/2 - y0[t]) * scale',
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
            'stroke_width_gain': 0.0625,
            'union_rule': 'per-pixel max over member layers',
            'exr_used': False,
        },
        'shapes': index,
        'stats': sub.stats(),
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
