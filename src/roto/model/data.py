"""Element -> tensors the network trains on.

Targets are the *dense per-frame* control points, not the artist's key values. The network
predicts geometry frame by frame; choosing which of those frames become keys is the DP's job
(``roto.keys``), and keeping the two separate is what lets each be measured on its own.

Dense tracks are built with ``roto.ir.sample`` rather than by interpolating the key arrays
directly, so the target is exactly what the renderer would draw -- per-key interp modes
included. The archive is ~75% linear and ~25% catmullrom, varying by shot, so re-deriving the
interpolation here would silently disagree with the picture on a quarter of the segments.

Targets are stored in **crop space**, in [0,1] across the alpha the network is given, not in
the IR's local normalised coordinates. See ``roto.model.geometry`` -- predicting local
coordinates directly was measured to stall at ~260 px, because they are absolute document
positions whose range is many times the crop the network can see. ``local`` is kept alongside
so the conversion can be checked, and the local points are what the IR is rebuilt from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset import load_alpha
from ..ir import opacity_at, sample
from .geometry import local_to_crop
from ..program import affine_from_matrix
from ..sfx.json_ir import from_json_ir


@dataclass
class ElementData:
    """One element, fully materialised. Dense tracks dominate memory; ~200 MB for all 13."""
    element_id: str
    directory: Path
    frames: np.ndarray          # (F,)
    points: np.ndarray          # (F, S, Pmax, Cmax, 2) float32, [0,1] across the crop
    local: np.ndarray           # (F, S, Pmax, Cmax, 2) float32, IR-native local normalised
    live: np.ndarray            # (F, S) bool
    affine: np.ndarray          # (F, G, 6) float32
    group_of: np.ndarray        # (S,) int32
    desc: np.ndarray            # (S, 3) float32 -- n_points, closed, coords_per_point
    point_mask: np.ndarray      # (S, Pmax, Cmax) bool
    px_per_norm: float
    matrices: np.ndarray        # (F, S, 4, 4) composed ancestor transform per shape per frame
    crop: dict                  # width/height/scale/out_px/offsets, for the inverse mapping
    out_px: int = 256

    @property
    def n_shapes(self) -> int:
        return self.points.shape[1]

    @property
    def n_groups(self) -> int:
        return self.affine.shape[1]

    def alpha(self, i: int) -> np.ndarray:
        return load_alpha(self.directory, int(self.frames[i]))

    _alphas: np.ndarray | None = None

    @property
    def alphas(self) -> np.ndarray:
        """(F, 256, 256) float32, materialised on first use and kept.

        All 13 elements at once is ~475 MB, which fits, but the read is slow enough that the
        training loop must not pay it per step.
        """
        if self._alphas is None:
            self._alphas = np.stack(
                [self.alpha(i) for i in range(len(self.frames))]).astype(np.float32)
        return self._alphas


def _group_index(mats: np.ndarray) -> tuple[np.ndarray, int]:
    """Collapse identical per-shape transform tracks to a group index. Exact equality: shapes
    under one tracked layer share the same matrices by construction."""
    keys: dict[bytes, int] = {}
    idx = np.empty(len(mats), np.int32)
    for i, m in enumerate(mats):
        k = np.ascontiguousarray(m, np.float64).tobytes()
        idx[i] = keys.setdefault(k, len(keys))
    return idx, len(keys)


def load_element(directory: str | Path) -> ElementData:
    d = Path(directory)
    meta = json.loads((d / 'meta.json').read_text())
    doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
    t = np.load(d / 'tensors.npz')
    frames = np.asarray(meta['frames']['index'], np.int32)
    n = int(t['n_shapes'])

    mats = [t[f'layer_matrix/{i}'].astype(np.float64) for i in range(n)]
    mframes = t['matrix_frames'].astype(np.int32)
    group_of, n_groups = _group_index(mats)

    shapes = [s for _, s in doc.shapes()]
    assert len(shapes) == n, f'{d.name}: IR has {len(shapes)} shapes, tensors say {n}'
    Pmax = max(int(t[f'points_norm/{i}'].shape[1]) for i in range(n))
    Cmax = max(int(t[f'points_norm/{i}'].shape[2]) for i in range(n))

    source = meta['source']
    crop = meta['crop']
    out_px = int(crop['out_px'][0])
    offsets = {int(k): tuple(v) for k, v in crop['offsets'].items()}
    pos = {int(f): k for k, f in enumerate(mframes)}

    local = np.zeros((len(frames), n, Pmax, Cmax, 2), np.float32)
    points = np.zeros((len(frames), n, Pmax, Cmax, 2), np.float32)
    matrices = np.zeros((len(frames), n, 4, 4), np.float64)
    live = np.zeros((len(frames), n), bool)
    pmask = np.zeros((n, Pmax, Cmax), bool)
    desc = np.zeros((n, 3), np.float32)
    for i, shape in enumerate(shapes):
        P, C = shape.n_points, int(t[f'points_norm/{i}'].shape[2])
        pmask[i, :P, :C] = True
        desc[i] = (P / Pmax, float(shape.closed), C / max(1, Cmax))
        for fi, f in enumerate(frames):
            v = np.asarray(sample(shape.path, int(f)), np.float32)
            local[fi, i, :P, :C] = v
            m = mats[i][pos[int(f)]]
            matrices[fi, i] = m
            points[fi, i, :P, :C] = local_to_crop(
                local[fi, i, :P, :C], m, width=source['width'], height=source['height'],
                offset=offsets[int(f)], scale=crop['scale'], out_px=out_px)
            live[fi, i] = opacity_at(shape, int(f)) > 0.5

    # Transform tracks are dense over `matrix_frames`; index them at the rendered frames.
    pos = {int(f): k for k, f in enumerate(mframes)}
    affine = np.zeros((len(frames), n_groups, 6), np.float32)
    for g in range(n_groups):
        src = mats[int(np.where(group_of == g)[0][0])]
        affine[:, g] = affine_from_matrix(src[[pos[int(f)] for f in frames]]).astype(np.float32)

    return ElementData(meta['element']['element_id'], d, frames, points, local, live, affine,
                       group_of, desc, pmask, float(crop['px_per_norm']), matrices,
                       {'width': source['width'], 'height': source['height'],
                        'scale': crop['scale'], 'out_px': out_px, 'offsets': offsets},
                       out_px)


def load_dataset(root: str | Path) -> list[ElementData]:
    return [load_element(p) for p in sorted(Path(root).iterdir())
            if (p / 'meta.json').exists()]
