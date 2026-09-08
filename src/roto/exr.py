"""Silhouette's own delivered mattes, for refereeing our render conventions against theirs.

Several rendering rules in this codebase are flagged "unverified" because they were
calibrated by eye rather than derived: the endpoint rule for open B-splines, whether a
zero-width open shape fills, the stroke-width unit, and whether a Subtract clips per shape
or at layer output. Together they decide every open-stroke shape, which is 51% of the
archive.

They do not have to stay assumptions. Each shot ships Silhouette's *own* render of its
mattes as EXR, so a convention can be settled by rendering both ways and asking which one
agrees with the reference. That is what this module exists for: it is validation material,
never training input. The training alpha stays our own render of the artist's shapes, so the
model is never asked to explain a vendor's compositing quirk.

Two practical notes. Each delivered EXR packs three independent mattes into its B, G and R
channels, so a "channel" is the unit that gets compared, not a file. And the comparison is
done at **full resolution**: a downscaled compare understates the disagreement by up to 0.02
soft IoU, and it does so precisely at the edge, which is the only place these conventions
differ.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

# Must precede the first EXR read: OpenCV gates its OpenEXR codec behind this and raises
# rather than returning None when it is unset. Set here so importing this module is enough.
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2                                                            # noqa: E402

from .ir import RotoDoc
from .metrics import soft_iou
from .render.raster import RenderConfig, render_union

CHANNELS = ('B', 'G', 'R')
"""OpenCV's channel order for a 3-channel EXR. Each carries a different matte."""

_FRAME_RE = re.compile(r'\.(\d+)\.exr$')


def matte_sequences(shot_dir: str | Path) -> dict[str, dict[int, Path]]:
    """``{matte_directory_name: {frame_in_file_name: path}}`` for one shot.

    Frame numbers are as written in the file names, which are Silhouette's own (1001-based),
    not the document's 0-based internal index. ``sfx_frame = internal + doc.start_frame``.
    """
    out: dict[str, dict[int, Path]] = {}
    for d in sorted(Path(shot_dir).iterdir()):
        if not d.is_dir():
            continue
        frames = {}
        for f in sorted(d.glob('*.exr')):
            m = _FRAME_RE.search(f.name)
            if m:
                frames[int(m.group(1))] = f
        if frames:
            out[d.name] = frames
    return out


def load_channels(path: str | Path, scale: float = 1.0) -> dict[str, np.ndarray]:
    """One delivered EXR as ``{channel: float32 alpha}``, clipped to [0,1].

    Silhouette writes linear float; values sit in [0,1] for a matte but are clipped rather
    than trusted, because a stray value above 1 would silently distort every soft IoU it
    takes part in.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'could not read {path}')
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        return {'Y': np.clip(img.astype(np.float32), 0.0, 1.0)}
    return {CHANNELS[i]: np.clip(img[..., i].astype(np.float32), 0.0, 1.0)
            for i in range(min(img.shape[2], len(CHANNELS)))}


def score_against_channel(doc: RotoDoc, sequences: dict[str, dict[int, Path]],
                          matte: str, channel: str, frames: Sequence[int],
                          cfg: RenderConfig | None = None, scale: float = 1.0
                          ) -> list[float]:
    """Soft IoU of ``doc`` rendered on each of ``frames`` against one delivered channel.

    ``frames`` are the document's own internal frame numbers.
    """
    cfg = cfg or RenderConfig(supersample=2)
    out = []
    for f in frames:
        path = sequences[matte].get(int(f) + doc.start_frame)
        if path is None:
            continue
        ref = load_channels(path, scale)[channel]
        pred = render_union(doc, int(f), cfg, scale)
        if pred.shape != ref.shape:                 # rounding on odd sizes
            h, w = min(pred.shape[0], ref.shape[0]), min(pred.shape[1], ref.shape[1])
            pred, ref = pred[:h, :w], ref[:h, :w]
        out.append(soft_iou(pred, ref))
    return out


def best_channel(doc: RotoDoc, sequences: dict[str, dict[int, Path]],
                 frames: Sequence[int], cfg: RenderConfig | None = None,
                 scale: float = 0.5) -> tuple[str, str, float]:
    """``(matte, channel, mean soft IoU)`` for the delivered channel this doc best explains.

    Matching runs at reduced scale on purpose -- it only has to identify *which* channel a
    layer corresponds to, and that is a coarse question. Whatever is being refereed is then
    scored at full resolution, where the conventions actually differ.
    """
    best = ('', '', -1.0)
    for matte in sequences:
        first = sequences[matte][sorted(sequences[matte])[0]]
        for channel in load_channels(first, 1.0):
            s = score_against_channel(doc, sequences, matte, channel, frames, cfg, scale)
            if s and float(np.mean(s)) > best[2]:
                best = (matte, channel, float(np.mean(s)))
    return best
