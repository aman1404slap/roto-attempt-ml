"""Deterministic IR -> alpha rasteriser.

Coordinate convention (Silhouette native): centre origin, y-down, **both axes divided by
image height**. So for an image W x H:

    px = x * H + W / 2
    py = y * H + H / 2

Layer transforms are row-major 4x4 with translation in the final row, so points are row
vectors and compose leaf-first:

    p' = p @ M_leaf @ M_parent @ ... @ M_root

Getting that order or transposition wrong still produces plausible-looking output, which
is why it is spelled out here and covered by golden tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from ..ir import (ADD, BEZIER, BSPLINE, SUBTRACT, Key, Layer, RotoDoc, Shape,
                  opacity_at, sample)
from .curves import DUPLICATE, OPEN_END_RULE, eval_bspline, eval_silhouette_bezier


STROKE_WIDTH_GAIN = 0.0625
"""``px_width = strokeWidth * height * GAIN``.

Empirically calibrated, not derived: ``width * height`` overestimates by ~16x on
TVC_sh0230 (0.039 -> 50px, where ~2-3px matches the hair the artist drew). A gain of 1.0
would put strokes at roughly 16x their real width, which is why an unscaled stroke
renders came out hairline.

**Now diffed against Silhouette's own renderer, and 1/16 is neither a unit nor an optimum.**
The suspicion that 0.0625 = 1/16 exactly meant Silhouette defines strokeWidth in a 16-per-unit
system is refuted: agreement with the delivered EXRs improves monotonically *past* 1/16 on
every layer tested and flattens only where ``default_stroke_px``'s 1 px floor takes over, so
these strokes are at or below one pixel in the reference render. A unit conversion would have
won on both shots; this loses to 1/64 on both. Left unchanged because the gain re-renders the
training alphas -- see ``v1.1/results/render_conventions.json``.
"""


def default_stroke_px(width: float, height: int) -> float:
    """Convert a shape's ``strokeWidth`` to pixels. v1's convention -- see the gain above."""
    return max(1.0, width * height * STROKE_WIDTH_GAIN)


def measured_stroke_px(width: float, height: int) -> float:
    """The stroke width the delivered EXRs actually show: a hairline, floored at 1 px.

    The gain sweep against Silhouette's own render improves monotonically as the gain falls
    and then stops changing entirely -- 1/64, 1/128 and 1/256 all score 0.49559 on FAM green 2
    against 1/16's 0.49313 -- because below 1/64 the 1 px floor is what every stroke gets. So
    the measured answer is not a smaller gain, it is *the floor*: these strokes are at or under
    one pixel in the reference, and the gain is doing nothing but adding width Silhouette does
    not draw."""
    return 1.0


@dataclass(slots=True)
class RenderConfig:
    samples_per_seg: int = 12
    supersample: int = 2
    """Render at ``supersample`` x then box-filter down, producing fractional coverage.

    Silhouette writes a grey anti-aliased edge; a hard 0/1 render disagrees with it on
    every boundary pixel, which is the entire residual in the round-trip numbers. Keeping
    the soft values is what makes ``soft_iou`` meaningful, so this defaults to 2 (the
    measured setting) rather than 1. Memory cost is ss^2 x the frame.
    """
    stroke_px: Callable[[float, int], float] = default_stroke_px
    fill_open_zero_width: bool = True
    """Open shapes with strokeWidth == 0: fill as if closed.

    516 of nfl_0080's shapes are open with zero width, and filling them is what reaches
    0.85 IoU there. **Refereed against the delivered EXRs and measured wrong**: on nfl_0080
    MB 2, where 508 of 510 shapes are open at zero width, ``False`` scores 0.9442 against
    ``True``'s 0.8992. Left at ``True`` because changing it re-renders the training alphas
    and so needs a dataset rebuild -- see ``v1.1/results/render_conventions.json``.
    """
    open_end_rule: str = OPEN_END_RULE
    """How an open B-spline is extended past its end points. See ``render.curves``."""
    clip_per_shape: bool = True
    """Clip the accumulator to [0,1] after *every* shape, rather than once at layer output.

    The two differ only where a Subtract follows overlapping Adds: clipping first discards
    the overshoot, so the Subtract removes less than it would have. Only two layers in the
    archive mix Subtracts with Adds at all (7 shapes in FAM blue 1, 2 in sh0230 L100), which
    is few enough to referee against the delivered EXR rather than assume. Measured on FAM
    blue 1 at full resolution: identical to four decimals either way, so the assumption is
    safe here -- see ``v1.1/results/render_conventions.json``.
    """


def measured_conventions(supersample: int = 4) -> RenderConfig:
    """The render conventions **refereed against Silhouette's own EXRs**, as one object.

    Three of the four conventions v1.1 measured came out against what the code shipped, and
    all three were left in place there deliberately: changing any of them re-renders the
    training alphas, and v1.1's whole value is being comparable to v1 row for row. That
    argument expires at the v2 dataset rebuild. Carrying them forward instead would train the
    model to draw strokes Silhouette does not draw -- invisible in a self-scored number, and
    visible the first day a real ``.sfx`` ships.

    ==========================  ================  ==========================================
    convention                  v1 / v1.1         measured, and by how much
    ==========================  ================  ==========================================
    ``fill_open_zero_width``    ``True``          ``False``: 0.9442 vs 0.8992 on nfl_0080
                                                  MB 2, where 508 of 510 shapes are open at
                                                  zero width
    ``stroke_px``               gain 1/16         the 1 px floor: 0.49559 vs 0.49313 on FAM
                                                  green 2, and flat below 1/64
    ``open_end_rule``           ``triplicate``    ``duplicate``: +0.0006 and +0.0078 on the
                                                  two layers that can referee it
    closed-shape rendering      --                **verified**, 0.9865-0.9949 on three layers
                                                  with no open shapes -- the control that
                                                  makes the other three readable
    ==========================  ================  ==========================================

    Exposed as a function rather than as three edits so the rebuild is one flag
    (``CropConfig.conventions='measured'``) and cannot land two of the three by accident.
    """
    return RenderConfig(supersample=supersample, stroke_px=measured_stroke_px,
                        fill_open_zero_width=False, open_end_rule=DUPLICATE)


V1_CONVENTIONS, MEASURED_CONVENTIONS = 'v1', 'measured'
CONVENTION_SETS = (V1_CONVENTIONS, MEASURED_CONVENTIONS)


def conventions(name: str, supersample: int) -> RenderConfig:
    """The named convention set at a given supersample. One place both callers agree on."""
    if name == MEASURED_CONVENTIONS:
        return measured_conventions(supersample)
    if name == V1_CONVENTIONS:
        return RenderConfig(supersample=supersample)
    raise ValueError(f'unknown conventions {name!r}, want one of {CONVENTION_SETS}')


def config_from_meta(render_meta: dict, supersample: int | None = None) -> RenderConfig:
    """The ``RenderConfig`` a dataset's alphas were **actually drawn with**, from its own
    ``meta.json``, with the recorded flags *verified* rather than trusted.

    This function exists because of a bug the v002 rebuild exposed, and the shape of that bug
    is worth keeping written down. Every scoring path in v1, v1.1 and v1.2 rendered with
    ``RenderConfig(supersample=meta['render']['supersample'])`` -- carrying one convention
    across from the dataset and silently reasserting the *class defaults* for the other three.
    That was invisible for two rounds because ``datasets/v001`` was built with v1's
    conventions, and v1's conventions **are** the defaults, so the omission could not produce
    a wrong number. Flip the dataset to the measured set and it can: the artist's own program,
    scored against its own alphas, reads 0.9925 on ``FAM blue_1`` and 0.9887 on ``green_2``
    -- the two layers with hundreds of open strokes -- purely because the scorer filled open
    zero-width shapes that the dataset had stroked, and stroked them at a different width.
    With the conventions read back, both are 1.000000.

    So the lesson generalises past this fix: a convention that lives in a *default* is a
    convention two independent code paths can disagree about while both look right. The
    conventions now travel with the data, and the requirement is checked here so that a
    dataset whose record disagrees with any convention set is an error rather than a
    re-baseline nobody notices.

    ``supersample`` overrides the recorded one, which is the one axis a caller legitimately
    varies (``RebuildConfig.supersample`` exists so anti-aliasing can be measured on purpose).
    Everything else comes from the record.
    """
    name = render_meta.get('conventions', V1_CONVENTIONS)
    ss = int(supersample if supersample else render_meta.get('supersample', 2))
    cfg = conventions(name, ss)
    for field, recorded in (('fill_open_zero_width', render_meta.get('fill_open_zero_width')),
                            ('open_end_rule', render_meta.get('open_end_rule')),
                            ('samples_per_seg', render_meta.get('samples_per_seg')),
                            ('clip_per_shape', render_meta.get('clip_per_shape'))):
        if recorded is None:                       # older meta.json: the field predates it
            continue
        got = getattr(cfg, field)
        if got != recorded:
            raise ValueError(
                f"dataset records conventions={name!r} with {field}={recorded!r}, but that "
                f'set has {field}={got!r}. The record and the code disagree about how these '
                'alphas were drawn; do not score against them until it is resolved.')
    return cfg


def trs_matrix(trs: dict[str, list[Key]], frame: float) -> np.ndarray | None:
    """Anchor/scale/rotate/position tracks -> one 4x4, or ``None`` when identity.

    Row-vector convention, so the order applied to a point is
    ``T(-anchor) . S . R . T(anchor + position)``.
    """
    if not trs:
        return None

    def vec(name: str, default: Sequence[float]) -> list[float]:
        track = trs.get(name)
        if not track:
            return list(default)
        v = np.ravel(np.asarray(sample(track, frame), dtype=np.float64))
        out = list(default)
        out[:min(len(v), len(out))] = list(v[:len(out)])
        return out

    ax, ay = vec('anchor', [0.0, 0.0])
    px, py = vec('position', [0.0, 0.0])
    sx, sy = vec('scale', [1.0, 1.0])
    rot = math.radians(vec('rotate', [0.0])[0])
    if (ax, ay, px, py, sx, sy, rot) == (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0):
        return None

    c, s = math.cos(rot), math.sin(rot)
    t1 = np.eye(4); t1[3, 0], t1[3, 1] = -ax, -ay
    sc = np.diag([sx, sy, 1.0, 1.0])
    r = np.eye(4); r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, s, -s, c
    t2 = np.eye(4); t2[3, 0], t2[3, 1] = ax + px, ay + py
    return t1 @ sc @ r @ t2


def layer_matrix(layer: Layer, frame: float) -> np.ndarray | None:
    """A layer's own 4x4 at ``frame``: TRS first, then the baked tracker matrix."""
    mat = sample(layer.transform, frame) if layer.transform else None
    trs = trs_matrix(layer.trs, frame)
    if mat is None:
        return trs
    return mat if trs is None else trs @ mat


def compose(mats: Sequence[np.ndarray]) -> np.ndarray:
    """Compose ancestor matrices given root-first, for row-vector points."""
    out = np.eye(4)
    for m in reversed(mats):
        out = out @ m
    return out


def apply_transform(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a composed 4x4 to (n, 2) normalised points."""
    h = np.concatenate([pts, np.zeros((len(pts), 1)), np.ones((len(pts), 1))], axis=1)
    r = h @ mat
    w = r[:, 3:4]
    return r[:, :2] / np.where(np.abs(w) < 1e-12, 1.0, w)


def shape_polyline(shape: Shape, frame: float, ancestors: Sequence[Layer],
                   cfg: RenderConfig) -> np.ndarray | None:
    """Normalised-space polyline for one shape at one frame, transforms applied."""
    if not shape.path:
        return None
    pts = sample(shape.path, frame)                      # (n_points, k, 2)

    mats = [m for m in (layer_matrix(l, frame) for l in ancestors) if m is not None]
    mat = compose(mats) if mats else None

    if shape.shape_type == BEZIER and pts.shape[1] == 3:
        p = pts.copy()
        if mat is not None:
            p = apply_transform(mat, p.reshape(-1, 2)).reshape(-1, 3, 2)
        return eval_silhouette_bezier(p, shape.closed, cfg.samples_per_seg)

    # B-spline and X-spline (X-spline tension is ignored -- 5 shapes archive-wide)
    p = pts[:, 0, :]
    if mat is not None:
        p = apply_transform(mat, p)
    if len(p) < 3:
        return p
    return eval_bspline(p, shape.closed, cfg.samples_per_seg, cfg.open_end_rule)


def render(doc: RotoDoc, frame: float, cfg: RenderConfig | None = None,
           scale: float = 1.0, crop: Sequence[float] | None = None) -> np.ndarray:
    """Rasterise every shape in ``doc`` at ``frame``. Returns float32 alpha in [0,1].

    ``crop`` is ``(x0, y0, w, h)`` in *source* pixels; the output is ``(h*scale, w*scale)``
    covering just that window. Rendering a 512px crop of a 2880x1978 frame directly, rather
    than rendering the full frame and slicing, is what makes high supersampling affordable
    where it matters. The crop box may extend outside the source frame -- the layer stays
    centred and the region outside simply renders empty.
    """
    cfg = cfg or RenderConfig()
    ss = max(1, int(cfg.supersample))
    x0, y0, cw, ch = (0.0, 0.0, float(doc.width), float(doc.height)) if crop is None \
        else (float(v) for v in crop)
    W = max(1, int(round(cw * scale))) * ss
    H = max(1, int(round(ch * scale))) * ss
    # The frame centre is (width/2, height/2) in *source* pixels, so it must be scaled --
    # not taken as W/2 of the rounded canvas, which drifts by up to half a pixel whenever
    # width*scale or height*scale is not an integer. Same formula for cropped and full
    # frames, so a crop is exactly the corresponding window of the full render.
    ox = (doc.width / 2.0 - x0) * scale * ss
    oy = (doc.height / 2.0 - y0) * scale * ss
    Hn = doc.height * scale * ss                         # normalisation basis, in px
    acc = np.zeros((H, W), np.float32)
    scratch = np.zeros((H, W), np.float32)

    for ancestors, shape in doc.shapes():
        alpha = opacity_at(shape, frame)
        for layer in ancestors:
            if layer.opacity:
                alpha *= float(np.clip(np.ravel(sample(layer.opacity, frame))[0] / 100.0,
                                       0.0, 1.0))
        if alpha <= 0.0:
            continue

        poly = shape_polyline(shape, frame, ancestors, cfg)
        if poly is None or len(poly) < 2:
            continue
        xy = np.empty_like(poly)
        xy[:, 0] = poly[:, 0] * Hn + ox
        xy[:, 1] = poly[:, 1] * Hn + oy
        ipts = np.round(xy).astype(np.int32)

        stroke = (not shape.closed) and (
            shape.stroke_width > 0 or not cfg.fill_open_zero_width)
        pad = 1
        if stroke:
            w = int(max(1, round(cfg.stroke_px(shape.stroke_width, doc.height) * scale * ss)))
            pad += w

        # Work inside the shape's own bounding box. A full-canvas scratch buffer costs a
        # 16MB memset per shape at a 512 crop with supersample 4, which dominated render
        # time for shape-dense layers. An inverted shape covers everything outside itself,
        # so it is the one case that still needs the whole canvas.
        if shape.invert:
            x0i, y0i, x1i, y1i = 0, 0, W, H
        else:
            x0i = max(0, int(ipts[:, 0].min()) - pad)
            y0i = max(0, int(ipts[:, 1].min()) - pad)
            x1i = min(W, int(ipts[:, 0].max()) + pad + 1)
            y1i = min(H, int(ipts[:, 1].max()) + pad + 1)
            if x1i <= x0i or y1i <= y0i:
                continue                              # entirely outside the frame or crop
        view = scratch[y0i:y1i, x0i:x1i]
        view[:] = 0.0
        ipoly = [ipts - np.array([[x0i, y0i]], np.int32)]

        if stroke:
            cv2.polylines(view, ipoly, False, 1.0, w, cv2.LINE_8)
        else:
            cv2.fillPoly(view, ipoly, 1.0, cv2.LINE_8)

        if shape.invert:
            np.subtract(1.0, view, out=view)
        if alpha < 1.0:
            view *= alpha

        target = acc[y0i:y1i, x0i:x1i]
        if shape.blend == SUBTRACT:
            np.subtract(target, view, out=target)
        else:
            np.add(target, view, out=target)
        if cfg.clip_per_shape:
            np.clip(target, 0.0, 1.0, out=target)

    if not cfg.clip_per_shape:
        np.clip(acc, 0.0, 1.0, out=acc)
    if ss > 1:
        acc = cv2.resize(acc, (W // ss, H // ss), interpolation=cv2.INTER_AREA)
    return acc


def render_union(doc: RotoDoc, frame: float, cfg: RenderConfig | None = None,
                 scale: float = 1.0, crop: Sequence[float] | None = None) -> np.ndarray:
    """Render each root separately and take the per-pixel maximum.

    A layer's matte is the *union* of its shapes, and union is not
    the same as this renderer's add-and-clip: adding two overlapping soft edges overshoots,
    and a Subtract in one root must not punch a hole in another. Every layer->channel
    comparison in the codebase uses this rule, so it lives here rather than being restated.
    """
    acc = None
    for root in doc.roots:
        one = render(RotoDoc(doc.width, doc.height, doc.duration, [root], doc.frame_rate,
                             doc.start_frame, doc.dialect, doc.source_path,
                             doc.source_label), frame, cfg, scale, crop)
        acc = one if acc is None else np.maximum(acc, one)
    if acc is None:
        raise ValueError('document has no roots to render')
    return acc


def content_bbox(alpha: np.ndarray, threshold: float = 0.0) -> tuple[int, int, int, int] | None:
    """``(x0, y0, w, h)`` of the non-empty region, or ``None`` if the alpha is blank."""
    cols = np.nonzero((alpha > threshold).any(axis=0))[0]
    rows = np.nonzero((alpha > threshold).any(axis=1))[0]
    if not len(cols) or not len(rows):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def live_shape_count(doc: RotoDoc, frame: float) -> int:
    """Shapes whose opacity is non-zero at ``frame`` -- useful for sanity checks."""
    return sum(1 for _, s in doc.shapes() if opacity_at(s, frame) > 0.0)
