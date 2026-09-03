"""The canonical roto intermediate representation.

This module is the project's contract: every reader, writer, renderer and model target
converts to or from these types. Design choices here are forced by measurements on the
archive. In particular:

* ``interp`` is stored **per key**, never globally. The archive is ~75% linear and ~25%
  catmullrom, and it varies by shot, so a global assumption silently corrupts every
  non-key frame.
* ``Shape.opacity`` is first class. Shape births/deaths are how artists control lifespans;
  ignoring them cost 3 points of IoU on MAT_0130 (0.97 -> 0.995).
* ``shape_type`` is preserved verbatim. 6919/6969 measured shapes are B-splines with no
  artist-authored Bezier handles. Converting to Bezier on ingest would manufacture
  parameters the artist never chose. Conversion belongs at the render boundary only.
* ``Layer.transform`` is kept separate from control points. Tracked layer transforms carry
  57-95% of on-screen shape motion, so the two must remain factorised.
* ``closed`` and ``stroke_width`` are both needed: 51% of measured shapes are open strokes
  rendered with a width, not filled regions.
* ``Layer.trs`` exists even though it is identity everywhere we have measured, because a
  layer positioned by TRS rather than a baked matrix would render in the wrong place with
  no error.

Coordinates are Silhouette-native throughout: centre origin, y-down, **both axes divided by
image height**. Conversion to pixels happens only in the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

LINEAR, HOLD, CATMULLROM = 'linear', 'hold', 'catmullrom'
INTERP_MODES = (LINEAR, HOLD, CATMULLROM)

ADD, SUBTRACT = 'Add', 'Subtract'

BSPLINE, BEZIER, XSPLINE = 'bspline', 'bezier', 'xspline'


@dataclass(slots=True)
class Key:
    """One keyframe. ``interp`` describes the segment that *starts* at this key."""
    frame: int
    interp: str
    value: Any

    def __post_init__(self) -> None:
        if self.interp not in INTERP_MODES:
            raise ValueError(f'unknown interp {self.interp!r}')


@dataclass(slots=True)
class Shape:
    """One spline. ``path`` keys hold arrays shaped (n_points, k, 2).

    ``k`` is 1 for B-splines/X-splines and 3 for Beziers, where the three pairs are
    (point, in-handle, out-handle) in Silhouette's own ordering.
    """
    name: str
    shape_type: str = BSPLINE
    closed: bool = True
    path: list[Key] = field(default_factory=list)
    opacity: list[Key] | None = None
    feather: list[Key] | None = None
    stroke_width: float = 0.0
    blend: str = ADD
    invert: bool = False
    uuid: str | None = None

    @property
    def n_points(self) -> int:
        return 0 if not self.path else int(self.path[0].value.shape[0])

    @property
    def frame_range(self) -> tuple[int, int]:
        return (self.path[0].frame, self.path[-1].frame) if self.path else (0, 0)

    def validate(self) -> None:
        if not self.path:
            raise ValueError(f'shape {self.name!r} has no path keys')
        if [k.frame for k in self.path] != sorted(k.frame for k in self.path):
            raise ValueError(f'shape {self.name!r} path keys not sorted')
        # Verified invariant across all 6 measured shots: topology is fixed per shape.
        shapes = {k.value.shape for k in self.path}
        if len(shapes) != 1:
            raise ValueError(
                f'shape {self.name!r} changes point count across keys: {shapes}. '
                'This breaks the fixed-topology assumption -- investigate before ignoring.')


@dataclass(slots=True)
class Layer:
    """A named group. ``transform`` is a track of 4x4 matrices, usually from a tracker.

    Matrices are stored exactly as Silhouette writes them: row-major with translation in
    the final row, so points are row vectors (``p @ M``). See ``render.raster.compose``.
    """
    name: str
    transform: list[Key] | None = None
    trs: dict[str, list[Key]] = field(default_factory=dict)
    """``transform.position/anchor/scale/rotate`` tracks, when present.

    Identity on all six measured shots, so it contributes nothing to current IoU -- but a
    layer that uses TRS instead of a baked matrix would render silently in the wrong place,
    which is the worst possible failure mode. Composed in ``render.raster.layer_matrix``.
    """
    children: list[Layer | Shape] = field(default_factory=list)
    opacity: list[Key] | None = None
    blend: str = ADD
    invert: bool = False
    uuid: str | None = None

    @property
    def has_transform(self) -> bool:
        return bool(self.transform) or bool(self.trs)

    @property
    def transform_is_animated(self) -> bool:
        if self.transform and len(self.transform) > 1:
            return True
        return any(len(t) > 1 for t in self.trs.values())


@dataclass(slots=True)
class RotoDoc:
    """One parsed Silhouette project."""
    width: int
    height: int
    duration: int
    roots: list[Layer] = field(default_factory=list)
    frame_rate: float = 24.0
    start_frame: int = 1001
    dialect: str | None = None
    source_path: str | None = None
    source_label: str | None = None

    # ---- traversal -------------------------------------------------------------

    def walk(self) -> Iterator[tuple[tuple[Layer, ...], Layer | Shape]]:
        """Yield ``(ancestor_layers, node)`` for every node, depth first, in file order."""
        def rec(nodes: Sequence[Layer | Shape], anc: tuple[Layer, ...]):
            for n in nodes:
                yield anc, n
                if isinstance(n, Layer):
                    yield from rec(n.children, anc + (n,))
        yield from rec(self.roots, ())

    def shapes(self) -> Iterator[tuple[tuple[Layer, ...], Shape]]:
        for anc, n in self.walk():
            if isinstance(n, Shape):
                yield anc, n

    def layers(self) -> Iterator[tuple[tuple[Layer, ...], Layer]]:
        for anc, n in self.walk():
            if isinstance(n, Layer):
                yield anc, n

    def layer(self, names: str | Sequence[str]) -> RotoDoc:
        """A shallow view containing only the named top-level layer(s).

        The reference doc's unit of inference is the *layer*, not the shot: a shot can
        hold 1600 shapes while one layer is tens. Several top-level layers can feed one
        matte, so this accepts a list.
        """
        wanted = [names] if isinstance(names, str) else list(names)
        picked = [r for r in self.roots if r.name in wanted]
        if len(picked) != len(wanted):
            missing = set(wanted) - {r.name for r in picked}
            raise KeyError(f'no top-level layer(s) {sorted(missing)}; have '
                           f'{[r.name for r in self.roots]}')
        return RotoDoc(self.width, self.height, self.duration, picked, self.frame_rate,
                       self.start_frame, self.dialect, self.source_path, self.source_label)

    def isolate(self, ancestors: Sequence[Layer], node: Layer | Shape) -> RotoDoc:
        """A doc holding only ``node``, with its ancestor chain rebuilt around it.

        Needed because a candidate matte source is often a *nested* layer, and rendering it
        detached would drop every tracked transform above it -- which looks plausible and is
        wrong. Ancestors are rebuilt as childless stubs so their transforms still compose.
        """
        cur: Layer | Shape = node
        for anc in reversed(list(ancestors)):
            cur = Layer(name=anc.name, transform=anc.transform, trs=anc.trs,
                        opacity=anc.opacity, blend=anc.blend, invert=anc.invert,
                        uuid=anc.uuid, children=[cur])
        root = cur if isinstance(cur, Layer) else Layer(name='<bare>', children=[cur])
        return RotoDoc(self.width, self.height, self.duration, [root], self.frame_rate,
                       self.start_frame, self.dialect, self.source_path, self.source_label)

    def validate(self) -> None:
        for _, s in self.shapes():
            s.validate()

    def stats(self) -> dict[str, Any]:
        sh = [s for _, s in self.shapes()]
        ly = [l for _, l in self.layers()]
        return {
            'layers': len(ly),
            'shapes': len(sh),
            'points': sum(s.n_points for s in sh),
            'keys': sum(len(s.path) for s in sh),
            'open_strokes': sum(1 for s in sh if not s.closed),
            'with_lifespan': sum(1 for s in sh if s.opacity and len(s.opacity) > 1),
            'ephemeral': sum(1 for s in sh if is_ephemeral(s)),
            'tracked_layers': sum(1 for l in ly if l.has_transform),
            'animated_transforms': sum(1 for l in ly if l.transform_is_animated),
            'shape_types': {t: sum(1 for s in sh if s.shape_type == t)
                            for t in {s.shape_type for s in sh}},
            'path_interp': {m: sum(1 for s in sh for k in s.path if k.interp == m)
                            for m in INTERP_MODES},
        }


# ---- track sampling ------------------------------------------------------------

def sample(keys: Sequence[Key], frame: float) -> Any:
    """Value of a track at ``frame``. Holds flat outside the keyed range."""
    if not keys:
        raise ValueError('empty track')
    if len(keys) == 1 or frame <= keys[0].frame:
        return keys[0].value
    if frame >= keys[-1].frame:
        return keys[-1].value

    i = 0
    for j, k in enumerate(keys):
        if k.frame <= frame:
            i = j
        else:
            break
    k0, k1 = keys[i], keys[i + 1]
    if k0.interp == HOLD or k1.frame == k0.frame:
        return k0.value
    u = (frame - k0.frame) / (k1.frame - k0.frame)

    if k0.interp == CATMULLROM and 0 < i < len(keys) - 2:
        a, b = keys[i - 1].value, keys[i + 2].value
        if np.shape(a) == np.shape(k0.value) == np.shape(k1.value) == np.shape(b):
            p0, p1, p2, p3 = a, k0.value, k1.value, b
            return 0.5 * (2 * p1 + (-p0 + p2) * u
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return k0.value * (1 - u) + k1.value * u


# ---- lifespans ------------------------------------------------------------------
#
# Artists gate a shape's existence with hold keys on opacity (0 -> 100 -> 0), not by
# adding or removing the shape. 51% of measured shapes are gated to <=2 frames: these are
# single-frame paint-stroke hair, and they have exactly one path key by construction.
#
# They must be separable from persistent shapes as a *field*, not a training-time filter:
# mixed into a keyframe-timing loss they teach "key every frame", which is the exact
# failure this project exists to prevent.

EPHEMERAL_MAX_FRAMES = 2


def live_frames(shape: Shape) -> list[int]:
    """Integer frames on which ``shape`` has non-zero opacity, within its keyed range.

    A shape with no opacity track is live wherever it exists, so this returns its path
    key range -- callers wanting whole-shot liveness should use the doc duration instead.
    """
    op = shape.opacity
    if not op or len(op) == 1:
        lo, hi = shape.frame_range
        return list(range(lo, hi + 1))
    lo = min(op[0].frame, shape.frame_range[0])
    hi = max(op[-1].frame, shape.frame_range[1])
    return [f for f in range(lo, hi + 1) if opacity_at(shape, f) > 0.0]


def is_ephemeral(shape: Shape, max_frames: int = EPHEMERAL_MAX_FRAMES) -> bool:
    """True for opacity-gated single-frame strokes; False for persistent animated shapes."""
    if not shape.opacity or len(shape.opacity) <= 1:
        return False
    return len(live_frames(shape)) <= max_frames


def shape_class(shape: Shape) -> str:
    """``'{persistent|ephemeral}_{fill|stroke}'`` -- the four training populations."""
    life = 'ephemeral' if is_ephemeral(shape) else 'persistent'
    kind = 'fill' if shape.closed else 'stroke'
    return f'{life}_{kind}'


def opacity_at(shape: Shape, frame: float) -> float:
    """Shape opacity in [0,1]. Silhouette stores 0-100."""
    if not shape.opacity:
        return 1.0
    return float(np.clip(np.ravel(sample(shape.opacity, frame))[0] / 100.0, 0.0, 1.0))
