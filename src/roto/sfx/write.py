"""IR -> Silhouette ``.sfx``. Serialisation, and nothing else.

**There is no machine learning in this file, and there never will be.** The model's job ends
when it produces a ``RotoDoc`` -- ``roto.program.decode`` already returns one -- and everything
from there to a file an artist can open is a deterministic format problem. Keeping that
boundary explicit is the point of the module.

Through v1.2 this was a documented seam that raised ``NotImplementedError``, on the grounds
that a writer nobody has opened in Silhouette is unverifiable. That was the right call while
the *representation* was still lossy: the program's transform track carried six numbers where
the archive needs eight, so a file written from it would have put one layer's shapes 350 crop
px from where the artist left them (``v1.2/tech.md`` S3.1). With that fixed there is nothing
left to wait for except the seat, and a file cannot be checked in a seat until it exists.

Three entry points, because "write a .sfx" is really three different jobs:

* :func:`serialise` -- ``RotoDoc`` -> XML bytes. The whole of the format work.
* :func:`write_sfx` -- a **standalone** project, built from nothing but the IR. This is the
  strong claim, and it is what the ledger's ``read(write(IR))`` row measures: everything the
  IR holds survives a trip through the file and comes back bit-identical.
* :func:`rewrite_sfx` -- the artist's **own project** with named layers substituted and every
  other byte of structure preserved. This is the one to open in a seat: the plate, the node
  graph, the pipes and the session settings are the ones Silhouette already renders, so a
  difference in the picture is a difference in our shapes rather than in our project skeleton.

What the format needs, from what the reader measured:

* **Two containers.** Dialects ``2020``/``2022.5``/``2025.5`` write a big-endian uint32
  uncompressed length followed by a zlib stream; dialect ``5`` writes plain XML.
  ``RotoDoc.dialect`` records ``'<version>/<container>'``, so a round trip writes back the
  container it read.
* **B-splines stay B-splines.** 6919 of 6969 measured shapes are B-splines with no
  artist-authored Bezier handles. Converting on write would manufacture parameters the artist
  never chose, and the deliverable is a file the artist *edits*.
* **Per-key interpolation.** The archive is ~75% linear and ~25% catmullrom, varying by shot,
  and Silhouette's own names for those modes are exactly the IR's. A global mode would
  silently corrupt every non-key frame.
* **Opacity hold keys are how shapes are born and die**, not shape insertion and removal.
  Getting this wrong on read cost three points of IoU and looked like a geometry bug.

**Numbers are written at shortest-round-trip precision** (``repr``), not at Silhouette's own
fixed 9 decimals. That is what makes the round-trip row *bit*-exact rather than
exact-to-a-tolerance, and it costs nothing: shortest-round-trip decimal is still ordinary
decimal, and Silhouette's parser reads it the same way ours does.

One asymmetry is worth stating rather than discovering later. The reader **normalises** as it
reads: a constant-identity ``transform.*`` track is dropped, a blank ``transform.matrix`` is
represented as ``None``, and a bare shape at node level is wrapped in a synthetic
``<bare:name>`` layer. So ``read(write(x)) == x`` holds for every ``x`` that came out of
:func:`~roto.sfx.read.read_sfx` -- which is every document in this project -- and does not
hold for a hand-built IR carrying redundant identity tracks. The writer restores the bare-shape
case (a root named ``<bare:...>`` is written back as a bare shape) because that one is load
bearing: MAT_0130 and sh0230 both have node-level shapes.
"""
from __future__ import annotations

import struct
import uuid as _uuid
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from ..ir import ADD, BEZIER, BSPLINE, XSPLINE, Key, Layer, RotoDoc, Shape

HEADER = b'<!-- Silhouette Project File -->\n'
"""The first bytes of a plain-container project, and how ``read_xml_bytes`` recognises one."""

PLAIN, ZLIB = 'plain', 'zlib'

DIALECT_CONTAINER = {'2020': ZLIB, '2022.5': ZLIB, '2025.5': ZLIB, '5': PLAIN}
"""Container per project version, as measured across the six archive shots. A version we have
never seen defaults to ``plain``, which is readable by every dialect."""

DEFAULT_DIALECT = '2022.5/zlib'
"""What a document with no recorded dialect is written as. 2022.5 is the newest dialect in the
archive that carries ``catmullrom`` keys, so it is the safest default for our own output."""

_SHAPE_TYPE_NAMES = {BSPLINE: 'Bspline', BEZIER: 'Bezier', XSPLINE: 'Xspline'}

BARE_PREFIX = '<bare:'
"""Root layers with this name are the reader's wrapper around a node-level shape."""


# ---- numbers -------------------------------------------------------------------

def num(x: Any) -> str:
    """One number, at shortest-round-trip precision.

    ``repr`` on a Python float is the shortest decimal that reads back to the same bits, which
    is what makes the round trip bit-exact. numpy scalars are cast first: ``repr`` of a numpy
    float in numpy >= 2 is ``np.float64(0.1)``, which no parser wants.
    """
    v = float(x)
    if v.is_integer() and abs(v) < 1e16:
        return str(int(v))
    return repr(v)


def _pairs(arr: np.ndarray) -> str:
    """``(n, 2)`` -> ``'(x,y)(x,y)...'``, the form ``read._pairs`` scans for."""
    a = np.asarray(arr, np.float64).reshape(-1, 2)
    return ''.join(f'({num(x)},{num(y)})' for x, y in a)


def _vector(arr: Any) -> str:
    """A numeric vector -> ``'(a,b,c)'``. Single values are written bare, as Silhouette does."""
    v = np.ravel(np.asarray(arr, np.float64))
    if len(v) == 1:
        return num(v[0])
    return '(' + ','.join(num(x) for x in v) + ')'


# ---- property helpers ----------------------------------------------------------

def _prop(parent: ET.Element, pid: str, *, constant: bool = False,
          expanded: bool = False) -> ET.Element:
    el = ET.SubElement(parent, 'Property', {'id': pid})
    if expanded:
        el.set('expanded', 'True')
    if constant:
        el.set('constant', 'True')
    return el


def _value_prop(parent: ET.Element, pid: str, text: str, *, constant: bool = True) -> None:
    ET.SubElement(_prop(parent, pid, constant=constant), 'Value').text = text or None


def _key_prop(parent: ET.Element, pid: str, keys: Sequence[Key],
              render: Any = _vector) -> None:
    """A keyed track. One ``<Key frame=.. interp=..>`` per key, in frame order."""
    el = _prop(parent, pid)
    for k in keys:
        ET.SubElement(el, 'Key', {'frame': str(int(k.frame)), 'interp': k.interp}
                      ).text = render(k.value)


def _matrix_text(m: Any) -> str:
    """A 4x4 as the 16 comma-separated numbers Silhouette writes, row-major."""
    a = np.asarray(m, np.float64).reshape(4, 4)
    return '(' + ','.join(num(x) for x in a.reshape(-1)) + ')'


def _new_uuid(existing: str | None) -> str:
    """Keep the document's own uuid where it has one; mint a stable-looking one where not.

    A uuid is how ``dataset.manifest.resolve`` re-finds a layer across saves, so inventing one
    for a layer that already had one would break the manifest. Inventing one for a layer that
    never had one is required -- Silhouette treats a missing uuid as a malformed object.
    """
    return existing or str(_uuid.uuid4())


# ---- the object tree -----------------------------------------------------------

class _Ids:
    """Monotonic element ids, as Silhouette assigns them. Shared across one document."""

    def __init__(self, start: int = 0) -> None:
        self.n = start

    def next(self) -> str:
        self.n += 1
        return str(self.n)


def _shape_element(parent: ET.Element, shape: Shape, ids: _Ids) -> ET.Element:
    """One ``<Object type="Shape">``.

    The property set is the subset the reader consumes plus the ones Silhouette requires to
    open a shape at all (``mode``, ``opacity``, ``invert``, ``strokeWidth``). Properties the
    IR does not model -- blur, adjust, capStyle, channel -- are written at their archive-wide
    defaults rather than omitted, because a missing property reads as *unset* in Silhouette
    rather than as the default, and an unset blend mode is not "Add".
    """
    kind = _SHAPE_TYPE_NAMES.get(shape.shape_type)
    if kind is None:
        raise ValueError(f'shape {shape.name!r}: cannot write shape_type '
                         f'{shape.shape_type!r}; want one of {sorted(_SHAPE_TYPE_NAMES)}')
    if not shape.path:
        raise ValueError(f'shape {shape.name!r} has no path keys; nothing to write')

    el = ET.SubElement(parent, 'Object', {
        'type': 'Shape', 'id': ids.next(), 'label': shape.name, 'expanded': 'True',
        'uuid': _new_uuid(shape.uuid), 'shape_type': kind})
    props = ET.SubElement(el, 'Properties')
    _value_prop(props, 'note', '')

    # path: one <Key> per keyframe, each holding a full <Path> with every control point.
    # ``closed`` and ``type`` are written on every key even though the reader only reads them
    # off the first -- Silhouette writes them per key, and a file that disagrees with itself
    # about whether a shape is closed is a file nobody can debug.
    path = _prop(props, 'path')
    for k in shape.path:
        key = ET.SubElement(path, 'Key', {'frame': str(int(k.frame)), 'interp': k.interp})
        pel = ET.SubElement(key, 'Path', {'closed': str(bool(shape.closed)), 'type': kind})
        for row in np.asarray(k.value, np.float64):
            ET.SubElement(pel, 'Point').text = _pairs(row)

    _value_prop(props, 'color', '(1.000000,1.000000,1.000000)')
    _value_prop(props, 'mode', shape.blend or ADD)
    _value_prop(props, 'blur', '0')
    _value_prop(props, 'blurType', 'Centered')
    _value_prop(props, 'adjust', '0')
    if shape.opacity:
        _key_prop(props, 'opacity', shape.opacity)
    else:
        _value_prop(props, 'opacity', '100')
    if shape.feather:
        _key_prop(props, 'feather', shape.feather)
    else:
        _value_prop(props, 'feather', '0')
    _value_prop(props, 'invert', 'true' if shape.invert else 'false')
    _value_prop(props, 'motionBlur', 'true')
    _value_prop(props, 'outlineColor', '(1.000000,0.000000,0.000000)')
    _value_prop(props, 'strokeWidth', num(shape.stroke_width), constant=False)
    _value_prop(props, 'capStyle', 'Flat')
    _value_prop(props, 'channel', 'Alpha')
    return el


def _layer_element(parent: ET.Element, layer: Layer, ids: _Ids) -> ET.Element:
    """One ``<Object type="Layer">``, recursing into its children.

    A root the reader synthesised around a node-level shape (``<bare:name>``) is written back
    as the bare shape it wrapped, so the tree that comes out of a round trip is the tree that
    went in rather than one layer deeper each time.
    """
    if layer.name.startswith(BARE_PREFIX) and len(layer.children) == 1 \
            and isinstance(layer.children[0], Shape):
        return _shape_element(parent, layer.children[0], ids)

    el = ET.SubElement(parent, 'Object', {
        'type': 'Layer', 'id': ids.next(), 'label': layer.name, 'expanded': 'True',
        'uuid': _new_uuid(layer.uuid)})
    props = ET.SubElement(el, 'Properties')
    _value_prop(props, 'note', '')
    _value_prop(props, 'color', '(1.000000,1.000000,1.000000)')
    _value_prop(props, 'mode', layer.blend or ADD)
    if layer.opacity:
        _key_prop(props, 'opacity', layer.opacity)
    else:
        _value_prop(props, 'opacity', '100')
    _value_prop(props, 'invert', 'true' if layer.invert else 'false')

    # transform.* first, then the baked matrix, which is the order ``layer_matrix`` composes
    # them in (TRS, then matrix) and the order Silhouette's own files list them in.
    _value_prop(props, 'transform', '')
    for name, default in (('anchor', '(0.000000000,0.000000000,0.000000000)'),
                          ('position', '(0.000000000,0.000000000,0.000000000)'),
                          ('scale', '(1.000000000,1.000000000,1.000000000)'),
                          ('rotate', '0')):
        track = layer.trs.get(name)
        if track:
            _key_prop(props, f'transform.{name}', track)
        else:
            _value_prop(props, f'transform.{name}', default)
    if layer.transform:
        _key_prop(props, 'transform.matrix', layer.transform, render=_matrix_text)
    else:
        # A blank <Value> is how Silhouette writes "no baked matrix", and it is what the
        # reader maps back to ``None``. Writing an identity *key* instead would round trip
        # as a one-key constant track, which is a different document.
        _value_prop(props, 'transform.matrix', '')

    objects = _prop(props, 'objects', constant=True, expanded=True)
    for child in layer.children:
        if isinstance(child, Shape):
            _shape_element(objects, child, ids)
        else:
            _layer_element(objects, child, ids)
    return el


# ---- the whole project ---------------------------------------------------------

def split_dialect(dialect: str | None) -> tuple[str, str]:
    """``'2022.5/zlib'`` -> ``('2022.5', 'zlib')``. A bare version picks its own container."""
    text = dialect or DEFAULT_DIALECT
    version, _, container = text.partition('/')
    container = container or DIALECT_CONTAINER.get(version, PLAIN)
    if container not in (PLAIN, ZLIB):
        raise ValueError(f'unknown container {container!r} in dialect {text!r}')
    return version, container


def _session_properties(session: ET.Element, doc: RotoDoc) -> ET.Element:
    props = ET.SubElement(session, 'Properties')
    _value_prop(props, 'note', '')
    _value_prop(props, 'size', f'({num(doc.width)}.000000000,{num(doc.height)}.000000000)')
    _value_prop(props, 'pixelAspect', '1')
    _value_prop(props, 'frameRate', num(doc.frame_rate))
    _value_prop(props, 'duration', num(doc.duration))
    _value_prop(props, 'startFrame', num(doc.start_frame))
    _value_prop(props, 'pixelFormat', 'RGB8')
    _value_prop(props, 'fieldMode', 'false')
    return props


def _source_item(project: ET.Element, doc: RotoDoc, ids: _Ids) -> None:
    """The ``<SourceItem>`` carrying width and height.

    Not decoration: ``read_sfx`` takes the document's resolution from the first ``<Source>``
    with a ``width`` attribute and raises without one, because every coordinate in the IR is
    normalised by the image height. A project with no source has no scale.
    """
    item = ET.SubElement(project, 'Item', {
        'type': 'SourceItem', 'id': ids.next(), 'label': doc.source_label or 'plate',
        'expanded': 'True', 'uuid': str(_uuid.uuid4())})
    src = ET.SubElement(item, 'Source', {
        'type': 'FileSource', 'id': ids.next(), 'label': doc.source_label or 'plate',
        'expanded': 'True', 'uuid': str(_uuid.uuid4()), 'frame': '0',
        'width': str(int(doc.width)), 'height': str(int(doc.height))})
    ET.SubElement(src, 'Path').text = doc.source_path or None
    props = ET.SubElement(src, 'Properties')
    _value_prop(props, 'note', '')
    _value_prop(props, 'layer', '')
    _value_prop(props, 'fieldHandling', 'None')
    _value_prop(props, 'fieldDominance', 'Even')


def serialise(doc: RotoDoc, dialect: str | None = None, label: str = 'roto') -> bytes:
    """``RotoDoc`` -> the XML bytes of a standalone project.

    The skeleton is the minimum ``read_sfx`` and Silhouette both need: a ``Project``, one
    ``Session`` carrying the frame range and resolution, one ``RotoNode`` holding the objects,
    and one ``SourceItem`` carrying the resolution. No pipes and no output node -- this file is
    the *representation*, and :func:`rewrite_sfx` is the one that keeps a real node graph.
    """
    version, _ = split_dialect(dialect or doc.dialect)
    ids = _Ids()
    project = ET.Element('Project', {
        'type': 'Project', 'version': version, 'id': ids.next(), 'label': label,
        'selected': 'True', 'expanded': 'True', 'uuid': str(_uuid.uuid4())})
    _value_prop(ET.SubElement(project, 'Properties'), 'note', '')

    item = ET.SubElement(project, 'Item', {
        'type': 'SessionItem', 'id': ids.next(), 'label': 'Session', 'expanded': 'True',
        'uuid': str(_uuid.uuid4())})
    session = ET.SubElement(item, 'Session', {
        'type': 'Session', 'id': ids.next(), 'label': 'Session', 'selected': 'True',
        'expanded': 'True', 'uuid': str(_uuid.uuid4()), 'frame': '0', 'frameStep': '1'})
    sprops = _session_properties(session, doc)

    nodes = _prop(sprops, 'nodes')
    node = ET.SubElement(nodes, 'Node', {
        'type': 'RotoNode', 'id': ids.next(), 'label': 'Roto', 'expanded': 'True',
        'uuid': str(_uuid.uuid4()), 'active': 'True'})
    nprops = ET.SubElement(node, 'Properties')
    _value_prop(nprops, 'note', '')
    objects = _prop(nprops, 'objects', constant=True, expanded=True)
    for root in doc.roots:
        _layer_element(objects, root, ids)

    _source_item(project, doc, ids)

    ET.indent(project, space='\t')
    return HEADER + ET.tostring(project, encoding='utf-8', xml_declaration=False)


def encode_container(xml: bytes, container: str) -> bytes:
    """Wrap XML bytes in the container a dialect uses. Inverse of ``read.read_xml_bytes``."""
    if container == PLAIN:
        return xml
    return struct.pack('>I', len(xml)) + zlib.compress(xml, 9)


def write_sfx(doc: RotoDoc, path: str | Path, dialect: str | None = None) -> Path:
    """Serialise ``doc`` to a standalone Silhouette project at ``path``.

    ``dialect`` overrides ``doc.dialect``; both are ``'<version>/<container>'``, and a
    document read from the archive round-trips into the container it came from.
    """
    _, container = split_dialect(dialect or doc.dialect)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_container(serialise(doc, dialect or doc.dialect, out.stem),
                                     container))
    return out


# ---- the surgical form ---------------------------------------------------------

def rewrite_sfx(template: str | Path, doc: RotoDoc, path: str | Path,
                layers: Iterable[str] | None = None) -> Path:
    """The artist's own project with named top-level layers replaced by ours.

    Everything outside the substituted layers is preserved **byte for byte in structure**: the
    plate, the node graph, the pipes, the session settings, the other roto layers, the stats.
    So when this is opened in a seat, a difference in the render is a difference in our shapes
    -- which is the only question worth a licence.

    ``layers`` names the top-level layers to replace, defaulting to every root in ``doc``.
    Matching is by uuid where the template has one and by label otherwise, and a name that
    matches no layer in the template raises rather than being silently skipped: writing a file
    that quietly kept the artist's own shapes would score 1.000 and prove nothing.
    """
    from .read import read_xml_bytes

    xml, container = read_xml_bytes(template)
    root = ET.fromstring(xml)
    wanted = list(layers) if layers is not None else [r.name for r in doc.roots]
    by_name = {r.name: r for r in doc.roots}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise KeyError(f'no such layer in the replacement doc: {missing}')

    # Highest id in the template, so new elements cannot collide with existing ones.
    highest = max((int(el.get('id')) for el in root.iter() if (el.get('id') or '').isdigit()),
                  default=0)
    ids = _Ids(highest)

    replaced: list[str] = []
    for node in root.iter('Node'):
        if node.get('type') != 'RotoNode':
            continue
        container_el = node.find('Properties')
        if container_el is None:
            continue
        objects = next((p for p in container_el.findall('Property')
                        if p.get('id') == 'objects'), None)
        if objects is None:
            continue
        for i, obj in enumerate(list(objects)):
            if obj.tag != 'Object' or obj.get('type') != 'Layer':
                continue
            label = obj.get('label')
            if label not in wanted:
                continue
            new = _layer_element(objects, by_name[label], ids)
            # ``_layer_element`` appends; move it into the slot the original occupied so
            # draw order -- which decides what a Subtract subtracts from -- is unchanged.
            objects.remove(new)
            objects.remove(obj)
            objects.insert(i, new)
            # Keep the template's own uuid: it is what Silhouette's own tracking data and
            # our manifest both key on.
            if obj.get('uuid'):
                new.set('uuid', obj.get('uuid'))
            replaced.append(label)

    if sorted(replaced) != sorted(wanted):
        raise KeyError(f'template {Path(template).name} has no top-level layer(s) '
                       f'{sorted(set(wanted) - set(replaced))}; it has '
                       f'{sorted(o.get("label") for n in root.iter("Node") for p in ((n.find("Properties") or ET.Element("x")).findall("Property")) if p.get("id") == "objects" for o in p if o.get("type") == "Layer")}')

    ET.indent(root, space='\t')
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_container(
        HEADER + ET.tostring(root, encoding='utf-8', xml_declaration=False), container))
    return out
