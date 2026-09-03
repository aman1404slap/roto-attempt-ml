"""IR -> Silhouette ``.sfx``. Out of scope for v1; the seam is here so v2 can drop it in.

**There is no machine learning in this file, and there never will be.** Emitting a ``.sfx``
is serialisation: the model's job ends when it produces a ``RotoDoc``, and everything from
there to a file an artist can open is a deterministic format problem. Keeping that boundary
explicit is the point of this module -- ``decode()`` in ``roto.program`` already returns a
``RotoDoc``, so the model is *already* finished, and adding .sfx output later changes nothing
upstream of here.

What it will take, from what the reader measured:

* **Two containers.** ``2020``/``2022.5``/``2025.5`` write zlib-compressed streams; dialect
  ``5`` writes plain. ``RotoDoc.dialect`` records which one a document came from, so a
  round-trip can write back the dialect it read.
* **B-splines stay B-splines.** 6919 of 6969 measured shapes are B-splines with no
  artist-authored Bezier handles. Converting on write would manufacture parameters the artist
  never chose, and the deliverable is a file the artist *edits*.
* **Per-key interpolation.** The archive is ~75% linear, ~25% catmullrom, varying by shot.
  A global mode silently corrupts every non-key frame.
* **Opacity hold keys are how shapes are born and die.** Not shape insertion/removal. Getting
  this wrong on read cost three points of IoU and looked like a geometry bug.

The blocker is not code. It is that no file we write has ever been opened in Silhouette, and
validating that needs one seat. Until then a writer is unverifiable, which is why v1 stops at
the IR and ``roto parse``'s JSON.
"""
from __future__ import annotations

from pathlib import Path

from ..ir import RotoDoc


def write_sfx(doc: RotoDoc, path: str | Path, dialect: str | None = None) -> Path:
    """Serialise ``doc`` to a Silhouette project. Not implemented in v1."""
    raise NotImplementedError(
        'writing .sfx is out of v1 scope. The model produces a RotoDoc '
        '(roto.program.decode); use roto.sfx.json_ir.write_json_ir for a readable target, '
        'and see this module docstring for what a real writer needs. Validating one '
        'requires a Silhouette seat.')
