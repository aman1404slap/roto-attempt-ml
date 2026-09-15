"""v2: the 50-shot POC, built on `data/spline_dataset_08_25_26`.

Governed by ``v2_charter.md`` (rules) and ``v2_implementation_plan.md`` (order), tracked in
``v2-tracker.md``.

**v1 is gone, and the boundary that made that possible stays.** ``roto.v2`` never imported
v1, enforced by ``tests/test_v2_boundary.py``, so removing it was a delete rather than an
untangling. The test remains as a tripwire: the deleted names must not come back. Two tiers
are left:

*Frozen core* -- shared, version-independent, pinned by the exactness ledger. v2 imports these
freely and changes them only under charter S2 (bump a version, trigger one re-baseline):

    roto.ir  roto.sfx.*  roto.render.*  roto.metrics  roto.exr
    roto.program     IR <-> dense arrays; its round trip is a ledger row
    roto.geometry    local <-> crop coordinates, closed-form and exactness-pinned

*Mechanics* -- version-independent machinery that predates v2 and is reused as-is:

    roto.dataset.crop  roto.dataset.arrays  roto.dataset.layers
    roto.keys          the keyframe DP; v2 changes how it is *biased*, not the algorithm
    roto.smoothing     the track smoother, shared by ``dataset.crop`` and v2's own copy

What was deleted: ``roto.model.*``, ``roto.shots``, ``roto.dataset.build``,
``roto.dataset.manifest``, ``roto.dataset.splits``, and every v1 script and test. Each encoded
a decision v2 remakes -- ``shots`` assumes the archive's directory layout, ``manifest``
excludes layers by thresholds calibrated on 18 archive layers (v2 tags instead of excluding --
charter S6, D4), and ``splits`` holds out frames and named layers but not whole shots. v1's
model is gone because charter S4 adds a head at every stage and the two diverged immediately;
the measured design decisions v1 recorded are carried forward with attribution in the
docstrings that state them.

``roto.geometry`` was ``roto.model.geometry`` until Step 2. It is coordinate math with no
learning in it, pinned by ``tests/test_geometry.py``, and filing it under ``model`` made a
shared exactness guarantee look like a v1 asset. Moved rather than copied, so there is one
implementation and one test.
"""
