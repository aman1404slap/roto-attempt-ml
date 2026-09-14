"""v2: the 50-shot POC, built on `data/spline_dataset_08_25_26`.

Governed by ``v2_charter.md`` (rules) and ``v2_implementation_plan.md`` (order), tracked in
``v2-tracker.md``.

**This package must not import from v1.** The boundary is enforced by
``tests/test_v2_boundary.py`` so that v1 can be deleted without touching v2. Three tiers:

*Frozen core* -- shared, version-independent, pinned by the exactness ledger. v2 imports these
freely and changes them only under charter S2 (bump a version, trigger one re-baseline):

    roto.ir  roto.sfx.*  roto.render.*  roto.metrics  roto.exr
    roto.program     IR <-> dense arrays; its round trip is a ledger row
    roto.geometry    local <-> crop coordinates, closed-form and exactness-pinned

*Mechanics* -- version-independent machinery that predates v2 and is reused as-is:

    roto.dataset.crop  roto.dataset.arrays  roto.dataset.layers
    roto.keys          the keyframe DP; v2 changes how it is *biased*, not the algorithm

*v1, deletable* -- nothing here may be imported by ``roto.v2``:

    roto.model.*  roto.shots
    roto.dataset.build  roto.dataset.manifest  roto.dataset.splits

``roto.geometry`` was ``roto.model.geometry`` until Step 2. It is coordinate math with no
learning in it, pinned by ``tests/test_geometry.py``, and filing it under ``model`` made a
shared exactness guarantee look like a v1 asset. Moved rather than copied, so there is one
implementation and one test.

``roto.v2.model`` is v2's own network, data path and training loop. It does not extend
``roto.model``: charter S4 adds heads at every stage (key timing at S1, lifespans and point
counts at S2, dynamic queries at S3), so the two diverge immediately, and v2 owning its model
is what lets v1's be deleted. The measured design decisions v1 recorded are carried forward
with attribution in the docstrings that state them.

v2 replaces the other three because each encodes a decision this round remakes: ``shots``
assumes the archive's directory layout, ``manifest`` excludes layers by thresholds calibrated
on 18 archive layers (v2 tags instead of excluding -- charter S6, D4), and ``splits`` holds out
frames and named layers but not whole shots.
"""
