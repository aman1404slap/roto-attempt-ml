"""``TrainConfig``: what a run is, as data.

Split out of :mod:`roto.v2.train` so the configuration can be read without importing torch.
That matters for exactly one caller and it is a real one: the service validates an incoming
run's rung before it launches anything, and a web process that has to import the training
stack to check a name against a list is a web process that needs a GPU image.

Nothing here computes. Every field is a number, a string or a flag, and the docstrings under
them are the measured reasons each default is what it is -- which is why they live with the
fields rather than in a note that can drift from them.
"""
from __future__ import annotations

from dataclasses import dataclass


DOC, CROP = 'doc', 'crop'
AFFINE_SPACES = (DOC, CROP)


RANDOM, PAIRS, RUNS = 'random', 'pairs', 'runs'
SAMPLING_MODES = (RANDOM, PAIRS, RUNS)


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(slots=True)
class TrainConfig:
    steps: int = 40000
    batch: int = 6
    lr: float = 3e-4
    dim: int = 192
    depth: int = 3
    affine_weight: float = 0.25
    """Weight on the transform term.

    With ``affine_space='doc'`` this has to be a lift: the entries are O(1) and unitless while
    point error is in pixels, so without one the term is noise next to the geometry term and
    the track never converges. 20 was picked to make it audible, not measured.

    With ``affine_space='crop'`` the term is already in crop pixels, so 1.0 means "a pixel of
    transform error costs what a pixel of point error costs" -- the same reading
    ``temporal_weight`` has. Leaving it at 20 there would price the transform above the
    geometry by 20x."""
    affine_space: str = CROP
    """``'doc'`` (v1: 6 document-space matrix entries, L1 on the entries) or ``'crop'``
    (v1.1: the 8-number local-to-crop projective map, L1 on where it puts five probe points).

    Two separate faults are fixed by the same switch, and both were measured:

    * The document-space target asks the head for translation in normalised *document* units
      from a crop that shows none of that -- the framing that stalled point regression at
      260 px until ``local_to_crop`` fixed it, repeated verbatim.
    * The 6-number form is **lossy**. Two layers carry perspective, and round-tripping the
      artist's own track through 6 numbers renders ``FAM red_1`` at 0.9100 instead of 1.000
      and displaces ``Layer_52`` by 23.8 crop px. That is a ceiling the head could not have
      beaten however well it predicted, and it is most of why v1's head read 0.756.
    """
    in_frames: int = 3
    """Consecutive alphas stacked as input channels. 1 reproduces v1."""
    align_window: bool = True
    """Warp each window neighbour into the anchor's crop window before stacking.

    Off reproduces v1.1's first ladder, where the window was tested *misaligned* and read as
    worthless. See ``data.ElementData.window``: the crop offset twitches 1.09 crop px per step
    on average, which is larger than the 0.77 px of jitter the window was meant to remove, so
    the three channels disagreed about where the shape was by more than the quantity under
    test. Only meaningful when ``in_frames > 1``."""
    self_attn: bool = True
    """Self-attention among shape queries. See ``net.SelfBlock``."""
    affine_depth: int = 0
    """Depth of the transform head's own decoder. ``0`` means ``depth``, which is v1's.

    The v1.1 handover's decision tree says that if the crop-space head still falls short,
    "give the head its own small decoder depth before giving up". It already has its own
    decoder -- ``net.RotoNet.gblocks`` is a separate cross-attention stack, sharing only the
    encoder -- so what this knob actually tests is whether the head is *capacity*-limited,
    and the alternative hypothesis is that it is schedule-limited like everything else here:
    at 40k the transform term is still falling (0.858 crop px and descending, ``final_long_v2``).
    Measured as one rung rather than argued."""
    key_weight: float = 0.0
    """Weight on the key-timing term. ``0`` leaves the head off entirely, which is v1.2.

    Unlike every other term here this one is not in crop pixels and cannot be made so -- it is
    a cross-entropy on a probability. So the weight is a real choice rather than a unit fix,
    and it is set by what the term is worth: at 1.0 the key term is comparable in magnitude to the
    point term at convergence (point error lands near 1 crop px, and a class-balanced BCE at
    this archive's 9.6% positive rate lands in the same order), which is the ratio that lets
    the head learn without the geometry giving anything up. Measured as a rung rather than argued -- see
    ``scripts/train_v13.py``."""
    key_balance: str = 'per_element'
    """Whether the key term's positive weight is one number or one per layer.

    ``'per_element'`` is the default and the fix the first probe asked for; ``'global'`` is what
    that probe ran and is kept so the comparison stays reproducible. See
    :func:`key_positive_rates` for what the probe measured and why one global weight is
    exploitable: key density ranges 13x across these layers, and learning that spread scores
    well on any metric that does not net it out."""
    key_pos_weight: float = 0.0
    """Positive-class weight for the key term. ``0`` means *measure it from the dataset*.

    The archive keys 9.6% of live shape-frames, so an unweighted BCE is minimised at
    "never a key" -- which scores 0.90 accuracy and F1 zero. This is the standard correction,
    ``(1 - p) / p``, and it is measured rather than tuned because it is a property of the
    data. Recorded on the run summary so a run can be read against the rate it was trained
    at."""
    curve_weight: float = 0.5
    """Weight on the polyline term, relative to the point term. 0.5 was the starting point;
    the point term is kept because it is what pins down a control polygon the artist can edit,
    while the curve term is what the render actually judges."""
    point_weight: float = 1.0
    """Weight on the dot-L1 term. ``1.0`` is S0 and S1, where the term is the reference unit
    every other weight is quoted against.

    Charter S4 demotes it at **S2**: as the model starts inventing structure, matching the
    artist's dots one for one stops being the right question and the curve the dots draw
    becomes it (charter L2). Plan section 5 sets the demotion at 0.25x with the curve term
    promoted to primary.

    Split out as its own weight -- rather than scaling the other three up -- so the change is
    one recorded field on the run rather than a reinterpretation of every other number, and so
    ``point_px`` stays readable in crop pixels in the log while its *contribution* changes.

    Measured before the S2 heads exist rather than bundled with them (rung ``s2a``): at S2
    shape count and identity are still given, so index-wise correspondence is still exact and
    the dot term is still legitimate. The demotion is preparation for S3's weakened
    correspondence, and at S2 it can only cost geometry."""
    temporal_weight: float = 1.0
    """Weight on the frame-to-frame consistency term."""
    alive_weight: float = 0.0
    """Weight on the lifespan term, charter S4 stage **S2**. ``0`` leaves the head off, which
    is S0 and S1.

    Like the key term this is a cross-entropy rather than a distance, so the weight is a real
    choice and not a unit fix. 1.0 puts it at the same standing as the key term, which is where
    the S2 rung starts; the design note's stop condition lowers it once if geometry regresses
    beyond the seed spread, and then stops rather than searching."""
    alive_balance: str = 'per_element'
    """``'per_element'`` or ``'global'``, exactly as ``key_balance``, and for a sharper reason.

    The live rate spans **0.093 to 1.000** across v003's trained elements. With one global
    weight the cheapest thing for the head to learn is each element's own base rate, and a per
    ``(element, shape)`` query row is precisely the capacity to do it -- the same exploit the
    key head's first probe fell into, at a wider spread."""
    alive_pos_weight: float = 0.0
    """Positive-class weight for the lifespan term. ``0`` means *measure it from the dataset*.

    ``(1 - r) / r`` at v003's 0.627 live rate is 0.60 -- below 1, because live cells are the
    *majority* here. That is the opposite of the key term's situation and worth stating: this
    head's degenerate answer is "always alive", not "never", and the balancing is what stops it
    being free. An element whose shapes are always live has ``r = 1`` and no negatives at all;
    its weight is pinned to 1.0 rather than to the ``(1-r)/r`` formula's 0, which would zero the
    positive class and delete the element from the term."""
    count_weight: float = 0.0
    """Weight on the point-count cross-entropy, charter S4 stage **S2**. ``0`` leaves the head
    off.

    Reported, never gated -- see ``v2-s2-design-note.md`` section 2.4. A point count does not
    vary with the frame and the query row does, so the row can carry it exactly and accuracy
    here measures memory rather than perception. The head is built at S2 anyway because it and
    its decode have to exist and be debugged somewhere, and S3 -- where identity goes away and
    the number becomes real -- is not the place to be debugging it."""
    count_style_weight: float = 0.0
    """Weight on plan section 5's point-economy style statistic, ``|E[P] - P| / P``.

    Charter L2 asks for point economy as a *soft statistical* target rather than a hard match,
    so this is an L1 on the head's expected count rather than a second cross-entropy. The
    ``/ P`` is deviation 4.3 of the design note and is measured, not chosen: plan section 5's
    unweighted form prices one point the same at P = 68 and at P = 4, and the render does not.
    Dropping a point costs 0.0004-0.053 soft IoU on fifteen of v003's elements and 0.42-0.50 on
    the two whose smallest shape has four points, where a 4-point closed B-spline drops to a
    near-degenerate 3."""
    n_slots: int = 0
    """Charter S4 stage **S3**: share one bank of this many shape queries across every element,
    instead of one learned row per ``(element, shape)``. ``0`` keeps the per-element table,
    which is S0 through S2.

    256 covers v003, whose largest element has 247 shapes. It is a recorded **interface**
    number rather than a hyperparameter -- changing it changes what a checkpoint means -- and
    it is also the reason the deferred 9,008-shape shots re-enter at S3 rather than before: no
    fixed maximum accommodates them, which is a fact about the design and not about the data.

    Turning this on does three things at once, and they are inseparable by construction: the
    query stops carrying identity, the *count* stops being given (a slot the alive head never
    fires on is not a shape), and the shape-type descriptor leaves the input because it is
    per-(element, shape) information. See ``net.QUERY_MODES``."""
    give_point_count: bool = True
    """Whether ``n_points`` is fed to the query as part of ``desc``. ``False`` is S2's training
    wheel coming off.

    Removing it does **not** make the count unmemorisable -- the query row can still carry it,
    and the count loss will put it there. What removal does buy is that the count is no longer
    handed over on a path that bypasses learning entirely, which is charter L1: at S2 the
    declared given structure is shape count and identity, and a point count is neither."""
    holdout_every: int = 0
    """Withhold every Nth frame from training and score it separately. 0 holds nothing.

    Ignored when the dataset records its own split, which ``datasets/v002`` does and
    ``datasets/v001`` does not -- see ``data.ElementData.split`` and ``use_split``."""
    use_split: bool = True
    """Obey the split the dataset recorded at build time. ``False`` trains on everything.

    The default is to obey it, because a split that a run can quietly ignore is not a split.
    The opt-out exists for exactly one row -- the one that has to be comparable to a
    pre-split result -- and it is a *flag on the run* rather than a second dataset, so the
    two rows differ in one recorded field and share every alpha. A run that opts out says so
    on its own checkpoint (``split_source``), so a table can never mix the two silently."""
    sampling: str = 'pairs'
    """How a step's frames are drawn: ``'random'`` (v1), ``'pairs'``, or ``'runs'``.
    See :func:`sample_indices` -- the temporal term needs ``'pairs'`` to have anything to
    act on, and ``'runs'`` is kept only because its cost is worth having on record."""
    sample_weight: str = 'sqrt'
    """``'frames'`` (v1: proportional to frame count) or ``'sqrt'`` (sqrt(frames * shapes)).

    v1's weighting let the two giant layers dominate wall-clock while the tiny ones overfit.
    ``'sqrt'`` compresses that range; which one wins is measured, not assumed."""
    log_every: int = 500
    seed: int = 0
    warmup: int = 200
    device: str | None = None
