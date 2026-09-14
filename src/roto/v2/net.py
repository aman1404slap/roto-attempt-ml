"""v2 geometry network: alpha in, control points and a transform track out (charter S4, S0).

Ported from ``roto.model.net`` at v2 Step 2 and owned by v2 from here. It is a port rather
than an import because charter S4 adds a head at every stage -- key timing at S1, lifespans and
point counts at S2, dynamic queries at S3 -- so the two diverge immediately, and v2 owning its
network is what lets v1's be deleted.

Everything v1 measured is carried forward, and the docstrings below state what each decision
cost when it was made the other way. Three are load bearing:

**Shape queries are per (element, shape), not per shape index.** One embedding table indexed by
position in document order, shared across elements, stalls at ~64 px: nothing in a union alpha
can tell "shape 500 of this element" from "shape 500 of that one", because the picture shows a
filled silhouette rather than a labelled contour.

This is memorisation capacity, and at S0 that is the point. It is also the ceiling the v2
tracker records: 875 rows for the Step 1 subset, but 27,120 for all 50 shots with one element
alone wanting 9,008. Charter S4's S3 replaces this table with queries the encoder produces, and
the gap between the two is the honest measure of what the encoder still has to learn.

**The point head is a direct projection, not a query-slot outer product.** ``MLP([query,
slot]) -> 2`` is a rank-limited outer product of one per-shape vector with one per-slot vector,
and it stalls: point error plateaus while every other head converges, because 93 distinct point
slots on one shape cannot be separated by two 192-d vectors.

**Coordinates are predicted in crop space** -- [0,1] across the alpha the network is given --
and converted back to the IR's local normalised space afterwards. See :mod:`roto.geometry`:
local coordinates are absolute document positions whose range is up to 8x the crop, and
regressing them directly was measured to stall at ~260 px.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TABLE, SLOTS = 'table', 'slots'
QUERY_MODES = (TABLE, SLOTS)
"""Where a shape query comes from. ``table`` is one learned row per ``(element, shape)`` --
memorisation capacity, and what S0 through S2 use. ``slots`` is charter S4's **S3**: one bank
shared across every element, so identity has to be inferred from the picture instead of looked
up. The gap between the two is the honest measure of what the encoder still has to learn, and
on the S2 checkpoints it is 0.9652 against 0.0377."""


class Prediction(NamedTuple):
    """What one forward pass produces. Fields are ``None`` for heads the model does not carry.

    A NamedTuple rather than a bare tuple because charter S4 adds a head at every stage: S1
    added key timing, S2 adds lifespans and point counts, S3 will add more. Positional
    unpacking would make each of those a silent renumbering at every call site, and the one
    thing worse than a missing head is a caller reading the wrong one. Fields are named, and a
    caller that wants a head it did not ask for gets ``None`` rather than a neighbour.
    """
    points: torch.Tensor
    """``(B, S, Pmax, Cmax, 2)`` control points in crop space."""
    affine: torch.Tensor
    """``(B, G, affine_dim)`` the transform track."""
    key: torch.Tensor | None
    """``(B, S)`` key-timing logits (S1)."""
    alive: torch.Tensor | None = None
    """``(B, S)`` lifespan logits (S2): is this shape on screen at this frame."""
    count: torch.Tensor | None = None
    """``(B, S, count_classes)`` point-count logits (S2). Class index **is** the point count,
    so the head is read with ``argmax`` and no lookup table travels with the checkpoint."""


def sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """Fixed 2D sinusoidal position encoding, ``(h*w, dim)``."""
    assert dim % 4 == 0
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                          torch.arange(w, dtype=torch.float32), indexing='ij')
    omega = torch.exp(torch.arange(dim // 4, dtype=torch.float32)
                      * -(math.log(10000.0) / max(1, dim // 4 - 1)))
    out = []
    for grid in (y.reshape(-1), x.reshape(-1)):
        a = grid[:, None] * omega[None]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, dim=1)


class AlphaEncoder(nn.Module):
    """256px alpha -> 16x16 feature tokens. Strided from the first layer to stay CPU-viable.

    ``in_frames`` is the temporal window: the number of consecutive alphas stacked as
    channels. At 1 the network sees a single frame and its prediction error is independent
    frame to frame, which is the jitter the whole pipeline downstream was built to absorb --
    9-frame smoothing, then a 1 px key tolerance sitting above the noise rather than above
    the artist's precision. Showing it 3 consecutive frames lets it steady its own hand
    instead, and costs one convolution's worth of input channels.
    """

    def __init__(self, dim: int = 192, in_frames: int = 1) -> None:
        super().__init__()
        c1, c2, c3 = dim // 8, dim // 4, dim // 2
        self.in_frames = in_frames
        self.stem = nn.Sequential(
            nn.Conv2d(in_frames, c1, 5, stride=2, padding=2), nn.GroupNorm(4, c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GroupNorm(8, c2), nn.GELU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.GroupNorm(8, c2), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GroupNorm(8, c3), nn.GELU(),
            nn.Conv2d(c3, c3, 3, padding=1), nn.GroupNorm(8, c3), nn.GELU(),
            nn.Conv2d(c3, dim, 3, stride=2, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.register_buffer('pos', sincos_2d(16, 16, dim), persistent=False)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        f = self.stem(alpha)                      # (B, D, 16, 16)
        t = f.flatten(2).transpose(1, 2)          # (B, 256, D)
        return t + self.pos[None].to(t.dtype)


class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.n1(q)
        q = q + self.attn(h, kv, kv, need_weights=False)[0]
        return q + self.ff(self.n2(q))


class SelfBlock(nn.Module):
    """Self-attention among shape queries, so shapes can negotiate rather than collide.

    v1 left this out for a measured reason: on a 16-core CPU with no GPU, O(S^2) at S=1036
    put a single epoch out of reach. The cost of its absence was also measured -- 0.97+ soft
    IoU at <=75 shapes against 0.73-0.78 at 592 and 1036, with two queries claiming one
    contour showing up as the scribbles on the contact sheet. On the GPU the same attention
    is 24 ms per call in 0.19 GB at S=1036, batch 6, so the tradeoff that justified dropping
    it no longer holds.

    No attention mask is needed: a batch is drawn from a single layer, so every query in it
    is a real shape rather than padding.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        h = self.norm(q)
        return q + self.attn(h, h, h, need_weights=False)[0]


class RotoNetV2(nn.Module):
    """Alpha -> control points per shape, a transform track per group, and one optional head
    per charter S4 stage: key timing (S1), lifespan and point count (S2). See
    :class:`Prediction`."""

    def __init__(self, max_shapes: int, max_groups: int, max_points: int,
                 max_coords: int, dim: int = 192, depth: int = 3,
                 in_frames: int = 1, self_attn: bool = False, affine_dim: int = 6,
                 align_window: bool = False, affine_depth: int | None = None,
                 key_head: bool = False, alive_head: bool = False,
                 count_head: bool = False, desc_dim: int = 3,
                 query_mode: str = 'table') -> None:
        super().__init__()
        self.max_points, self.max_coords = max_points, max_coords
        self.in_frames, self.self_attn = in_frames, self_attn
        self.affine_dim, self.align_window = affine_dim, align_window
        self.affine_depth = int(affine_depth or depth)
        self.key_head = bool(key_head)
        self.alive_head = bool(alive_head)
        self.count_head = bool(count_head)
        self.desc_dim = int(desc_dim)
        self.query_mode = query_mode
        if query_mode not in QUERY_MODES:
            raise ValueError(f'unknown query_mode {query_mode!r}, want one of {QUERY_MODES}')
        self.encoder = AlphaEncoder(dim, in_frames)
        # ``table``: one row per (element, shape) -- memorisation capacity, and S0-S2's.
        # ``slots``: ``max_shapes`` rows shared by **every** element, so a row can no longer
        # mean "shape 7 of this layer" and everything element-specific has to arrive through
        # the cross-attention with the alpha. That substitution is the whole of S3.
        self.shape_bank = nn.Embedding(max_shapes, dim)
        # Plan section 6's "query init from encoder tokens (learned pooling)". Mean-pooled
        # tokens projected and added to every slot, so the slots start conditioned on *this*
        # picture rather than identical across the dataset; the cross-attention stack then
        # separates them. Small on purpose -- if a slot bank plus a global summary were
        # enough, the cross-attention would have nothing left to do, and the point of the
        # rung is to find out how much it can do.
        self.token_pool = nn.Linear(dim, dim) if query_mode == SLOTS else None
        self.group_bank = nn.Embedding(max_groups, dim)
        # 3 wide is (n_points, closed, coords_per_point) -- S0 and S1. 2 wide drops
        # ``n_points``, which is charter S4's S2 training wheel coming off: at S2 the declared
        # given structure is shape count and identity, and a point count is neither. Recorded
        # in ``arch`` so a 3-wide checkpoint keeps loading unchanged.
        self.desc = nn.Linear(self.desc_dim, dim)
        self.blocks = nn.ModuleList([CrossBlock(dim) for _ in range(depth)])
        self.sblocks = nn.ModuleList([SelfBlock(dim) for _ in range(depth)]) \
            if self_attn else None
        self.gblocks = nn.ModuleList([CrossBlock(dim) for _ in range(self.affine_depth)])
        self.point_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, max_points * max_coords * 2))
        self.affine_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, affine_dim))
        # One logit per (frame, shape). Small on purpose: the point head needs a direct
        # projection because 93 point slots cannot be separated by two vectors, but "is this
        # frame a key" is one number and the query already carries everything it depends on.
        self.key_logit = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        ) if key_head else None
        # Lifespan (S2), shaped exactly like the key head: one number per (frame, shape), off
        # the query token that has already looked at this frame's alpha. "Is this shape on
        # screen now" is one bit and the query carries what it depends on.
        #
        # It is a harder question than it looks, and not for want of capacity. A layer's matte
        # is the **union** of its shapes, so a shape that has just gone dark inside a pile of
        # overlapping ones changes the picture by nothing at all -- the signal the head needs
        # is sometimes not in the input. That is a ceiling on this head, not a bug in it, and
        # ``v2-s2-design-note.md`` section 2.5 measures where it bites.
        self.alive_logit = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        ) if alive_head else None
        # Point count (S2). One logit per possible count, class index == the count, so
        # ``argmax`` reads it and no vocabulary travels with the checkpoint. Wider than the two
        # binary heads because it is a max_points-way decision rather than a bit, and it reads
        # the query *before* the point head's projection so the two do not have to agree by
        # construction -- if they disagree, that is a fact worth being able to see.
        self.count_logits = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, max_points + 1)
        ) if count_head else None
        # Predictions are in crop space, where the centre of the picture is 0.5. Starting at
        # zero would put every control point in the top-left corner and spend the first
        # thousand steps translating rather than shaping.
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.constant_(self.point_head[-1].bias, 0.5)

    def forward(self, alpha: torch.Tensor, shape_ids: torch.Tensor,
                group_ids: torch.Tensor, desc: torch.Tensor) -> Prediction:
        """``alpha`` (B,in_frames,256,256); ``shape_ids`` (B,S); ``group_ids`` (B,G);
        ``desc`` (B,S,``desc_dim``).

        Returns a :class:`Prediction`. Every head's slot is always present rather than
        conditionally absent, so a caller which ignores one still has to say so; a head the
        model does not carry reads ``None`` rather than a neighbour's tensor.
        """
        tokens = self.encoder(alpha)
        if self.query_mode == SLOTS:
            # No ``desc`` term. ``closed`` and ``coords_per_point`` are per-(element, shape)
            # facts, so feeding them to a shared slot would hand back the identity the stage
            # exists to remove. 669 of v003's 675 shapes are closed and 661 are B-splines, so
            # fixing both costs almost nothing -- see the S3 note.
            q = self.shape_bank(shape_ids) + self.token_pool(tokens.mean(1))[:, None, :]
        else:
            q = self.shape_bank(shape_ids) + self.desc(desc)
        for i, blk in enumerate(self.blocks):
            if self.sblocks is not None:
                q = self.sblocks[i](q)            # shapes negotiate, then look at the image
            q = blk(q, tokens)
        g = self.group_bank(group_ids)
        for blk in self.gblocks:
            g = blk(g, tokens)
        B, S, _ = q.shape
        pts = self.point_head(q).view(B, S, self.max_points, self.max_coords, 2)
        aff = self.affine_head(g)
        head = lambda m: m(q).squeeze(-1) if m is not None else None
        return Prediction(pts, aff, head(self.key_logit), head(self.alive_logit),
                          self.count_logits(q) if self.count_logits is not None else None)
