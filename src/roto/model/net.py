"""v1 geometry network: clean alpha in, control points and a transform track out.

Scope, stated plainly because the number this produces is easy to over-read. The shape
*breakdown* is given -- how many shapes, how many control points each, which transform group
each belongs to, and which frames each is alive for. That is the teacher-forced setting the
model is given, and it is what makes v1 a reconstruction result rather than a roto result.
What the network actually has to produce from the picture is the geometry: where every control
point sits, and how each group moves.

Two design choices are load-bearing.

**Shape queries are per (layer, shape), not per shape index.** The first design used one
embedding table indexed by position in document order and shared it across layers, so shape
500 of one layer and shape 500 of another drew the same query vector. Nothing in a union
alpha can tell those apart -- the picture shows a filled silhouette, not which contour is
"shape 500" -- so the model had no way to resolve the ambiguity and stalled at ~64 px. Each
layer now owns a contiguous block of the table, 2753 rows in total.

This is memorisation capacity, and for v1 that is the point: the task is to regenerate these
shots, not to generalise to unseen ones. It also means **the v1 number does not transfer**.
v2 must replace these embeddings with queries the encoder produces, and the gap between the
two is the honest measure of what the encoder still has to learn.

**Self-attention among shape queries is optional, and v1.1 turns it on.** v1 left it out
because O(S^2) at S=1036 on a 16-core CPU put a single epoch out of reach, and the cost of
its absence was visible in the numbers: 0.97+ soft IoU at <=75 shapes against 0.73-0.78 at
592 and 1036, with two queries claiming one contour drawn as scribbles on the contact sheet.
Measured on this GPU the same attention is 24 ms per call in 0.19 GB at S=1036 batch 6, so
the constraint that justified dropping it is gone. ``self_attn=False`` reproduces v1 exactly
and is the default, so old checkpoints still load.

**The point head is a direct projection, not a query-slot outer product.** The obvious cheap
head is ``MLP([query, point_slot]) -> 2``, which is a rank-limited outer product of one
per-shape vector with one per-slot vector. It was measured to stall: point error plateaus
while every other head converges, because 93 distinct point slots on one shape cannot be
separated by two 192-d vectors. A direct projection to all ``Pmax * Cmax * 2`` outputs costs
more parameters and actually fits.

Coordinates are predicted in **crop space** -- [0,1] across the alpha the network is given --
and converted back to the IR's local normalised space afterwards. See ``roto.model.geometry``
for why: local coordinates are absolute document positions whose range is up to 8x the crop,
and regressing them directly was measured to stall at ~260 px.

**The transform head is 8-wide in v1.1 and was 6-wide in v1**, because the 6-number affine
target it was trained against is not a lossless description of these transforms -- two layers
carry perspective, and on one of them round-tripping the *artist's own* track through 6
numbers moves control points by 23.8 crop px. ``affine_dim`` selects the width;
``proj_from_matrix``'s 8-number normalised homography is what 8 means. It defaults to 6 so
that v1's checkpoints, whose ``arch`` predates the option, still load and still score the way
they did.

**``align_window`` is carried on the network rather than passed at the call site**, for the
same reason ``in_frames`` is: a model trained on neighbour alphas warped into the anchor's
crop window must be *given* them warped at inference, and a mismatch would surface as a
mysterious quality loss rather than an error. See ``roto.model.data.LayerData.window``.

**The transform head already owns its decoder, and ``affine_depth`` is how deep.** The group
queries run through ``gblocks`` -- their own stack of cross-attention blocks, never shared
with the shape queries; only the encoder is shared. So "give the head its own decoder" was
already true in v1, which matters when reading the v1.1 handover: what the head did *not*
have was a depth of its own, fixed at ``depth`` alongside the point decoder. ``affine_depth``
separates them so more capacity can be spent on motion without touching the geometry path.
``None`` means ``depth``, which is exactly v1 and v1.1, so every existing checkpoint loads
and scores unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RotoNet(nn.Module):
    """Alpha -> (control points per shape, 6-DOF affine per group)."""

    def __init__(self, max_shapes: int, max_groups: int, max_points: int,
                 max_coords: int, dim: int = 192, depth: int = 3,
                 in_frames: int = 1, self_attn: bool = False, affine_dim: int = 6,
                 align_window: bool = False, affine_depth: int | None = None) -> None:
        super().__init__()
        self.max_points, self.max_coords = max_points, max_coords
        self.in_frames, self.self_attn = in_frames, self_attn
        self.affine_dim, self.align_window = affine_dim, align_window
        self.affine_depth = int(affine_depth or depth)
        self.encoder = AlphaEncoder(dim, in_frames)
        self.shape_bank = nn.Embedding(max_shapes, dim)
        self.group_bank = nn.Embedding(max_groups, dim)
        self.desc = nn.Linear(3, dim)             # n_points, closed, coords_per_point
        self.blocks = nn.ModuleList([CrossBlock(dim) for _ in range(depth)])
        self.sblocks = nn.ModuleList([SelfBlock(dim) for _ in range(depth)]) \
            if self_attn else None
        self.gblocks = nn.ModuleList([CrossBlock(dim) for _ in range(self.affine_depth)])
        self.point_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, max_points * max_coords * 2))
        self.affine_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, affine_dim))
        # Predictions are in crop space, where the centre of the picture is 0.5. Starting at
        # zero would put every control point in the top-left corner and spend the first
        # thousand steps translating rather than shaping.
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.constant_(self.point_head[-1].bias, 0.5)

    def forward(self, alpha: torch.Tensor, shape_ids: torch.Tensor,
                group_ids: torch.Tensor, desc: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``alpha`` (B,in_frames,256,256); ``shape_ids`` (B,S); ``group_ids`` (B,G); ``desc`` (B,S,3).

        Returns points ``(B, S, Pmax, Cmax, 2)`` and the transform track
        ``(B, G, affine_dim)``.
        """
        tokens = self.encoder(alpha)
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
        return pts, aff
