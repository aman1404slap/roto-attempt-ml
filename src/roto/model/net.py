"""v1 geometry network: clean alpha in, control points and a transform track out.

Scope, stated plainly because the number this produces is easy to over-read. The shape
*breakdown* is given -- how many shapes, how many control points each, which transform group
each belongs to, and which frames each is alive for. That is the teacher-forced setting the
handoff specifies, and it is what makes v1 a reconstruction result rather than a roto result.
What the network actually has to produce from the picture is the geometry: where every control
point sits, and how each group moves.

Two design choices are load-bearing.

**No self-attention among shape queries.** A decoder that lets 1036 queries attend to each
other costs O(S^2) and, on a 16-core CPU with no GPU, that alone would put a single epoch out
of reach. Queries cross-attend to the image and not to each other. The cost is that shapes
cannot negotiate -- two queries may both claim the same contour -- which is a real limitation
and the first thing to revisit when a GPU is available.

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
    """256px alpha -> 16x16 feature tokens. Strided from the first layer to stay CPU-viable."""

    def __init__(self, dim: int = 192) -> None:
        super().__init__()
        c1, c2, c3 = dim // 8, dim // 4, dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 5, stride=2, padding=2), nn.GroupNorm(4, c1), nn.GELU(),
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


class RotoNet(nn.Module):
    """Alpha -> (control points per shape, 6-DOF affine per group)."""

    def __init__(self, max_shapes: int, max_groups: int, max_points: int,
                 max_coords: int, dim: int = 192, depth: int = 3) -> None:
        super().__init__()
        self.max_points, self.max_coords = max_points, max_coords
        self.encoder = AlphaEncoder(dim)
        self.shape_bank = nn.Embedding(max_shapes, dim)
        self.group_bank = nn.Embedding(max_groups, dim)
        self.desc = nn.Linear(3, dim)             # n_points, closed, coords_per_point
        self.blocks = nn.ModuleList([CrossBlock(dim) for _ in range(depth)])
        self.gblocks = nn.ModuleList([CrossBlock(dim) for _ in range(depth)])
        self.point_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, max_points * max_coords * 2))
        self.affine_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 6))
        # Predictions are in crop space, where the centre of the picture is 0.5. Starting at
        # zero would put every control point in the top-left corner and spend the first
        # thousand steps translating rather than shaping.
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.constant_(self.point_head[-1].bias, 0.5)

    def forward(self, alpha: torch.Tensor, shape_ids: torch.Tensor,
                group_ids: torch.Tensor, desc: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``alpha`` (B,1,256,256); ``shape_ids`` (B,S); ``group_ids`` (B,G); ``desc`` (B,S,3).

        Returns points ``(B, S, Pmax, Cmax, 2)`` and affine ``(B, G, 6)``.
        """
        tokens = self.encoder(alpha)
        q = self.shape_bank(shape_ids) + self.desc(desc)
        for blk in self.blocks:
            q = blk(q, tokens)
        g = self.group_bank(group_ids)
        for blk in self.gblocks:
            g = blk(g, tokens)
        B, S, _ = q.shape
        pts = self.point_head(q).view(B, S, self.max_points, self.max_coords, 2)
        aff = self.affine_head(g)
        return pts, aff
