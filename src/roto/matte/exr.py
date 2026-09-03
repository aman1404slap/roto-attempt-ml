"""Defensive matte loader.

The delivered EXRs are inconsistent across vendors in ways that fail *silently*:

* channels: RGB or RGBA, with unused slots left empty
* one channel per element, packed arbitrarily; no channel names, no layer names
* every channel is 8-bit quantised (256 levels) despite half-float storage
* two of six shots store ``sRGB_to_linear(k/255)`` instead of ``k/255`` -- a colour
  transform wrongly applied to an alpha channel. Nothing in the header marks this, so a
  naive read halves every soft edge.

So we sniff rather than trust. See FINDINGS.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402  (must follow the env var)

_K = np.arange(256) / 255.0
_SRGB_LEVELS = np.sort(np.float16(
    np.where(_K <= 0.04045, _K / 12.92, ((_K + 0.055) / 1.055) ** 2.4)).astype(np.float32))
_LINEAR_LEVELS = np.sort(np.float16(_K).astype(np.float32))

CHANNEL_INDEX = {'B': 0, 'G': 1, 'R': 2, 'A': 3}


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def detect_encoding(channel: np.ndarray, tol: float = 1e-6) -> str:
    """``'srgb'``, ``'linear'``, or ``'unknown'`` for one channel's value set."""
    vals = np.unique(channel)
    if len(vals) <= 1:
        return 'linear'
    rounded = np.round(vals, 6)
    if np.isin(rounded, np.round(_SRGB_LEVELS, 6)).mean() > 0.98:
        return 'srgb'
    if np.isin(rounded, np.round(_LINEAR_LEVELS, 6)).mean() > 0.98:
        return 'linear'
    return 'unknown'


def load_matte(path: str | Path, channel: str = 'R', decode: bool = True
               ) -> tuple[np.ndarray, str]:
    """Load one channel as coverage in [0,1]. Returns ``(alpha, encoding)``.

    With ``decode=True`` an sRGB-encoded channel is converted back to true coverage.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'could not read EXR: {path}')
    if img.ndim == 2:
        img = img[:, :, None]
    idx = CHANNEL_INDEX[channel.upper()]
    if idx >= img.shape[2]:
        raise KeyError(f'{path}: no channel {channel!r} (file has {img.shape[2]})')
    a = img[..., idx].astype(np.float32)
    enc = detect_encoding(a)
    if decode and enc == 'srgb':
        a = linear_to_srgb(a).astype(np.float32)
    return np.clip(a, 0.0, 1.0), enc


_CACHE: dict[tuple[str, float, bool], dict[str, np.ndarray]] = {}
_CACHE_MAX = 8


def load_channels(path: str | Path, scale: float = 1.0, decode: bool = True,
                  threshold: float = 0.005) -> dict[str, np.ndarray]:
    """Every channel that carries a matte, as ``{name: alpha}``, optionally downscaled.

    Empty channels are dropped rather than returned as zeros, because "which channels hold
    an element" is itself unknown for these deliveries -- see eval.match. Results are cached
    because layer/channel matching re-reads the same few frames many times over.
    """
    key = (str(path), float(scale), bool(decode))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'could not read EXR: {path}')
    if img.ndim == 2:
        img = img[:, :, None]
    if scale != 1.0:
        img = cv2.resize(img, (int(round(img.shape[1] * scale)),
                               int(round(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, None]

    out: dict[str, np.ndarray] = {}
    for name, idx in CHANNEL_INDEX.items():
        if idx >= img.shape[2]:
            continue
        a = img[..., idx].astype(np.float32)
        if float(a.max()) <= threshold:
            continue
        # Detect on the full value set *before* clipping; downscaling has already mixed
        # levels, so fall back to the unscaled file when the sniff is inconclusive.
        enc = detect_encoding(a)
        if enc == 'unknown' and scale != 1.0:
            enc = detect_encoding(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., idx]
                                  .astype(np.float32))
        if decode and enc == 'srgb':
            a = linear_to_srgb(a).astype(np.float32)
        out[name] = np.clip(a, 0.0, 1.0)

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def occupied_channels(path: str | Path, threshold: float = 0.001) -> list[str]:
    """Which channels actually carry a matte, rather than being empty padding."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    names = [n for n, i in sorted(CHANNEL_INDEX.items(), key=lambda kv: kv[1])]
    return [n for n, i in zip(names, range(img.shape[2]))
            if float(img[..., i].max()) > threshold]


def sequence(folder: str | Path) -> list[Path]:
    """Frame-sorted EXR paths in a matte folder."""
    files = sorted(Path(folder).glob('*.exr'))
    if not files:
        raise FileNotFoundError(f'no .exr files in {folder}')
    return files
