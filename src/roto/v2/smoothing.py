"""Temporal smoothing of a predicted track.

Ported from ``roto.smoothing`` at v2 Step 2, unchanged.
"""

from __future__ import annotations

import numpy as np

BOXCAR, SAVGOL = 'boxcar', 'savgol'
KINDS = (BOXCAR, SAVGOL)

SAVGOL_ORDER = 2
"""Polynomial order. 2 keeps a quadratic peak exactly; 3 tracks an inflection as well but
suppresses less noise at these window lengths. The review suggested 2-3; 2 is the measured
default and ``order`` is a parameter so the other can be tried."""


def _odd_window(window: int, n: int) -> int:
    """Largest odd window that fits in a track of ``n`` frames, or 1 to mean 'no filtering'."""
    w = min(window, n if n % 2 else n - 1)
    if w % 2 == 0:
        w -= 1
    return max(1, w)


def _savgol_weights(window: int, order: int) -> np.ndarray:
    """``(window, window)`` matrix: row i gives the weights producing output sample i.

    Interior rows are all the same centred kernel; the first and last ``window // 2`` rows
    are the edge fits. Built from the Vandermonde pseudo-inverse, so it is a least-squares
    polynomial fit by construction rather than a hand-tabulated kernel.
    """
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    V = np.vander(x, order + 1, increasing=True)          # (window, order+1)
    pinv = np.linalg.pinv(V)                              # (order+1, window)
    # Value of the fitted polynomial at offset t is  [t^0..t^order] @ pinv @ y.
    return np.stack([np.vander([t], order + 1, increasing=True)[0] @ pinv for t in x])


def smooth_track(track: np.ndarray, window: int, kind: str = BOXCAR,
                 order: int = SAVGOL_ORDER) -> np.ndarray:
    """Smooth ``track`` (``(T, ...)``) along time. ``window <= 1`` returns it unchanged."""
    if kind not in KINDS:
        raise ValueError(f'unknown smoothing kind {kind!r}, want one of {KINDS}')
    if window <= 1 or len(track) < 3:
        return track
    w = _odd_window(window, len(track))
    if w <= 1:
        return track
    half = w // 2
    flat = track.reshape(len(track), -1)

    if kind == BOXCAR:
        # Centred moving average, edges held -- v1's filter, kept so the two are comparable.
        pad = np.concatenate([np.repeat(flat[:1], half, 0), flat,
                              np.repeat(flat[-1:], half, 0)])
        kernel = np.ones(w) / w
        out = np.stack([np.convolve(pad[:, j], kernel, mode='valid')
                        for j in range(flat.shape[1])], axis=1)
        return out.reshape(track.shape)

    weights = _savgol_weights(w, min(order, w - 1))
    out = np.empty_like(flat)
    centre = weights[half]
    if len(flat) > w:
        # Interior: one centred kernel, applied by correlation.
        stack = np.stack([flat[i:len(flat) - w + 1 + i] for i in range(w)])   # (w, T-w+1, D)
        out[half:len(flat) - half] = np.einsum('w,wtd->td', centre, stack)
    else:
        out[half:len(flat) - half] = weights[half:len(flat) - half] @ flat
    out[:half] = weights[:half] @ flat[:w]
    out[len(flat) - half:] = weights[w - half:] @ flat[-w:]
    return out.reshape(track.shape)
