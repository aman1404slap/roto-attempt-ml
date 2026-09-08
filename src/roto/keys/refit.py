"""Key *values* fitted to the raw track, once the key *frames* are chosen.

v1 conflated two jobs in one filter. Smoothing was applied so the keyframe search could see
motion instead of noise -- correct -- and then the key values were stored *from the smoothed
track*, which quietly bakes the filter's bias into the answer. Any smoothing attenuates
motion extremes, so a reconstructed bounce lands short of the artist's.

Splitting the two jobs costs one least-squares solve. Key *timing* still comes from the
denoised track, because that is a question about structure and noise would corrupt it. Key
*values* are then fitted to the **raw** track: keep the chosen frames, and ask what values at
those frames make the interpolated curve pass as close as possible to what the model actually
predicted.

That is a linear least-squares problem, and exactly linear. The renderer's interpolation --
linear, hold, and Catmull-Rom alike -- is a linear function of the key values, so the track it
produces is ``A @ values`` for a matrix ``A`` depending only on the key frames and their
interpolation modes. ``basis_matrix`` builds ``A`` by evaluating ``ir.sample`` itself on unit
key values rather than re-deriving the bases here, so it cannot drift from what the renderer
draws.
"""
from __future__ import annotations

import numpy as np

from ..ir import Key, sample


def basis_matrix(key_frames: np.ndarray, interp_modes: list[str],
                 frames: np.ndarray) -> np.ndarray:
    """``(T, K)``: column j is the track produced by a unit value on key j alone.

    Because ``ir.sample`` is linear in the key values, any track this key set can express is
    a combination of these columns -- which is what makes the refit a least-squares problem
    rather than an optimisation.
    """
    cols = []
    for j in range(len(key_frames)):
        keys = [Key(int(f), interp_modes[i], np.array([1.0 if i == j else 0.0]))
                for i, f in enumerate(key_frames)]
        cols.append(np.array([float(np.ravel(sample(keys, int(f)))[0]) for f in frames]))
    return np.stack(cols, axis=1) if cols else np.zeros((len(frames), 0))


def refit_key_values(raw: np.ndarray, key_frames: np.ndarray, interp_modes: list[str],
                     frames: np.ndarray) -> np.ndarray:
    """Least-squares key values reproducing ``raw`` through the chosen keys.

    ``raw`` is ``(T, ...)`` on ``frames``; the result is ``(K, ...)`` on ``key_frames``.
    With a key on every frame the fit is exact and returns ``raw`` itself, so this is safe to
    apply unconditionally -- it can only reduce the interpolated track's distance to ``raw``,
    in the least-squares sense, relative to sampling ``raw`` at the key frames.
    """
    A = basis_matrix(key_frames, interp_modes, frames)
    if A.shape[1] == 0:
        return raw[:0]
    flat = raw.reshape(len(raw), -1)
    x, *_ = np.linalg.lstsq(A, flat, rcond=None)
    return x.reshape((A.shape[1],) + raw.shape[1:])
