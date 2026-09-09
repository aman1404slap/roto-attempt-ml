"""Keyframe selection by curve simplification."""
from .dp import (MAX_TOL_SCALE, MIN_TOL_SCALE, Selection, f1, local_tolerances,
                 segment_errors, segment_ratio, select)

__all__ = ['MAX_TOL_SCALE', 'MIN_TOL_SCALE', 'Selection', 'f1', 'local_tolerances',
           'segment_errors', 'segment_ratio', 'select']
