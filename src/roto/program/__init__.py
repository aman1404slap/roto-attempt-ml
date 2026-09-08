"""The model's target: an layer's spline program as dense arrays, and back again."""
from .tensors import (AFFINE_DOF, PROJ_DOF, ProgramSpec, ProgramTensors, affine_from_matrix,
                      decode, load_program, matrix_from_affine, matrix_from_proj,
                      proj_from_matrix)

__all__ = ['AFFINE_DOF', 'PROJ_DOF', 'ProgramSpec', 'ProgramTensors', 'affine_from_matrix',
           'decode', 'load_program', 'matrix_from_affine', 'matrix_from_proj',
           'proj_from_matrix']
