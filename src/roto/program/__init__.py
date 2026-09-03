"""The model's target: an element's spline program as dense arrays, and back again."""
from .tensors import ProgramSpec, ProgramTensors, affine_from_matrix, decode, load_program, matrix_from_affine

__all__ = ['ProgramSpec', 'ProgramTensors', 'affine_from_matrix', 'decode', 'load_program',
           'matrix_from_affine']
