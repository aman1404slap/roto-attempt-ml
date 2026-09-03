"""Clean-alpha dataset: top-level layers -> (rendered alpha, spline program) examples."""
from .build import BuiltElement, build, load_alpha, load_meta
from .crop import CropConfig, CropPlan, crop_plan
from .manifest import RotoLayer, LayerStats, discover, resolve

__all__ = ['BuiltElement', 'build', 'load_alpha', 'load_meta', 'CropConfig', 'CropPlan',
           'crop_plan', 'RotoLayer', 'LayerStats', 'discover', 'resolve']
