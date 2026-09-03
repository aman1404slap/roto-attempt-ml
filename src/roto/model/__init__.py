"""v1 model: clean alpha -> spline geometry, with keyframes chosen by roto.keys."""
from .data import LayerData, load_dataset, load_element
from .net import RotoNet
from .train import TrainConfig, train

__all__ = ['LayerData', 'RotoNet', 'TrainConfig', 'load_dataset', 'load_element', 'train']
