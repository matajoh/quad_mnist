"""Multi-class training with quadrant MNIST."""

from .dataset import MultilabelDataset
from .model import FCNMulti

__all__ = [
    "MultilabelDataset",
    "FCNMulti"
]
