"""
HandyTree: An interactive decision tree building tool.

This package provides tools for building and analyzing decision trees with
interactive features for exploration and optimization.
"""

__version__ = "0.1.0"

from .tree import HandyTree, Node
from .utils import (
    isin,
    notin,
    operator_map,
    operator_reverse_map,
)
from .metrics import (
    get_precision,
    get_recall,
    information_gain,
    intrinsic_information,
    information_gain_ratio,
    information_value,
)

__all__ = [
    "HandyTree",
    "Node",
    "isin",
    "notin",
    "operator_map",
    "operator_reverse_map",
    "get_precision",
    "get_recall",
    "information_gain",
    "intrinsic_information",
    "information_gain_ratio",
    "information_value",
]
