"""
AsymmeTree: Interactive asymmetric decision trees for business-ready imbalanced classification.

AsymmeTree is an interactive decision tree classifier specifically designed for highly imbalanced datasets. Unlike traditional decision trees that optimize for node purity, AsymmeTree focuses on maximizing precision while capturing sufficient recall, making it ideal for fraud detection, anomaly detection, and other rare event prediction tasks.
"""

__version__ = "0.1.0"

from .tree import AsymmeTree, Node
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
    "AsymmeTree",
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
