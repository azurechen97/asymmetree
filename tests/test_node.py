import pytest
import json
import pandas as pd
import numpy as np
from asymmetree import Node


class TestNode:
    """Test cases for Node class."""

    def test_node_init_defaults(self):
        """Test Node initialization with default values."""
        node = Node()

        assert node.index is None
        assert node.parent is None
        assert node.children is None
        assert node.split_feature is None
        assert node.operator is None
        assert node.split_value is None
        assert node.is_leaf is True
        assert node.prediction is None
        assert node.metrics is None
        assert node.depth is None
        assert node.id is None

    def test_node_init_with_values(self):
        """Test Node initialization with custom values."""
        index = pd.Index([0, 1, 2])
        node = Node(
            index=index,
            split_feature="age",
            operator=">=",
            split_value=25,
            is_leaf=False,
            prediction=1,
            depth=2,
            id=5,
        )

        assert node.index is index
        assert node.split_feature == "age"
        assert node.operator == ">="
        assert node.split_value == 25
        assert node.is_leaf is False
        assert node.prediction == 1
        assert node.depth == 2
        assert node.id == 5

    def test_node_repr(self):
        """Test Node string representation."""
        node = Node(
            id=1,
            depth=0,
            is_leaf=True,
            split_feature="income",
            operator="<",
            split_value=50000,
            prediction=0,
        )

        repr_str = repr(node)
        assert "Node(" in repr_str
        assert "id=1" in repr_str
        assert "depth=0" in repr_str
        assert "is_leaf=True" in repr_str
        assert "split_feature=income" in repr_str

    def test_node_str(self):
        """Test Node string method."""
        node = Node(id=2, depth=1)
        assert str(node) == repr(node)

    def test_node_to_dict_basic(self):
        """Test Node to_dict conversion with basic attributes."""
        node = Node(
            id=1,
            depth=0,
            is_leaf=True,
            split_feature="feature1",
            operator=">=",
            split_value=10,
            prediction=1,
        )

        result = node.to_dict()

        # Check that index and metrics are excluded
        assert "index" not in result
        assert "metrics" not in result

        # Check included attributes
        assert result["id"] == 1
        assert result["depth"] == 0
        assert result["is_leaf"] is True
        assert result["split_feature"] == "feature1"
        assert result["operator"] == ">="
        assert result["split_value"] == 10
        assert result["prediction"] == 1
        assert result["parent"] is None
        assert result["children"] is None

    def test_node_to_dict_with_parent_and_children(self):
        """Test Node to_dict with parent and children relationships."""
        # Create parent node
        parent = Node(id=0)

        # Create child nodes
        child1 = Node(id=1, parent=parent)
        child2 = Node(id=2, parent=parent)

        # Set up parent-child relationships
        parent.children = [child1, child2]

        # Test parent node dict
        parent_dict = parent.to_dict()
        assert parent_dict["parent"] is None
        assert parent_dict["children"] == [1, 2]

        # Test child node dict
        child_dict = child1.to_dict()
        assert child_dict["parent"] == 0
        assert child_dict["children"] is None

    def test_node_to_dict_with_invalid_children(self):
        """Test Node to_dict with non-Node children (edge case)."""
        node = Node(id=1)
        node.children = ["not_a_node", None]  # Invalid children

        result = node.to_dict()
        assert result["children"] == []

    def test_node_to_json(self):
        """Test Node to_json conversion."""
        node = Node(id=1, depth=0, is_leaf=True, prediction=1)

        json_str = node.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["id"] == 1
        assert parsed["depth"] == 0
        assert parsed["is_leaf"] is True
        assert parsed["prediction"] == 1

    def test_node_with_metrics(self):
        """Test Node with metrics attribute."""
        metrics = {"precision": 0.8, "recall": 0.6, "f_score": 0.7}
        node = Node(id=1, metrics=metrics)

        # Metrics should be excluded from to_dict
        result = node.to_dict()
        assert "metrics" not in result

        # But metrics should still be accessible on the node
        assert node.metrics == metrics

    def test_node_with_index(self):
        """Test Node with pandas Index."""
        index = pd.Index([10, 20, 30])
        node = Node(id=1, index=index)

        # Index should be excluded from to_dict
        result = node.to_dict()
        assert "index" not in result

        # But index should still be accessible on the node
        assert node.index is index
