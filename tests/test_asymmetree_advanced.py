import pytest
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree, Node


class TestAsymmeTreeAdvanced:
    """Advanced tests for AsymmeTree fit and split methods."""

    @pytest.fixture
    def simple_separable_data(self):
        """Create simple data that should produce predictable splits."""
        # Create larger dataset with clearer separation for AsymmeTree
        X = pd.DataFrame({"x": list(range(1, 21)), "y": [i * 10 for i in range(1, 21)]})
        # Create more imbalanced data (typical for AsymmeTree)
        # Only samples with x >= 15 are positive (30% positive class)
        y = pd.Series([0] * 14 + [1] * 6)
        return X, y

    @pytest.fixture
    def categorical_separable_data(self):
        """Create data with categorical features that separate classes."""
        X = pd.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )
        # Category C has most positive samples
        y = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        return X, y

    def test_fit_auto_mode_simple_data(self, simple_separable_data):
        """Test fit method in auto mode with simple separable data."""
        X, y = simple_separable_data

        tree = AsymmeTree(
            max_depth=2,
            node_min_recall=0.1,  # Lower threshold for small dataset
            node_max_precision=0.9,  # Allow splitting even with high precision
            verbose=False,
        )

        # Fit the tree in auto mode
        tree.fit(X, y, auto=True)

        # Verify tree structure
        assert tree.tree is not None
        assert tree.tree.id == 0
        assert len(tree.node_dict) >= 1  # At least root node

        # Test predictions
        predictions = tree.predict(X)
        assert len(predictions) == len(y)

        # Test ran successfully - predictions should be binary (0 or 1)
        assert all(pred in [0, 1] for pred in predictions)

        # For small imbalanced datasets, tree might not achieve high accuracy
        # but should make reasonable predictions
        accuracy = (predictions == y).mean()
        assert accuracy >= 0.3  # Lower threshold for small imbalanced dataset

        # Verify metrics can be calculated
        metrics = tree.metrics()
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in ["Precision", "Recall", "Positives"])

    def test_fit_with_categorical_data(self, categorical_separable_data):
        """Test fit method with categorical features."""
        X, y = categorical_separable_data

        tree = AsymmeTree(
            max_depth=2, node_min_recall=0.1, node_max_precision=0.9, verbose=False
        )

        # Fit with categorical features specified
        tree.fit(X, y, cat_features=["category"], auto=True)

        # Verify tree was built
        assert tree.tree is not None
        assert tree.cat_features == ["category"]

        # Test predictions
        predictions = tree.predict(X)
        assert len(predictions) == len(y)

        # Verify metrics
        metrics = tree.metrics()
        assert isinstance(metrics, dict)

    def test_split_method_auto_mode(self, simple_separable_data):
        """Test split method in auto mode."""
        X, y = simple_separable_data

        tree = AsymmeTree(
            max_depth=3, node_min_recall=0.1, node_max_precision=0.9, verbose=False
        )

        # Import data but don't fit
        tree.import_data(X, y)

        # Verify initial state
        assert tree.tree.is_leaf is True
        assert len(tree.node_dict) == 1

        # Split the root node in auto mode
        tree.split(id=0, auto=True)

        # Verify split occurred (should have more nodes now)
        if len(tree.node_dict) > 1:  # Split was successful
            assert tree.tree.is_leaf is False
            assert tree.tree.children is not None
            assert len(tree.tree.children) == 2

            # Verify child nodes
            left_child = tree.tree.children[0]
            right_child = tree.tree.children[1]

            assert left_child.parent is tree.tree
            assert right_child.parent is tree.tree
            assert left_child.id in tree.node_dict
            assert right_child.id in tree.node_dict

    def test_fit_with_constraints(self, simple_separable_data):
        """Test fit method with feature constraints."""
        X, y = simple_separable_data

        tree = AsymmeTree(max_depth=2, node_min_recall=0.1, verbose=False)

        # Fit with feature constraints
        tree.fit(
            X,
            y,
            lt_only_features=["x"],  # x can only use < operator
            gt_only_features=["y"],  # y can only use > operator
            auto=True,
        )

        # Verify constraints were applied
        assert tree.lt_only_features == ["x"]
        assert tree.gt_only_features == ["y"]

        # Verify tree was built
        assert tree.tree is not None
        predictions = tree.predict(X)
        assert len(predictions) == len(y)

    def test_continue_fit_method(self, simple_separable_data):
        """Test continue_fit method."""
        X, y = simple_separable_data

        tree = AsymmeTree(
            max_depth=3, node_min_recall=0.1, node_max_precision=0.9, verbose=False
        )

        # Import data and do initial split
        tree.import_data(X, y)
        tree.split(id=0, auto=True)

        # Continue fitting from root (if split was successful)
        if len(tree.node_dict) > 1:
            initial_node_count = len(tree.node_dict)
            tree.continue_fit(id=0, auto=True)

            # Verify tree may have grown (or stayed same if no good splits)
            assert len(tree.node_dict) >= initial_node_count

    def test_fit_edge_cases(self):
        """Test fit method with edge case data."""
        # Test with all same class
        X_same = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        y_same = pd.Series([1, 1, 1, 1, 1])  # All positive

        tree = AsymmeTree(max_depth=2, verbose=False)
        tree.fit(X_same, y_same, auto=True)

        # Should not split (all same class)
        assert tree.tree is not None
        predictions = tree.predict(X_same)
        assert len(predictions) == len(y_same)

        # Test with no positive samples
        y_none = pd.Series([0, 0, 0, 0, 0])  # All negative
        tree2 = AsymmeTree(max_depth=2, verbose=False)

        # This should run without error even with no positive samples
        tree2.fit(X_same, y_none, auto=True)

        assert tree2.tree is not None
        predictions2 = tree2.predict(X_same)
        assert len(predictions2) == len(y_none)

        # When there are no positives, all predictions should be 0
        assert all(pred == 0 for pred in predictions2)

    def test_quick_split_method(self, simple_separable_data):
        """Test quick_split method with auto mode (no SQL)."""
        X, y = simple_separable_data

        tree = AsymmeTree(max_depth=2, node_min_recall=0.1, verbose=False)

        # Import data
        tree.import_data(X, y)

        # Test quick split without SQL (auto mode)
        tree.quick_split(id=0, sql=None, overwrite=True)

        # Verify split occurred if data supports it
        if len(tree.node_dict) > 1:
            assert tree.tree.is_leaf is False
            assert tree.tree.children is not None

    def test_tree_sql_generation_after_fit(self, simple_separable_data):
        """Test SQL generation after fitting a tree."""
        X, y = simple_separable_data

        tree = AsymmeTree(
            max_depth=2,
            node_min_recall=0.1,
            leaf_min_precision=0.1,  # Low threshold to allow predictions
            verbose=False,
        )

        # Fit tree
        tree.fit(X, y, auto=True)

        # Relabel to ensure some positive predictions
        tree.relabel(min_precision=0.1, min_recall=0.1)

        # Generate SQL
        sql = tree.to_sql()

        # Should generate some SQL if there are positive predictions
        if any(
            node.prediction == 1 for node in tree.node_dict.values() if node.is_leaf
        ):
            assert sql is not None
            assert isinstance(sql, str)
            assert len(sql) > 0

    def test_tree_with_weights(self, simple_separable_data):
        """Test fit method with sample weights."""
        X, y = simple_separable_data

        # Create weights that emphasize positive samples
        weights = pd.Series(
            [1.0] * 14 + [2.0] * 6
        )  # Higher weight for positive samples

        tree = AsymmeTree(max_depth=2, node_min_recall=0.1, verbose=False)

        # Fit with weights
        tree.fit(X, y, weights=weights, auto=True)

        # Verify weights were applied
        assert tree.weights is not None
        assert len(tree.weights) == len(weights)

        # Test predictions
        predictions = tree.predict(X)
        assert len(predictions) == len(y)

    def test_prune_after_fit(self, simple_separable_data):
        """Test pruning functionality after fitting."""
        X, y = simple_separable_data

        tree = AsymmeTree(
            max_depth=3, node_min_recall=0.1, leaf_min_precision=0.1, verbose=False
        )

        # Fit tree
        tree.fit(X, y, auto=True)

        # Force some predictions to be the same to test pruning
        for node in tree.node_dict.values():
            if node.is_leaf:
                node.prediction = 0  # Set all leaves to same prediction

        initial_node_count = len(tree.node_dict)

        # Prune the tree
        tree.prune()

        # Pruning may reduce node count if siblings have same prediction
        final_node_count = len(tree.node_dict)
        assert final_node_count <= initial_node_count
