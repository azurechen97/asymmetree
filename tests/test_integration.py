import pytest
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree


class TestIntegration:
    """Integration tests for AsymmeTree workflow."""

    def test_full_workflow(self):
        """Test complete workflow: create tree, import data, predict."""
        # Create synthetic imbalanced dataset
        np.random.seed(42)
        n_samples = 50

        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.uniform(0, 100, n_samples),
                "category": np.random.choice(["A", "B"], n_samples),
            }
        )

        # Create highly imbalanced target (only 2 positive samples)
        y = pd.Series([0] * 48 + [1] * 2)

        # Initialize tree
        tree = AsymmeTree(
            max_depth=2,
            node_min_recall=0.01,  # Lower threshold for small dataset
            verbose=False,
        )

        # Import data
        tree.import_data(X, y, cat_features=["category"])

        # Basic checks
        assert tree.X is not None
        assert tree.y is not None
        assert len(tree.X) == n_samples
        assert tree.tree.id == 0
        assert tree.tree.is_leaf is True

        # Test prediction
        predictions = tree.predict(X)
        assert len(predictions) == n_samples
        assert all(pred in [0, 1] for pred in predictions)

        # Test metrics
        metrics = tree.metrics()
        assert "Precision" in metrics
        assert "Recall" in metrics
        assert "Positives" in metrics

        # Test serialization round trip
        tree_dict = tree.to_dict()
        json_str = tree.to_json()

        # Create new tree and deserialize
        new_tree = AsymmeTree()
        new_tree.from_dict(tree_dict)

        assert new_tree.tree is not None
        assert new_tree.tree.id == 0

        # Test predictions on new tree (should work even without data)
        # Note: predictions may differ since no data is loaded

    def test_tree_manipulation(self):
        """Test tree structure manipulation methods."""
        # Create simple dataset
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        y = pd.Series([0, 0, 1, 1, 1])

        tree = AsymmeTree(verbose=False)
        tree.import_data(X, y)

        # Test toggle prediction
        original_pred = tree.tree.prediction
        tree.toggle_prediction(0)
        assert tree.tree.prediction == (1 - original_pred)

        # Test relabel
        tree.tree.metrics = {"Precision": 0.5, "Recall": 0.8}
        tree.relabel(min_precision=0.4, min_recall=0.1)
        assert tree.tree.prediction == 1

        tree.relabel(min_precision=0.6, min_recall=0.1)
        assert tree.tree.prediction == 0

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        tree = AsymmeTree()

        # Test with no data
        with pytest.raises(AttributeError):
            tree.predict(pd.DataFrame({"x": [1, 2, 3]}))

        # Test invalid node operations
        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        tree.import_data(X, y)

        with pytest.raises(ValueError):
            tree.toggle_prediction(999)  # Invalid node ID

        with pytest.raises(ValueError):
            tree.clear_children(999)  # Invalid node ID
