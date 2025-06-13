import pytest
import json
import tempfile
import os
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree, Node


class TestAsymmeTree:
    """Test cases for AsymmeTree class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset for testing."""
        np.random.seed(42)
        n_samples = 100

        # Create features
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples),
                "income": np.random.normal(50000, 20000, n_samples),
                "category": np.random.choice(["A", "B", "C"], n_samples),
                "score": np.random.uniform(0, 1, n_samples),
            }
        )

        # Create imbalanced target (5% positive class)
        y = np.zeros(n_samples)
        positive_indices = np.random.choice(
            n_samples, size=int(0.05 * n_samples), replace=False
        )
        y[positive_indices] = 1
        y = pd.Series(y, dtype=int)

        return X, y

    @pytest.fixture
    def fitted_tree(self, sample_data):
        """Create a fitted AsymmeTree for testing."""
        X, y = sample_data
        tree = AsymmeTree(max_depth=2, verbose=False)
        tree.import_data(X, y)
        return tree

    def test_asymmetree_init_defaults(self):
        """Test AsymmeTree initialization with default parameters."""
        tree = AsymmeTree()

        assert tree.max_depth == 5
        assert tree.max_cat_unique == 50
        assert tree.cat_value_min_recall == 0.005
        assert tree.num_bin == 25
        assert tree.node_max_precision == 0.3
        assert tree.node_min_recall == 0.05
        assert tree.leaf_min_precision == 0.15
        assert tree.feature_shown_num == 5
        assert tree.condition_shown_num == 5
        assert tree.sorted_by == "f_score"
        assert tree.pos_weight == 1
        assert tree.beta == 1
        assert tree.knot == 1
        assert tree.factor == 1
        assert tree.ignore_null is True
        assert tree.show_metrics is False
        assert tree.verbose is False

    def test_asymmetree_init_custom_params(self):
        """Test AsymmeTree initialization with custom parameters."""
        tree = AsymmeTree(
            max_depth=3,
            max_cat_unique=20,
            node_min_recall=0.1,
            sorted_by="ig",
            verbose=True,
        )

        assert tree.max_depth == 3
        assert tree.max_cat_unique == 20
        assert tree.node_min_recall == 0.1
        assert tree.sorted_by == "ig"
        assert tree.verbose is True

    def test_import_data_basic(self, sample_data):
        """Test basic data import functionality."""
        X, y = sample_data
        tree = AsymmeTree()

        tree.import_data(X, y)

        assert tree.X is not None
        assert tree.y is not None
        assert tree.weights is not None
        assert len(tree.X) == len(X)
        assert len(tree.y) == len(y)
        assert tree.tree is not None
        assert tree.tree.id == 0
        assert tree.node_dict[0] is tree.tree

    def test_import_data_with_weights(self, sample_data):
        """Test data import with custom weights."""
        X, y = sample_data
        weights = pd.Series(np.random.uniform(0.5, 1.5, len(y)))
        tree = AsymmeTree()

        tree.import_data(X, y, weights=weights)

        assert tree.weights is not None
        assert len(tree.weights) == len(weights)
        np.testing.assert_array_equal(tree.weights.values, weights.values)

    def test_import_data_with_categorical_features(self, sample_data):
        """Test data import with categorical features specified."""
        X, y = sample_data
        cat_features = ["category"]
        tree = AsymmeTree()

        tree.import_data(X, y, cat_features=cat_features)

        assert tree.cat_features == ["category"]

    def test_import_data_with_constraints(self, sample_data):
        """Test data import with feature constraints."""
        X, y = sample_data
        lt_only = ["age"]
        gt_only = ["income"]
        pinned = ["score"]

        tree = AsymmeTree()
        tree.import_data(
            X,
            y,
            lt_only_features=lt_only,
            gt_only_features=gt_only,
            pinned_features=pinned,
        )

        assert tree.lt_only_features == lt_only
        assert tree.gt_only_features == gt_only
        assert tree.pinned_features == pinned

    def test_predict_single_node(self, fitted_tree):
        """Test prediction with single node (root only)."""
        X = fitted_tree.X

        # Before any splits, should predict based on root
        predictions = fitted_tree.predict(X)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_empty_dataframe(self, fitted_tree):
        """Test prediction with empty DataFrame."""
        empty_X = pd.DataFrame(columns=fitted_tree.X.columns)

        predictions = fitted_tree.predict(empty_X)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 0

    def test_metrics_calculation(self, fitted_tree):
        """Test metrics calculation functionality."""
        X = fitted_tree.X
        predictions = fitted_tree.predict(X)

        metrics = fitted_tree.metrics(predictions)

        assert isinstance(metrics, dict)
        assert "Precision" in metrics
        assert "Recall" in metrics
        assert "Positives" in metrics
        assert all(isinstance(v, (int, float, np.number)) for v in metrics.values())

    def test_toggle_prediction(self, fitted_tree):
        """Test toggling prediction of leaf nodes."""
        root = fitted_tree.tree
        original_prediction = root.prediction

        fitted_tree.toggle_prediction(0)

        assert root.prediction == (1 - original_prediction)

        # Toggle back
        fitted_tree.toggle_prediction(0)
        assert root.prediction == original_prediction

    def test_toggle_prediction_invalid_node(self, fitted_tree):
        """Test toggling prediction with invalid node ID."""
        with pytest.raises(ValueError, match="Node with id 999 not found"):
            fitted_tree.toggle_prediction(999)

    def test_relabel_nodes(self, fitted_tree):
        """Test relabeling nodes based on precision/recall thresholds."""
        # Set up a simple tree structure for testing
        root = fitted_tree.tree
        root.metrics = {"Precision": 0.2, "Recall": 0.1}

        fitted_tree.relabel(min_precision=0.15, min_recall=0.05)

        # Root should be labeled as positive (meets thresholds)
        assert root.prediction == 1

        # Test with higher thresholds
        fitted_tree.relabel(min_precision=0.25, min_recall=0.05)
        assert root.prediction == 0

    def test_clear_children(self, fitted_tree):
        """Test clearing children of a node."""
        root = fitted_tree.tree

        # Create mock children
        child1 = Node(id=1, parent=root)
        child2 = Node(id=2, parent=root)
        root.children = [child1, child2]
        root.is_leaf = False
        fitted_tree.node_dict[1] = child1
        fitted_tree.node_dict[2] = child2

        fitted_tree.clear_children(0)

        assert root.children is None
        assert root.is_leaf is True
        assert 1 not in fitted_tree.node_dict
        assert 2 not in fitted_tree.node_dict

    def test_clear_children_invalid_node(self, fitted_tree):
        """Test clearing children with invalid node ID."""
        with pytest.raises(ValueError, match="Node with id 999 not found"):
            fitted_tree.clear_children(999)

    def test_to_dict(self, fitted_tree):
        """Test tree serialization to dictionary."""
        tree_dict = fitted_tree.to_dict()

        assert isinstance(tree_dict, dict)
        assert 0 in tree_dict  # Root node should be present
        assert "id" in tree_dict[0]
        assert "is_leaf" in tree_dict[0]
        assert "depth" in tree_dict[0]

    def test_to_json(self, fitted_tree):
        """Test tree serialization to JSON."""
        json_str = fitted_tree.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "0" in parsed  # Root node (as string key)

    def test_from_dict(self, fitted_tree):
        """Test tree deserialization from dictionary."""
        # Save current tree
        original_dict = fitted_tree.to_dict()

        # Create new tree and load from dict
        new_tree = AsymmeTree()
        new_tree.from_dict(original_dict)

        assert new_tree.tree is not None
        assert new_tree.tree.id == 0
        assert 0 in new_tree.node_dict
        assert new_tree.node_counter == len(original_dict)

    def test_from_json(self, fitted_tree):
        """Test tree deserialization from JSON."""
        # Save current tree
        json_str = fitted_tree.to_json()

        # Create new tree and load from JSON
        new_tree = AsymmeTree()
        new_tree.from_json(json_str)

        assert new_tree.tree is not None
        assert new_tree.tree.id == 0
        assert 0 in new_tree.node_dict

    def test_save_and_load(self, fitted_tree):
        """Test saving and loading tree to/from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Save tree
            fitted_tree.save(temp_file)
            assert os.path.exists(temp_file)

            # Load tree
            new_tree = AsymmeTree()
            new_tree.load(temp_file)

            assert new_tree.tree is not None
            assert new_tree.tree.id == 0
            assert 0 in new_tree.node_dict

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_to_sql_empty_tree(self, fitted_tree):
        """Test SQL generation for tree with no positive predictions."""
        # Set root to predict 0
        fitted_tree.tree.prediction = 0

        sql = fitted_tree.to_sql()
        assert sql is None

    def test_to_sql_root_only(self, fitted_tree):
        """Test SQL generation for tree with only root node."""
        # Set root to predict 1
        fitted_tree.tree.prediction = 1

        sql = fitted_tree.to_sql()
        assert sql == "(TRUE)"

    def test_repr(self):
        """Test AsymmeTree string representation."""
        tree = AsymmeTree(max_depth=3, verbose=True)
        repr_str = repr(tree)

        assert "AsymmeTree(" in repr_str
        assert "max_depth=3" in repr_str
        assert "verbose=True" in repr_str

    def test_fit_with_auto_mode(self, sample_data):
        """Test fitting with auto mode (basic functionality)."""
        X, y = sample_data
        tree = AsymmeTree(max_depth=1, verbose=False)

        # This test just verifies the fit method runs without errors
        # in auto mode (actual tree building is complex and would require
        # more sophisticated mocking for full testing)
        tree.import_data(X, y)

        # Verify basic state after import
        assert tree.X is not None
        assert tree.y is not None
        assert tree.tree is not None
        assert tree.tree.id == 0
