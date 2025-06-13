import ast
import re
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype

from asymmetree.metrics import *
from asymmetree.utils import *


class Node:
    """A node in the AsymmeTree decision tree.

    Represents a single node in the decision tree structure, containing split conditions,
    predictions, metrics, and references to parent/child nodes.

    Attributes:
        index: Index of data samples belonging to this node.
        parent: Reference to parent node.
        children: List of child nodes (None for leaf nodes).
        split_feature: Feature name used for splitting at this node.
        operator: Comparison operator used for the split condition.
        split_value: Value used in the split condition.
        is_leaf: Whether this node is a leaf node.
        prediction: Prediction value (0 or 1) for leaf nodes.
        metrics: Dictionary containing node performance metrics.
        depth: Depth of this node in the tree (root = 0).
        id: Unique identifier for this node.
    """

    def __init__(
        self,
        index=None,
        parent=None,
        children=None,
        split_feature=None,
        operator=None,
        split_value=None,
        is_leaf=True,
        prediction=None,
        metrics=None,
        depth=None,
        id=None,
    ):
        """Initialize a new Node.

        Args:
            index: Index of data samples belonging to this node.
            parent: Reference to parent node.
            children: List of child nodes.
            split_feature: Feature name used for splitting.
            operator: Comparison operator for split condition.
            split_value: Value used in split condition.
            is_leaf: Whether this is a leaf node. Defaults to True.
            prediction: Prediction value for leaf nodes.
            metrics: Performance metrics dictionary.
            depth: Node depth in tree.
            id: Unique node identifier.
        """
        self.index = index
        self.parent = parent
        self.children = children
        self.split_feature = split_feature
        self.operator = operator
        self.split_value = split_value
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.metrics = metrics
        self.depth = depth
        self.id = id

    def __repr__(self):
        """Return string representation of the node."""
        return f"Node(id={self.id}, depth={self.depth}, is_leaf={self.is_leaf}, split_feature={self.split_feature}, operator={self.operator}, split_value={self.split_value}, prediction={self.prediction}, metrics={self.metrics})"

    def __str__(self):
        """Return string representation of the node."""
        return self.__repr__()

    def to_dict(self):
        """Convert node to dictionary representation.

        Returns:
            dict: Dictionary containing node attributes with parent/children as IDs.
        """
        d = self.__dict__.copy()
        del d["index"]
        del d["metrics"]

        d["parent"] = self.parent.id if isinstance(self.parent, Node) else None
        d["children"] = (
            [child.id for child in self.children if isinstance(child, Node)]
            if isinstance(self.children, list)
            else None
        )

        return d

    def to_json(self):
        """Convert node to JSON string.

        Returns:
            str: JSON representation of the node.
        """
        return json.dumps(self.to_dict())


class AsymmeTree:
    """Interactive decision tree classifier optimized for highly imbalanced datasets.

    AsymmeTree implements a novel training methodology for extreme class imbalance scenarios
    like fraud detection and anomaly detection. Unlike traditional trees that optimize for
    node purity, AsymmeTree focuses on maximizing precision while capturing sufficient recall.

    Key Features:
        - **Imbalanced-Optimized Splitting**: Left child = "positive node" (higher positive ratio),
          right child = "neutral node" (lower positive ratio)
        - **F-Score Based Optimization**: Optimizes splits using the F-score of the positive
          node, balancing precision and recall for imbalanced settings
        - **Traditional Methods Supported**: Also supports purity-based approaches using
          information gain (ig), information gain ratio (igr), or information value (iv)
        - **Dual Mode Operation**: Interactive, automatic, or hybrid modes allowing domain
          expertise integration throughout the tree building process
        - **Flexible Constraints**: Handles categorical/numerical features with custom operators

    Ideal for binary classification with significant class imbalance (< 5% positive class):
    fraud detection, medical diagnosis, quality control, marketing conversion optimization.

    Maintains interpretability while achieving superior performance on imbalanced datasets
    compared to traditional entropy or Gini-based approaches.
    """

    def __init__(
        self,
        max_depth=5,
        max_cat_unique=50,
        cat_value_min_recall=0.005,
        num_bin=25,
        node_max_precision=0.3,
        node_min_recall=0.05,
        leaf_min_precision=0.15,
        feature_shown_num=5,
        condition_shown_num=5,
        sorted_by="f_score",
        pos_weight=1,
        beta=1,
        knot=1,
        factor=1,
        ignore_null=True,
        show_metrics=False,
        verbose=False,
    ):
        """Initialize AsymmeTree with configuration parameters.

        Args:
            max_depth (int): Maximum depth of the tree. Defaults to 5.
            max_cat_unique (int): Maximum unique values for categorical features. Defaults to 50.
            cat_value_min_recall (float): Minimum recall threshold for categorical values. Defaults to 0.005.
            num_bin (int): Number of bins for numerical feature discretization. Defaults to 25.
            node_max_precision (float): Maximum precision threshold for node splitting. Defaults to 0.3.
            node_min_recall (float): Minimum recall threshold for nodes. Defaults to 0.05.
            leaf_min_precision (float): Minimum precision threshold for leaf nodes. Defaults to 0.15.
            feature_shown_num (int): Number of features to show in interactive mode. Defaults to 5.
            condition_shown_num (int): Number of conditions to show in interactive mode. Defaults to 5.
            sorted_by (str): Metric to sort splits by ('f_score', 'ig', 'igr', 'iv'). Defaults to 'f_score'.
            pos_weight (float): Weight for positive class in calculations. Defaults to 1.
            beta (float): Beta parameter for F-beta score. Defaults to 1 (F1-score).
            knot (float): Threshold for precision scaling. Defaults to 1.
            factor (float): Scaling factor for precision above knot. Defaults to 1.
            ignore_null (bool): Whether to ignore null values. Defaults to True.
            show_metrics (bool): Whether to show metrics in tree display. Defaults to False.
            verbose (bool): Whether to print verbose output. Defaults to False.
        """
        # Parameters initialization
        self.max_depth = max_depth
        self.max_cat_unique = max_cat_unique
        self.cat_value_min_recall = cat_value_min_recall
        self.num_bin = num_bin
        self.node_max_precision = node_max_precision
        self.node_min_recall = node_min_recall
        self.leaf_min_precision = leaf_min_precision

        # Display parameters initialization
        self.feature_shown_num = feature_shown_num
        self.condition_shown_num = condition_shown_num
        self.sorted_by = sorted_by
        self.pos_weight = pos_weight
        self.beta = beta
        self.knot = knot
        self.factor = factor
        self.ignore_null = ignore_null
        self.show_metrics = show_metrics
        self.verbose = verbose

        # Data initialization
        self.X = None
        self.y = None
        self.weights = None
        self.cat_features = None
        self.lt_only_features = None
        self.gt_only_features = None
        self.pinned_features = None
        self.extra_metrics = None
        self.extra_metrics_data = None

        self.total_pos = None
        self.root_precision = None
        self.root_recall = None

        # Tree initialization
        self.tree = None
        self.node_dict = {}
        self.node_counter = 0

    def import_data(
        self,
        X,
        y,
        weights=None,
        cat_features=None,
        lt_only_features=None,
        gt_only_features=None,
        pinned_features=None,
        extra_metrics=None,
        extra_metrics_data=None,
        total_pos=None,
    ):
        """Import training data and configure feature constraints.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.array): Target variable (binary).
            weights (pd.Series or np.array, optional): Sample weights.
            cat_features (list, optional): List of categorical feature names.
            lt_only_features (list, optional): Features that can only use '<' operators.
            gt_only_features (list, optional): Features that can only use '>' operators.
            pinned_features (list, optional): Features to prioritize in splitting.
            extra_metrics (dict, optional): Additional metrics to calculate.
            extra_metrics_data (pd.DataFrame, optional): Data for extra metrics.
            total_pos (float, optional): Total positive samples for recall calculation.
        """
        self.X = X
        self.y = y
        self.weights = weights
        self.cat_features = cat_features
        self.lt_only_features = lt_only_features
        self.gt_only_features = gt_only_features
        self.pinned_features = pinned_features
        self.extra_metrics = extra_metrics
        self.extra_metrics_data = extra_metrics_data
        self.total_pos = total_pos

        # Data preprocessing
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess imported data and initialize tree structure.

        Handles data type conversions, feature type detection, weight initialization,
        and creates the root node of the tree.
        """
        if isinstance(self.y, np.ndarray):
            self.y = pd.Series(self.y)
        if isinstance(self.y, pd.Series) and not self.y.index.equals(self.X.index):
            self.y.index = self.X.index

        X_columns = self.X.columns.tolist()
        str_columns = [col for col in X_columns if is_string_dtype(self.X[col])]
        if self.cat_features is None:
            self.cat_features = []
        self.cat_features = [
            col for col in X_columns if col in str_columns or col in self.cat_features
        ]
        if self.lt_only_features is None:
            self.lt_only_features = []
        if self.gt_only_features is None:
            self.gt_only_features = []
        if self.pinned_features is None:
            self.pinned_features = []

        if self.weights is None:
            self.weights = np.ones(len(self.y), dtype=np.float64)
        if isinstance(self.weights, np.ndarray):
            self.weights = pd.Series(self.weights, index=self.X.index)
        if isinstance(self.weights, pd.Series) and not self.weights.index.equals(
            self.X.index
        ):
            self.weights.index = self.X.index

        if isinstance(
            self.extra_metrics_data, pd.DataFrame
        ) and not self.extra_metrics_data.index.equals(self.X.index):
            self.extra_metrics_data.index = self.X.index

        root_pos = (self.y * self.weights).sum()
        root_count = self.weights.sum()
        self.total_pos = root_pos if self.total_pos is None else self.total_pos
        self.root_precision = root_pos / root_count if root_count > 0 else 0
        self.root_recall = root_pos / self.total_pos if self.total_pos > 0 else 0

        # Tree initialization
        if not self.tree:
            self.tree = Node(
                index=self.X.index,
                depth=0,
                is_leaf=True,
                prediction=1,
                id=0,
            )
            # Set metrics for the root node
            self.tree.metrics = {
                "Precision": self.root_precision,
                "Recall": self.root_recall,
                "Positives": root_pos,
            }
            self.node_dict[0] = self.tree
            self.node_counter = 1
        else:
            self.reset_tree_data(node=self.tree)

    def reset_tree_data(self, node: Node = None, mask=None):
        """Reset node data and metrics after data changes.

        Recursively updates node indices and metrics for all nodes in the tree
        based on the current data and provided mask.

        Args:
            node (Node, optional): Node to reset. Defaults to root node.
            mask (pd.Series, optional): Boolean mask for data filtering.
        """
        if not node:
            node = self.tree
        if node is self.tree or mask is None:
            mask = pd.Series(True, index=self.X.index)
            node.index = self.X.index
            root_pos = (self.y * self.weights).sum()
            node.metrics = {
                "Precision": self.root_precision,
                "Recall": self.root_recall,
                "Positives": root_pos,
            }
        else:
            X_sub = self.X.loc[mask]
            y_sub = self.y.loc[mask]
            weights_sub = self.weights.loc[mask]
            node.index = X_sub.index
            node.metrics = {
                "Precision": get_precision(
                    np.ones(len(y_sub), dtype="float64"), y_sub, weights_sub
                ),
                "Recall": get_recall(
                    np.ones(len(y_sub), dtype="float64"),
                    y_sub,
                    weights_sub,
                    self.total_pos,
                ),
                "Positives": (y_sub * weights_sub).sum(),
            }

        if not node.is_leaf and isinstance(node.children, list):
            left_child = node.children[0]
            right_child = node.children[1]
            left_mask = self._split_to_mask(
                self.X,
                left_child.split_feature,
                left_child.operator,
                left_child.split_value,
            )
            self.reset_tree_data(node=left_child, mask=left_mask & mask)
            self.reset_tree_data(node=right_child, mask=(~left_mask) & mask)

    def split(self, id: int = None, auto=False):
        """Interactively split a node in the tree.

        Provides interactive interface for selecting features and split conditions,
        or automatically selects the best split based on the configured metric.

        Args:
            id (int, optional): ID of node to split. Defaults to root (0).
            auto (bool): Whether to automatically select best split. Defaults to False.

        Raises:
            ValueError: If node with given ID is not found.
        """
        if id is None:
            id = 0
        if id not in self.node_dict:
            raise ValueError(f"Node with id {id} not found")

        node = self.node_dict[id]
        self.print(self.tree, current_node=node, show_metrics=self.show_metrics)

        if not node.is_leaf and isinstance(node.children, list):
            warning_msg = input(
                f"WARNING: This operation will OVERWRITE the children of node {id}. Continue? (y/N): "
            )
            if warning_msg.lower() != "y":
                return
            self.clear_children(id)

        if id == 0:
            X = self.X
            y = self.y
            weights = self.weights
            extra_metrics_data = self.extra_metrics_data
        else:
            X = self.X.loc[node.index]
            y = self.y.loc[node.index]
            weights = self.weights.loc[node.index]
            if self.extra_metrics_data is not None and isinstance(
                self.extra_metrics_data, pd.DataFrame
            ):
                extra_metrics_data = self.extra_metrics_data.loc[node.index]
            else:
                extra_metrics_data = None

        lt_only_features = self.lt_only_features
        gt_only_features = self.gt_only_features
        cat_features = self.cat_features
        extra_metrics = self.extra_metrics
        pinned_features = self.pinned_features

        try:
            _, _ = self._split_node(
                X,
                y,
                weights,
                cat_features,
                lt_only_features,
                gt_only_features,
                pinned_features,
                extra_metrics,
                extra_metrics_data,
                auto,
                node,
            )
        except ExitSplit as e:
            print(str(e))

        print("\nCurrent tree:")
        self.print(self.tree, show_metrics=self.show_metrics)
        print("\nOverall performance:")
        y_pred = self.predict(self.X)
        self.performance(
            pred=y_pred,
            true=self.y,
            weights=self.weights,
            extra_metrics=self.extra_metrics,
            extra_metrics_data=self.extra_metrics_data,
            total_pos=self.total_pos,
        )

    def quick_split(self, id: int = None, sql: str = None, overwrite=False):
        """Quickly split a node using SQL-style condition or auto-selection.

        Provides a fast way to split nodes either automatically or with a
        pre-specified SQL-style condition.

        Args:
            id (int, optional): ID of node to split. Defaults to root (0).
            sql (str, optional): SQL-style condition for splitting (e.g., '>=100').
            overwrite (bool): Whether to overwrite existing children. Defaults to False.

        Raises:
            ValueError: If node ID not found or can't overwrite without permission.
        """
        if id is None:
            id = 0
        if id not in self.node_dict:
            raise ValueError(f"Node with id {id} not found")

        node = self.node_dict[id]
        self.print(self.tree, current_node=node, show_metrics=self.show_metrics)

        if not node.is_leaf and isinstance(node.children, list):
            if overwrite:
                print(
                    f"WARNING: This operation will OVERWRITE the children of node {id}."
                )
            else:
                raise ValueError(
                    f"Can't overwrite the children of node {id} as argument overwrite is False."
                )
            self.clear_children(id)

        if id == 0:
            X = self.X
            y = self.y
            weights = self.weights
            extra_metrics_data = self.extra_metrics_data
        else:
            X = self.X.loc[node.index]
            y = self.y.loc[node.index]
            weights = self.weights.loc[node.index]
            if self.extra_metrics_data is not None and isinstance(
                self.extra_metrics_data, pd.DataFrame
            ):
                extra_metrics_data = self.extra_metrics_data.loc[node.index]
            else:
                extra_metrics_data = None

        lt_only_features = self.lt_only_features
        gt_only_features = self.gt_only_features
        cat_features = self.cat_features
        extra_metrics = self.extra_metrics
        pinned_features = self.pinned_features

        try:
            if sql is None:
                _, _ = self._split_node(
                    X,
                    y,
                    weights,
                    cat_features,
                    lt_only_features,
                    gt_only_features,
                    pinned_features,
                    extra_metrics,
                    extra_metrics_data,
                    True,
                    node,
                )
            else:
                feature, operator, value = self._parse_partial_sql(sql)
                _, _ = self._quick_split_node(
                    feature,
                    operator,
                    value,
                    X,
                    y,
                    weights,
                    extra_metrics,
                    extra_metrics_data,
                    node,
                )
        except ExitSplit as e:
            print(str(e))

        print("\nCurrent tree:")
        self.print(self.tree, show_metrics=self.show_metrics)
        print("\nOverall performance:")
        y_pred = self.predict(self.X)
        self.performance(
            pred=y_pred,
            true=self.y,
            weights=self.weights,
            extra_metrics=self.extra_metrics,
            extra_metrics_data=self.extra_metrics_data,
            total_pos=self.total_pos,
        )

    def fit(
        self,
        X,
        y,
        weights=None,
        cat_features=None,
        lt_only_features=None,
        gt_only_features=None,
        pinned_features=None,
        extra_metrics=None,
        extra_metrics_data=None,
        total_pos=None,
        auto=False,
    ):
        """Fit the AsymmeTree model to training data.

        Builds the decision tree automatically or interactively based on the auto parameter.
        Includes automatic pruning when auto=True.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.array): Target variable (binary).
            weights (pd.Series or np.array, optional): Sample weights.
            cat_features (list, optional): List of categorical feature names.
            lt_only_features (list, optional): Features that can only use '<' operators.
            gt_only_features (list, optional): Features that can only use '>' operators.
            pinned_features (list, optional): Features to prioritize in splitting.
            extra_metrics (dict, optional): Additional metrics to calculate.
            extra_metrics_data (pd.DataFrame, optional): Data for extra metrics.
            total_pos (float, optional): Total positive samples for recall calculation.
            auto (bool): Whether to build tree automatically. Defaults to False.

        Raises:
            ValueError: If tree already has nodes or no data provided.
        """
        if len(self.node_dict) > 1:
            raise ValueError(
                "Tree already has nodes. Initialize a new tree to use fit."
            )
        if X is not None:
            self.import_data(
                X,
                y,
                weights,
                cat_features,
                lt_only_features,
                gt_only_features,
                extra_metrics,
                extra_metrics_data,
                total_pos,
            )
        elif self.X is None:
            raise ValueError(
                "No data imported. Please input data or use import_data method."
            )

        X = self.X
        y = self.y
        weights = self.weights
        cat_features = self.cat_features
        lt_only_features = self.lt_only_features
        gt_only_features = self.gt_only_features
        extra_metrics = self.extra_metrics
        extra_metrics_data = self.extra_metrics_data
        total_pos = self.total_pos

        print(
            f"Root Precision: {self.root_precision:.2%}, Total Pos: {self.total_pos:.2f}"
        )
        extra_metrics_str = []
        if extra_metrics is not None and isinstance(extra_metrics_data, pd.DataFrame):
            for metric in extra_metrics:
                extra_metrics_str.append(
                    f"{metric}: {extra_metrics[metric](extra_metrics_data):.4f}".rstrip(
                        "0"
                    ).rstrip(".")
                )
                print(", ".join(extra_metrics_str))

        self._build_tree(
            X,
            y,
            weights,
            cat_features,
            lt_only_features,
            gt_only_features,
            pinned_features,
            extra_metrics,
            extra_metrics_data,
            auto,
            self.tree,
        )

        # Prune the tree
        if auto:
            print("\nPrune the tree...")
            self.relabel()
            self.prune()

        print("\nTree built successfully. Current tree:")
        self.print(self.tree, show_metrics=self.show_metrics)
        print("\nOverall performance:")
        y_pred = self.predict(X)
        self.performance(
            pred=y_pred,
            true=y,
            weights=weights,
            extra_metrics=extra_metrics,
            extra_metrics_data=extra_metrics_data,
            total_pos=total_pos,
        )

    def continue_fit(self, id, auto=False):
        """Continue building the tree from a specific node.

        Resumes tree construction from the specified node, useful for partial
        tree building or expanding specific branches.

        Args:
            id (int): ID of node to continue building from.
            auto (bool): Whether to build automatically. Defaults to False.

        Raises:
            ValueError: If node with given ID is not found.
        """
        if id not in self.node_dict:
            raise ValueError(f"Node with id {id} not found")
        node = self.node_dict[id]
        X = self.X.loc[node.index]
        y = self.y.loc[node.index]
        weights = self.weights.loc[node.index]
        if self.extra_metrics_data is not None and isinstance(
            self.extra_metrics_data, pd.DataFrame
        ):
            extra_metrics_data = self.extra_metrics_data.loc[node.index]
        else:
            extra_metrics_data = None
        if not node.is_leaf and isinstance(node.children, list):
            self.print(self.tree, current_node=node, show_metrics=self.show_metrics)
            warning_msg = input(
                f"WARNING: This operation will OVERWRITE the children of node {id}. Continue? (y/N): "
            )
            if warning_msg.lower() != "y":
                return
            self.clear_children(id)

        self._build_tree(
            X,
            y,
            weights,
            self.cat_features,
            self.lt_only_features,
            self.gt_only_features,
            self.pinned_features,
            self.extra_metrics,
            extra_metrics_data,
            auto,
            node,
        )

        # Prune the tree
        if auto:
            print("\nPrune the tree...")
            self.relabel()
            self.prune()

        print("\nTree built successfully. Current tree:")
        self.print(self.tree, show_metrics=self.show_metrics)
        print("\nOverall performance:")
        y_pred = self.predict(X)
        self.performance(
            pred=y_pred,
            true=y,
            weights=weights,
            extra_metrics=self.extra_metrics,
            extra_metrics_data=extra_metrics_data,
            total_pos=self.total_pos,
        )

    def toggle_prediction(self, id: int = None):
        """Toggle the prediction of a leaf node between 0 and 1.

        Args:
            id (int): ID of leaf node to toggle.

        Raises:
            ValueError: If node not found or node is not a leaf.
        """
        if id not in self.node_dict:
            raise ValueError(f"Node with id {id} not found")
        node = self.node_dict[id]
        if not node.is_leaf:
            raise ValueError(f"Node with id {id} is not a leaf")
        node.prediction = 1 - node.prediction

    def relabel(self, min_precision=None, min_recall=None):
        """Relabel leaf nodes based on precision and recall thresholds.

        Updates prediction labels for all leaf nodes based on whether they
        meet the minimum precision and recall requirements.

        Args:
            min_precision (float, optional): Minimum precision threshold.
                Defaults to leaf_min_precision.
            min_recall (float, optional): Minimum recall threshold.
                Defaults to node_min_recall.
        """
        if min_precision is None:
            min_precision = self.leaf_min_precision
        if min_recall is None:
            min_recall = self.node_min_recall
        for id in self.node_dict:
            node = self.node_dict[id]
            precision = node.metrics["Precision"]
            recall = node.metrics["Recall"]
            if node.is_leaf:
                node.prediction = (
                    1 if precision >= min_precision and recall >= min_recall else 0
                )

    def prune(self, node: Node = None):
        """Prune the tree by merging nodes with identical predictions.

        Recursively removes unnecessary splits where both children have
        the same prediction value.

        Args:
            node (Node, optional): Node to start pruning from. Defaults to root.
        """
        if node is None:
            node = self.tree
        if (
            not node.is_leaf
            and isinstance(node.children, list)
            and len(node.children) > 1
        ):
            left_child = node.children[0]
            right_child = node.children[1]
            self.prune(left_child)
            self.prune(right_child)
            if (
                left_child.is_leaf
                and right_child.is_leaf
                and left_child.prediction == right_child.prediction
            ):
                node.prediction = left_child.prediction
                node.is_leaf = True
                node.children = None
                _ = self.node_dict.pop(id)

    def clear_children(self, id: int = None):
        """Recursively remove all children of a node.

        Removes all descendant nodes and converts the specified node back to a leaf.

        Args:
            id (int): ID of node whose children should be cleared.

        Raises:
            ValueError: If node with given ID is not found.
        """
        if id not in self.node_dict:
            raise ValueError(f"Node with id {id} not found")
        node = self.node_dict[id]
        if node.is_leaf or not isinstance(node.children, list):
            return
        for child in node.children:
            self.clear_children(child.id)
            _ = self.node_dict.pop(child.id)
        node.children = None
        node.is_leaf = True

    def metrics(
        self,
        pred=None,
        true=None,
        weights=None,
        extra_metrics=None,
        extra_metrics_data=None,
        total_pos=None,
    ):
        """Calculate performance metrics for predictions.

        Args:
            pred (np.array, optional): Predictions. Defaults to model predictions.
            true (pd.Series, optional): True labels. Defaults to training labels.
            weights (pd.Series, optional): Sample weights. Defaults to training weights.
            extra_metrics (dict, optional): Additional metrics. Defaults to model extra_metrics.
            extra_metrics_data (pd.DataFrame, optional): Data for extra metrics.
            total_pos (float, optional): Total positive samples.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        if pred is None:
            pred = self.predict(self.X)
        if true is None:
            true = self.y
        if weights is None:
            weights = self.weights
        if extra_metrics is None:
            extra_metrics = self.extra_metrics
        if extra_metrics_data is None:
            extra_metrics_data = self.extra_metrics_data
        if total_pos is None:
            total_pos = self.total_pos

        metrics = {
            "Precision": get_precision(pred, true, weights),
            "Recall": get_recall(pred, true, weights, total_pos),
            "Positives": (pred * true * weights).sum(),
        }
        if extra_metrics is not None and isinstance(extra_metrics_data, pd.DataFrame):
            for metric in extra_metrics:
                metrics[metric] = extra_metrics[metric](
                    extra_metrics_data.loc[pred == 1]
                )
        return metrics

    def performance(
        self,
        pred=None,
        true=None,
        weights=None,
        extra_metrics=None,
        extra_metrics_data=None,
        total_pos=None,
    ):
        """Print performance metrics for predictions.

        Args:
            pred (np.array, optional): Predictions. Defaults to model predictions.
            true (pd.Series, optional): True labels. Defaults to training labels.
            weights (pd.Series, optional): Sample weights. Defaults to training weights.
            extra_metrics (dict, optional): Additional metrics. Defaults to model extra_metrics.
            extra_metrics_data (pd.DataFrame, optional): Data for extra metrics.
            total_pos (float, optional): Total positive samples.
        """
        metrics = self.metrics(
            pred, true, weights, extra_metrics, extra_metrics_data, total_pos
        )
        print(
            f"Precision: {metrics['Precision']:.2%}, Recall: {metrics['Recall']:.2%}, Positives: {metrics['Positives']:.2f}"
        )
        extra_metrics_str = []
        if extra_metrics is not None and isinstance(extra_metrics_data, pd.DataFrame):
            for metric in extra_metrics:
                extra_metrics_str.append(
                    f"{metric}: {metrics[metric]:.4f}".rstrip("0").rstrip(".")
                )
        print(", ".join(extra_metrics_str))

    def predict(self, X):
        """Generate predictions for given features.

        Args:
            X (pd.DataFrame): Feature matrix to predict on.

        Returns:
            np.array: Binary predictions (0 or 1).
        """
        predictions = self._predict_mask(X, self.tree).astype(int)
        return predictions

    def print(
        self, node: Node = None, current_node: Node = None, show_metrics=False, depth=0
    ):
        """Print tree structure in a readable format.

        Args:
            node (Node, optional): Node to start printing from. Defaults to root.
            current_node (Node, optional): Node to highlight as current.
            show_metrics (bool): Whether to display node metrics. Defaults to False.
            depth (int): Current depth for indentation. Defaults to 0.
        """
        if node is None:
            node = self.tree

        prefix = "|   " * depth
        node_str = f"Node {node.id}: "

        if depth == 0:
            node_str += "Root"
        else:
            node_str += self._node_to_sql(node)

        if show_metrics:
            node_str += "; " + ", ".join(
                [
                    f"{metric}: {node.metrics[metric]:.4f}".rstrip("0").rstrip(".")
                    for metric in node.metrics
                ]
            )

        if current_node is not None and current_node.id == node.id:
            node_str = "<<" + node_str + ">> <- Current Node"
        print(prefix + node_str)

        if isinstance(node.children, list):
            for child in node.children:
                self.print(child, current_node, show_metrics, depth + 1)

        if node.is_leaf and node.children is None:
            print(f"{prefix}--> Prediction: {node.prediction}")

    def to_sql(self, node: Node = None):
        """Convert tree to SQL WHERE clause representation.

        Args:
            node (Node, optional): Node to start conversion from. Defaults to root.

        Returns:
            str: SQL WHERE clause representing the tree logic.
        """

        def dfs(node: Node, path, result):
            if not node:
                return
            path.append(self._node_to_sql(node))
            if node.is_leaf and node.prediction == 1:
                result.append(list(path))
            if isinstance(node.children, list):
                for child in node.children:
                    dfs(child, path, result)
            path.pop()

        if node is None:
            node = self.tree

        result = []
        dfs(node, [], result)

        if len(result) == 0:
            sql = None
        elif len(result) == 1:
            sql = "(" + " AND ".join(result[0][1:]) + ")"
            if sql == "()":
                sql = "(TRUE)"
        else:
            sql = (
                "("
                + " OR ".join(
                    [
                        "(" + " AND ".join(path[1:]) + ")"
                        for path in result
                        if len(path) > 0
                    ]
                )
                + ")"
            )
        return sql

    def to_dict(self):
        """Convert tree to dictionary representation.

        Returns:
            dict: Dictionary containing all nodes with their attributes.
        """
        nodes = {}
        for id in self.node_dict:
            nodes[id] = self.node_dict[id].to_dict()
        return nodes

    def to_json(self):
        """Convert tree to JSON string.

        Returns:
            str: JSON representation of the tree.
        """
        return json.dumps(self.to_dict())

    def from_dict(self, nodes: dict):
        """Load tree from dictionary representation.

        Args:
            nodes (dict): Dictionary containing node data.
        """
        if "0" in nodes:
            nodes = {int(k): nodes[k] for k in nodes}
        self.node_dict = {}
        for id in nodes:
            node = Node()
            for key in nodes[id]:
                if key != "parent" and key != "children":
                    setattr(node, key, nodes[id][key])
            self.node_dict[id] = node

        for id in nodes:
            self.node_dict[id].parent = (
                self.node_dict[nodes[id]["parent"]]
                if isinstance(nodes[id]["parent"], int)
                else None
            )
            self.node_dict[id].children = (
                [self.node_dict[child_id] for child_id in nodes[id]["children"]]
                if isinstance(nodes[id]["children"], list)
                else None
            )

        self.tree = self.node_dict[0]
        self.node_counter = len(self.node_dict)

        if self.X is not None:
            self.reset_tree_data(node=self.tree)
        else:
            print(
                "Tree imported successfully. Please use import_data method to import data."
            )

    def from_json(self, json_str):
        """Load tree from JSON string.

        Args:
            json_str (str): JSON representation of the tree.
        """
        nodes = json.loads(json_str)
        nodes = {int(k): nodes[k] for k in nodes}
        self.from_dict(nodes)

    def save(self, file_path):
        """Save tree to file.

        Args:
            file_path (str): Path to save the tree file.
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, file_path):
        """Load tree from file.

        Args:
            file_path (str): Path to the tree file.
        """
        with open(file_path, "r") as f:
            nodes = json.load(f)
        nodes = {int(k): nodes[k] for k in nodes}
        self.from_dict(nodes)

    def _best_splits(
        self, X, y, weights, cat_features, lt_only_features, gt_only_features
    ):
        """Find the best split conditions for all features.

        Analyzes each feature to find optimal split conditions based on the
        configured sorting metric (f_score, information gain, etc.).

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            weights (pd.Series): Sample weights.
            cat_features (list): List of categorical feature names.
            lt_only_features (list): Features restricted to '<' operators.
            gt_only_features (list): Features restricted to '>' operators.

        Returns:
            dict: Dictionary mapping feature names to sorted lists of split options.
        """
        splits = {}
        node_pos = (y * weights).sum()
        for feature in X.columns:
            if feature not in cat_features and is_numeric_dtype(X[feature]):
                if self.verbose:
                    print(
                        f"Feature {feature} is neither categorical nor numeric, skipping..."
                    )
                continue
            X_feature = X[feature].copy()
            if feature in cat_features:
                if self.ignore_null:
                    categories = X_feature.dropna().unique()
                else:
                    categories = X_feature.unique()
                num_unique = len(categories)

                if num_unique <= 1:
                    if self.verbose:
                        print(
                            f"Feature {feature} has only one unique value, skipping..."
                        )
                    continue

                if self.max_cat_unique is not None and num_unique > self.max_cat_unique:
                    if self.verbose:
                        print(
                            f"Feature {feature} has too many unique values, only use the most frequent {self.max_cat_unique} values..."
                        )
                    cnts = X_feature.value_counts(sort=True, ascending=False)
                    categories = cnts.index[: self.max_cat_unique]
                    X_feature.loc[~X_feature.isin(categories)] = pd.NA

                pos_amts = (y * weights).groupby(X_feature).sum()
                total_amts = weights.groupby(X_feature).sum()
                precision = pos_amts / total_amts
                recall = pos_amts / node_pos
                metrics_df = pd.DataFrame(
                    {
                        "pos_amt": pos_amts,
                        "total_amt": total_amts,
                        "precision": precision,
                        "recall": recall,
                    }
                )
                metrics_df = metrics_df.loc[
                    recall >= self.cat_value_min_recall
                ].sort_values(by="precision", ascending=False)
                if len(metrics_df) == 0:
                    if self.verbose:
                        print(
                            f"Feature {feature} has low recall on all values, skipping..."
                        )
                    continue

                category_sorted = np.array(metrics_df.index)

                split_metrics = []
                split = []

                for category in category_sorted:
                    split.append(category)
                    grp = (
                        metrics_df.index.isin(split)
                        if is_bool_dtype(X_feature)
                        else split
                    )
                    pos_amt = metrics_df.loc[grp, "pos_amt"].sum()
                    total_recall = pos_amt / self.total_pos
                    if total_recall < self.node_min_recall:
                        continue

                    mask = X_feature.isin(split)
                    if np.sum(mask) == 0 or np.sum(~mask) == 0:
                        break

                    subsets = [mask, ~mask]

                    ig = None
                    igr = None
                    iv = None
                    precision = None
                    recall = None
                    f_score = None

                    if self.sorted_by == "ig" or self.sorted_by == "igr":
                        ig = information_gain(y, subsets, weights, self.pos_weight)
                        if self.sorted_by == "igr":
                            igr = information_gain_ratio(
                                ig, intrinsic_information(y, subsets, weights)
                            )
                    if self.sorted_by == "iv":
                        iv = information_value(y, subsets, weights, self.pos_weight)
                    if self.sorted_by == "f_score":
                        precision = pos_amt / metrics_df.loc[grp, "total_amt"].sum()
                        recall = pos_amt / node_pos
                        f_score = get_f_score(
                            precision, recall, self.beta, self.knot, self.factor
                        )

                    if len(split) > num_unique // 2:
                        not_split = [c for c in categories if c not in split]
                        if len(not_split) == 1:
                            split_value = not_split[0]
                            operator = "<>"
                        else:
                            split_value = not_split.copy()
                            operator = "not in"
                    else:
                        if len(split) == 1:
                            split_value = split[0]
                            operator = "="
                        else:
                            split_value = split.copy()
                            operator = "in"
                    split_info = {
                        "feature": feature,
                        "operator": operator,
                        "split_value": split_value,
                        "metrics": {
                            "precision": precision,
                            "recall": recall,
                            "f_score": f_score,
                            "ig": ig,
                            "igr": igr,
                            "iv": iv,
                            "pos_amt": pos_amt,
                            "total_recall": total_recall,
                        },
                    }
                    split_metrics.append(split_info)
                if len(split_metrics) > 0:
                    split_metrics.sort(
                        key=lambda x: x["metrics"][self.sorted_by], reverse=True
                    )
                    splits[feature] = split_metrics

            elif is_numeric_dtype(X_feature):
                _, thresholds = pd.qcut(
                    X_feature, self.num_bin, retbins=True, duplicates="drop"
                )
                if len(thresholds) < 2:
                    if self.verbose:
                        print(
                            f"Feature {feature} has less than 2 unique values, skipping..."
                        )
                    continue
                thresholds = thresholds[1:-1]

                split_metrics = []
                for threshold in thresholds:
                    mask = X_feature >= threshold
                    if np.sum(mask) == 0 or np.sum(~mask) == 0:
                        continue

                    subsets = [mask, ~mask]
                    ig = None
                    igr = None
                    iv = None

                    if self.sorted_by == "ig" or self.sorted_by == "igr":
                        ig = information_gain(y, subsets, weights, self.pos_weight)
                        if self.sorted_by == "igr":
                            igr = information_gain_ratio(
                                ig, intrinsic_information(y, subsets, weights)
                            )
                    if self.sorted_by == "iv":
                        iv = information_value(y, subsets, weights, self.pos_weight)

                    operators = [">=", "<"]
                    max_precision = 0
                    best_split_info = None
                    for j in [0, 1]:
                        if operators[j] == ">=" and feature in lt_only_features:
                            continue
                        if operators[j] == "<" and feature in gt_only_features:
                            continue

                        y_pred = np.where(subsets[j], 1, 0)
                        pos_amt = (y * weights * y_pred).sum()
                        total_recall = pos_amt / self.total_pos
                        if total_recall < self.node_min_recall:
                            continue

                        precision = pos_amt / np.sum(y_pred * weights)
                        recall = pos_amt / node_pos
                        f_score = get_f_score(
                            precision, recall, self.beta, self.knot, self.factor
                        )
                        operator = operators[j]
                        split_info = {
                            "feature": feature,
                            "operator": operator,
                            "split_value": threshold,
                            "metrics": {
                                "precision": precision,
                                "recall": recall,
                                "f_score": f_score,
                                "ig": ig,
                                "igr": igr,
                                "iv": iv,
                                "pos_amt": pos_amt,
                                "total_recall": total_recall,
                            },
                        }

                        if not self.sorted_by in ["ig", "igr", "iv"]:
                            split_metrics.append(split_info)
                        elif precision > max_precision:
                            best_split_info = split_info
                            max_precision = precision

                    if best_split_info is not None:
                        split_metrics.append(best_split_info)
                if len(split_metrics) > 0:
                    split_metrics.sort(
                        key=lambda x: x["metrics"][self.sorted_by], reverse=True
                    )
                    splits[feature] = split_metrics

        return splits

    def _build_tree(
        self,
        X,
        y,
        weights,
        cat_features,
        lt_only_features,
        gt_only_features,
        pinned_features,
        extra_metrics,
        extra_metrics_data,
        auto,
        node,
    ):
        """Recursively build the decision tree.

        Core tree building logic that handles node splitting, stopping criteria,
        and recursive expansion of child nodes.

        Args:
            X (pd.DataFrame): Feature matrix for current node.
            y (pd.Series): Target variable for current node.
            weights (pd.Series): Sample weights for current node.
            cat_features (list): Categorical feature names.
            lt_only_features (list): Features restricted to '<' operators.
            gt_only_features (list): Features restricted to '>' operators.
            pinned_features (list): Features to prioritize.
            extra_metrics (dict): Additional metrics to calculate.
            extra_metrics_data (pd.DataFrame): Data for extra metrics.
            auto (bool): Whether to build automatically.
            node (Node): Current node being processed.
        """
        self.print(self.tree, node, self.show_metrics)

        precision = node.metrics["Precision"]
        recall = node.metrics["Recall"]
        pos_amt = node.metrics["Positives"]

        def print_exit_message():
            metrics_str = f"Precision: {precision:.2%}, Recall: {recall:.2%}, Positives: {pos_amt:.2f}"
            if extra_metrics is not None and isinstance(
                extra_metrics_data, pd.DataFrame
            ):
                for metric in extra_metrics:
                    metrics_str += f", {metric}: {extra_metrics[metric](extra_metrics_data):.4f}".rstrip(
                        "0"
                    ).rstrip(
                        "."
                    )
            print(metrics_str)

        if (
            node.operator in (">=", ">") and node.split_feature in lt_only_features
        ) or (node.operator in ("<=", "<") and node.split_feature in gt_only_features):
            print(
                "Cannot split due to lt_only_features or gt_only_features constraints."
            )
            print_exit_message()
            return

        if node.depth >= self.max_depth:
            print(f"Reached max depth {self.max_depth}.")
            print_exit_message()
            return

        if precision > self.node_max_precision:
            print(
                f"Precision {precision:.2%} reached satisfactory level {self.node_max_precision:.2%}."
            )
            print_exit_message()
            return

        try:
            left_mask, right_mask = self._split_node(
                X,
                y,
                weights,
                cat_features,
                lt_only_features,
                gt_only_features,
                pinned_features,
                extra_metrics,
                extra_metrics_data,
                auto,
                node,
            )
        except ExitSplit as e:
            print(str(e))
            if str(e) == "Exit by user.":
                print("Continue to next node.")
            print_exit_message()
            return

        if isinstance(node.children, list) and len(node.children) >= 2:
            left_node = node.children[0]
            right_node = node.children[1]
            print(f"\nFinding split for node {left_node.id}:")
            if isinstance(extra_metrics_data, pd.DataFrame):
                self._build_tree(
                    X[left_mask],
                    y[left_mask],
                    weights[left_mask],
                    cat_features,
                    lt_only_features,
                    gt_only_features,
                    pinned_features,
                    extra_metrics,
                    extra_metrics_data[left_mask],
                    auto,
                    left_node,
                )
            else:
                self._build_tree(
                    X[left_mask],
                    y[left_mask],
                    weights[left_mask],
                    cat_features,
                    lt_only_features,
                    gt_only_features,
                    pinned_features,
                    None,
                    None,
                    auto,
                    left_node,
                )

            print(f"\nFinding split for node {right_node.id}:")
            if isinstance(extra_metrics_data, pd.DataFrame):
                self._build_tree(
                    X[right_mask],
                    y[right_mask],
                    weights[right_mask],
                    cat_features,
                    lt_only_features,
                    gt_only_features,
                    pinned_features,
                    extra_metrics,
                    extra_metrics_data[right_mask],
                    auto,
                    right_node,
                )
            else:
                self._build_tree(
                    X[right_mask],
                    y[right_mask],
                    weights[right_mask],
                    cat_features,
                    lt_only_features,
                    gt_only_features,
                    pinned_features,
                    None,
                    None,
                    auto,
                    right_node,
                )

    def _split_node(
        self,
        X,
        y,
        weights,
        cat_features,
        lt_only_features,
        gt_only_features,
        pinned_features,
        extra_metrics,
        extra_metrics_data,
        auto,
        node,
    ):
        """Handle interactive or automatic node splitting.

        Manages the process of selecting features and split conditions,
        either through user interaction or automatic selection.

        Args:
            X (pd.DataFrame): Feature matrix for current node.
            y (pd.Series): Target variable for current node.
            weights (pd.Series): Sample weights for current node.
            cat_features (list): Categorical feature names.
            lt_only_features (list): Features restricted to '<' operators.
            gt_only_features (list): Features restricted to '>' operators.
            pinned_features (list): Features to prioritize.
            extra_metrics (dict): Additional metrics to calculate.
            extra_metrics_data (pd.DataFrame): Data for extra metrics.
            auto (bool): Whether to split automatically.
            node (Node): Node to split.

        Returns:
            tuple: Left and right masks for data splitting.

        Raises:
            ExitSplit: When splitting should be terminated.
        """
        if cat_features is None:
            cat_features = []
        if lt_only_features is None:
            lt_only_features = []
        if gt_only_features is None:
            gt_only_features = []
        if pinned_features is None:
            pinned_features = []

        recall = node.metrics["Recall"]

        if len(y) == 0 or np.sum(y) == 0:
            raise ExitSplit("No positives in the node.")

        if recall < self.node_min_recall:
            raise ExitSplit(f"Node recall {recall:.2%} is too low.")

        if len(y) == np.sum(y):
            raise ExitSplit("All samples are positives.")

        splits = self._best_splits(
            X, y, weights, cat_features, lt_only_features, gt_only_features
        )

        if splits is None or len(splits) == 0:
            raise ExitSplit("No valid splits found.")

        top_features = sorted(
            list(splits.items()),
            key=lambda x: x[1][0]["metrics"][self.sorted_by],
            reverse=True,
        )
        ranked_start_idx = 0
        if len(pinned_features) > 0:
            pinned_feature_splits = [
                (feature, split)
                for feature, split in top_features
                if feature in pinned_features
            ]
            top_features = pinned_feature_splits + top_features
            ranked_start_idx = len(pinned_feature_splits)

        print(f"\nCurrent depth: {node.depth}")

        def input_feature():
            feature_max_idx = self.feature_shown_num
            chosen_feature = None
            while True:
                chosen_feature = input(
                    f"\nSelect a feature by displayed rank number (default {ranked_start_idx}) or feature name (case sensitive).\nCommands:\n  /m: More options\n  /q: Quit\n>> "
                )
                if chosen_feature == "/q":
                    break
                elif chosen_feature == "/m":
                    print()
                    if feature_max_idx > len(top_features):
                        print(f"Only {len(top_features)} features are available.")
                    else:
                        for i, split in enumerate(
                            top_features[
                                feature_max_idx : feature_max_idx
                                + self.feature_shown_num
                            ]
                        ):
                            if feature_max_idx + i == ranked_start_idx:
                                print(f"\nBest features and split values:")
                                feature, splits_info = split
                                print(f"{feature_max_idx + i}. {feature}")
                                split_info = splits_info[0]
                                self._print_metrics(
                                    split_info,
                                    X,
                                    y,
                                    weights,
                                    extra_metrics,
                                    extra_metrics_data,
                                    "Best Split",
                                )
                                feature_max_idx += self.feature_shown_num
                elif chosen_feature == "":
                    chosen_feature = top_features[ranked_start_idx][0]
                    break
                elif chosen_feature.isdigit() and int(chosen_feature) < len(
                    top_features
                ):
                    chosen_feature = top_features[int(chosen_feature)][0]
                    break
                elif chosen_feature not in top_features:
                    print(f"Invalid feature name. Please try again.")
                elif chosen_feature not in splits:
                    print(f"No split found for feature {chosen_feature}.")
                else:
                    break
            return chosen_feature

        def input_condition(chosen_feature):
            split_max_idx = self.condition_shown_num
            split_info = None
            while True:
                chosen_split = input(
                    f"\nSelect a split condition by displayed rank number (default 0) or a SQL-style condition (e.g. >=100).\nCommands:\n  /m: More options\n  /b Back to feature selection\n  /q: Quit\n>> "
                )
                if chosen_split == "/q" or chosen_split == "/b":
                    split_info = chosen_split
                    break
                elif chosen_split == "/m":
                    print()
                    if split_max_idx > len(splits[chosen_feature]):
                        print(
                            f"Only {len(splits[chosen_feature])} splits are available."
                        )
                    else:
                        for j, split_info in enumerate(
                            splits[chosen_feature][
                                split_max_idx : split_max_idx + self.condition_shown_num
                            ]
                        ):
                            self._print_metrics(
                                split_info,
                                X,
                                y,
                                weights,
                                extra_metrics,
                                extra_metrics_data,
                                f"Split {split_max_idx + j}",
                            )
                            split_max_idx += self.condition_shown_num
                elif chosen_split == "":
                    split_info = splits[chosen_feature][0]
                    break
                elif chosen_split.isdigit():
                    try:
                        chosen_split = int(chosen_split)
                        split_info = splits[chosen_feature][chosen_split]
                        break
                    except:
                        print(f"Invalid split index. Please try again.")
                else:
                    try:
                        while True:
                            custom_operator, custom_value = self._parse_partial_sql(
                                chosen_split
                            )
                            split_info = self._custom_split(
                                chosen_feature,
                                custom_operator,
                                custom_value,
                                X,
                                y,
                                weights,
                            )
                            print("\nDetected custom split condition.")
                            self._print_metrics(
                                split_info,
                                X,
                                y,
                                weights,
                                extra_metrics,
                                extra_metrics_data,
                                "Custom Split",
                            )
                            satisfied = input(
                                f"\nIf satisfied, press Enter. Otherwise, enter another condition.\n>> "
                            )
                            if satisfied == "":
                                break
                            else:
                                chosen_split = satisfied
                        break
                    except:
                        print(f"Invalid SQL condition. Please try again.")
            return split_info

        if auto:
            chosen_feature, split_info = top_features[ranked_start_idx]
            split_info = split_info[0]
            print(f"\nAuto-selected feature {chosen_feature}.")
            self._print_metrics(
                split_info,
                X,
                y,
                weights,
                extra_metrics,
                extra_metrics_data,
                "Auto-selected Split",
            )
        else:
            if ranked_start_idx >= 0:
                print("Pinned features and split values:")
            for i, split in enumerate(top_features[: self.feature_shown_num]):
                if i == ranked_start_idx:
                    print(f"\nBest features and split values:")
                feature, splits_info = split
                print(f"{i}. {feature}")
                split_info = splits_info[0]
                self._print_metrics(
                    split_info,
                    X,
                    y,
                    weights,
                    extra_metrics,
                    extra_metrics_data,
                    "Best Split",
                )

            split_info = None
            while not isinstance(split_info, dict):
                chosen_feature = input_feature()
                if chosen_feature == "/q":
                    raise ExitSplit("Exit by user.")

                print(f"\nFeature {chosen_feature} is selected. Top splits:")
                for j, split_info in enumerate(
                    splits[chosen_feature][: self.condition_shown_num]
                ):
                    self._print_metrics(
                        split_info,
                        X,
                        y,
                        weights,
                        extra_metrics,
                        extra_metrics_data,
                        f"Split {j}",
                    )
                split_info = input_condition(chosen_feature)
                if split_info == "/q":
                    raise ExitSplit("Exit by user.")

        left_mask, right_mask = self._create_children(
            chosen_feature,
            split_info["operator"],
            split_info["split_value"],
            X,
            y,
            weights,
            node,
        )
        return left_mask, right_mask

    def _quick_split_node(
        self,
        feature,
        operator,
        value,
        X,
        y,
        weights,
        extra_metrics,
        extra_metrics_data,
        node,
    ):
        """Perform quick split with predefined condition.

        Args:
            feature (str): Feature name for splitting.
            operator (str): Comparison operator.
            value: Split value.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            weights (pd.Series): Sample weights.
            extra_metrics (dict): Additional metrics.
            extra_metrics_data (pd.DataFrame): Data for extra metrics.
            node (Node): Node to split.

        Returns:
            tuple: Left and right masks for data splitting.
        """
        recall = node.metrics["Recall"]

        if len(y) == 0 or np.sum(y) == 0:
            raise ExitSplit("No positives in the node.")

        if recall < self.node_min_recall:
            raise ExitSplit(f"Node recall {recall:.2%} is too low.")

        if len(y) == np.sum(y):
            raise ExitSplit("All samples are positives.")

        split_info = self._custom_split(feature, operator, value, X, y, weights)

        self._print_metrics(
            split_info, X, y, weights, extra_metrics, extra_metrics_data, "Custom Split"
        )

        left_mask, right_mask = self._create_children(
            feature, operator, value, X, y, weights, node
        )
        return left_mask, right_mask

    def _create_children(self, feature, operator, value, X, y, weights, node):
        """Create child nodes after splitting.

        Creates left and right child nodes, calculates their metrics,
        and updates the tree structure.

        Args:
            feature (str): Feature used for splitting.
            operator (str): Comparison operator.
            value: Split value.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            weights (pd.Series): Sample weights.
            node (Node): Parent node.

        Returns:
            tuple: Left and right masks for the split.
        """
        node.is_leaf = False
        node.children = []

        left_mask = self._split_to_mask(X, feature, operator, value)
        left_node = Node(
            index=X[left_mask].index,
            parent=node,
            split_feature=feature,
            operator=operator,
            split_value=value,
            is_leaf=True,
            depth=node.depth + 1,
            id=node.id * 2,
        )
        left_precision = get_precision(
            np.ones(len(y[left_mask]), dtype="float64"),
            y[left_mask],
            weights[left_mask],
        )
        left_recall = get_recall(
            np.ones(len(y[left_mask]), dtype="float64"),
            y[left_mask],
            weights[left_mask],
            self.total_pos,
        )
        left_node.metrics = {
            "Precision": left_precision,
            "Recall": left_recall,
            "Positives": np.sum(y[left_mask] * weights[left_mask]),
        }
        left_node.prediction = 1

        self.node_dict[left_node.id] = left_node
        self.node_counter = len(self.node_dict)
        node.children.append(left_node)
        right_mask = ~left_mask
        na_cnt = 0
        if self.ignore_null:
            na_cnt = (right_mask & X[feature].isna()).sum()
            if na_cnt > 0:
                right_mask = right_mask & X[feature].notna()
        right_node = Node(
            index=X[right_mask].index,
            parent=node,
            split_feature=feature,
            operator=self._reverse_operator(operator),
            split_value=value,
            is_leaf=True,
            depth=node.depth + 1,
            id=node.id * 2 + 1,
        )
        right_precision = get_precision(
            np.ones(len(y[right_mask]), dtype="float64"),
            y[right_mask],
            weights[right_mask],
        )
        right_recall = get_recall(
            np.ones(len(y[right_mask]), dtype="float64"),
            y[right_mask],
            weights[right_mask],
            self.total_pos,
        )
        right_node.metrics = {
            "Precision": right_precision,
            "Recall": right_recall,
            "Positives": np.sum(y[right_mask] * weights[right_mask]),
        }
        right_node.prediction = 0
        self.node_dict[right_node.id] = right_node
        self.node_counter = len(self.node_dict)
        node.children.append(right_node)

        print(
            f"\nChild nodes created: {left_node.id} (positive) and {right_node.id} (negative)"
        )
        if na_cnt > 0:
            print(f"Omitted {na_cnt} rows with null values in Node {right_node.id}.")

        return left_mask, right_mask

    def _custom_split(self, feature, operator, value, X, y, weights):
        """Create a custom split condition and calculate its metrics.

        Evaluates a user-defined split condition by applying it to the data
        and calculating all relevant performance metrics including precision,
        recall, F-score, information gain, and information value.

        Args:
            feature (str): Feature name for the split condition.
            operator (str): Comparison operator (e.g., '>=', '<', 'in', '=').
            value: Value(s) to compare against in the split condition.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            weights (pd.Series): Sample weights.

        Returns:
            dict: Dictionary containing split information and calculated metrics.

        Raises:
            ValueError: If split condition is invalid or results in no positives.
        """
        try:
            mask = self._split_to_mask(X, feature, operator, value)
        except:
            raise ValueError("Invalid split condition.")
        if mask.sum() == 0 or (weights[mask] * y[mask]).sum() == 0:
            raise ValueError("No positives in the subset.")

        subsets = [mask, ~mask]
        y_pred = np.where(mask, 1, 0)

        pos_amt = (y * weights * y_pred).sum()
        total_recall = pos_amt / self.total_pos
        ig = information_gain(y, subsets, weights, self.pos_weight)
        igr = information_gain_ratio(ig, intrinsic_information(y, subsets, weights))
        iv = information_value(y, subsets, weights, self.pos_weight)
        precision = pos_amt / np.sum(y_pred * weights)
        recall = pos_amt / self.total_pos
        f_score = get_f_score(precision, recall, self.beta, self.knot, self.factor)
        split_info = {
            "feature": feature,
            "operator": operator,
            "split_value": value,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
                "ig": ig,
                "igr": igr,
                "iv": iv,
                "pos_amt": pos_amt,
                "total_recall": total_recall,
            },
        }
        return split_info

    def _predict_mask(self, X, node):
        """Generate prediction mask for samples at a given node.

        Recursively traverses the tree from the specified node to determine
        which samples should receive positive predictions based on the tree
        structure and split conditions.

        Args:
            X (pd.DataFrame): Feature matrix to generate predictions for.
            node (Node): Node to start prediction from.

        Returns:
            pd.Series: Boolean mask indicating positive predictions.
        """
        if (
            node.is_leaf or not isinstance(node.children, list)
        ) and node.prediction == 0:
            pred_mask = pd.Series(False, index=X.index)
        elif node is self.tree:
            pred_mask = pd.Series(True, index=X.index)
        else:
            pred_mask = self._split_to_mask(
                X, node.split_feature, node.operator, node.split_value
            )

        if node.is_leaf or not isinstance(node.children, list):
            return pred_mask

        child_mask = pd.Series(False, index=X.index)
        for child in node.children:
            child_mask = child_mask | self._predict_mask(X, child)
        return pred_mask & child_mask

    def _value_to_sql(self, value):
        """Convert a Python value to its SQL string representation.

        Handles various data types including strings, numbers, lists, and null values,
        formatting them appropriately for SQL WHERE clauses.

        Args:
            value: Python value to convert to SQL format.

        Returns:
            str: SQL-formatted string representation of the value.
        """
        if isinstance(value, str):
            sql_value = f"'{value}'"
        elif value is None or value == np.nan or value == "":
            sql_value = "NULL"
        elif isinstance(value, (float, np.floating)):
            sql_value = f"{value:.6f}".rstrip("0").rstrip(".")
        elif isinstance(value, list):
            sql_value = f"({', '.join([self._value_to_sql(v) for v in value])})"
        else:
            sql_value = str(value)
        return sql_value

    def _node_to_sql(self, node):
        """Convert a node's split condition to SQL WHERE clause format.

        Args:
            node (Node): Node containing split condition information.

        Returns:
            str: SQL WHERE clause representing the node's split condition.
        """
        return f"{node.split_feature} {node.operator} {self._value_to_sql(node.split_value)}"

    def _parse_partial_sql(self, sql):
        """Parse a partial SQL condition string to extract operator and values.

        Parses SQL-style conditions that contain only operator and values
        (without feature name), commonly used in interactive mode.

        Args:
            sql (str): Partial SQL condition string (e.g., '>=100', 'in (1,2,3)').

        Returns:
            tuple: (operator, values) extracted from the SQL string.

        Raises:
            ValueError: If the SQL condition format is invalid.
        """
        sql = sql.strip()
        pattern = r"(<>|!=|\bnot\s+in\b|\bin\b|[<>=]=?)\s*(.+)"
        match = re.search(pattern, sql, re.IGNORECASE)
        try:
            operator = match.group(1).lower().strip()
            value = match.group(2).strip()
            value = ast.literal_eval(value)
            if isinstance(value, tuple):
                value = list(value)
            return operator, value
        except:
            raise ValueError("Invalid SQL condition.")

    def _parse_sql(self, sql):
        """Parse a complete SQL condition string to extract feature, operator, and values.

        Parses full SQL WHERE clause conditions containing feature name,
        operator, and values.

        Args:
            sql (str): Complete SQL condition string (e.g., 'age >= 25', 'city in ("NYC", "LA")').

        Returns:
            tuple: (feature, operator, values) extracted from the SQL string.

        Raises:
            ValueError: If the SQL condition format is invalid.
        """
        sql = sql.strip()
        pattern = r"(\S+?)\s*(<>|!=|\bnot\s+in\b|\bin\b|[<>=]=?)\s*(.+)"
        match = re.search(pattern, sql, re.IGNORECASE)
        try:
            feature = match.group(1).strip()
            operator = match.group(2).lower().strip()
            value = match.group(3).strip()
            value = ast.literal_eval(value)
            if isinstance(value, tuple):
                value = list(value)
            return feature, operator, value
        except:
            raise ValueError("Invalid SQL condition.")

    def _split_to_mask(self, X, feature, operator, value):
        """Convert split condition to boolean mask for data filtering.

        Applies the split condition (feature, operator, value) to the data
        and returns a boolean mask indicating which samples satisfy the condition.

        Args:
            X (pd.DataFrame): Feature matrix.
            feature (str): Feature name for the condition.
            operator (str): Comparison operator.
            value: Value(s) to compare against.

        Returns:
            pd.Series or bool: Boolean mask or False if operator not supported.
        """
        if operator in operator_map:
            return operator_map[operator](X[feature], value)
        return False

    def _reverse_operator(self, operator):
        """Get the logical opposite of a comparison operator.

        Returns the inverse operator for creating complementary split conditions
        in child nodes.

        Args:
            operator (str): Original comparison operator.

        Returns:
            str: Reversed/opposite operator.
        """
        if operator in operator_reverse_map:
            return operator_reverse_map[operator]
        return f"not {operator}"

    def _print_metrics(
        self,
        split_info,
        X,
        y,
        weights,
        extra_metrics,
        extra_metrics_data,
        split_name="Split",
    ):
        """Print detailed performance metrics for a split condition.

        Displays comprehensive metrics including information gain, precision,
        recall, F-score, and any additional custom metrics for a given split.

        Args:
            split_info (dict): Dictionary containing split condition and metrics.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            weights (pd.Series): Sample weights.
            extra_metrics (dict, optional): Additional custom metrics to calculate.
            extra_metrics_data (pd.DataFrame, optional): Data for extra metrics.
            split_name (str): Name/label for the split in the output. Defaults to "Split".
        """
        feature = split_info["feature"]
        operator = split_info["operator"]
        value = split_info["split_value"]
        metrics = split_info["metrics"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f_score = metrics["f_score"]
        ig = metrics["ig"]
        igr = metrics["igr"]
        iv = metrics["iv"]
        pos_amt = metrics["pos_amt"]
        total_recall = metrics["total_recall"]

        mask = self._split_to_mask(X, feature, operator, value)
        subsets = [mask, ~mask]
        y_pred = np.where(mask, 1, 0)

        if ig is None:
            ig = information_gain(y, subsets, weights, self.pos_weight)
        if igr is None:
            igr = information_gain_ratio(ig, intrinsic_information(y, subsets, weights))
        if iv is None:
            iv = information_value(y, subsets, weights, self.pos_weight)
        if precision is None:
            precision = get_precision(y_pred, y, weights)
        if recall is None:
            recall = get_recall(y_pred, y, weights)
        if f_score is None:
            f_score = get_f_score(precision, recall, self.beta, self.knot, self.factor)
        if pos_amt is None:
            pos_amt = (y * weights * y_pred).sum()
        if total_recall is None:
            total_recall = pos_amt / self.total_pos

        print(f"  {split_name}: {operator} {self._value_to_sql(value)}")
        print(f"    IG: {ig:.4g}, IGR: {igr:.4g}, IV: {iv:.4g}")
        print(
            f"    Precision: {precision:.2%}, Node Recall: {recall:.2%}, F-score: {f_score:.4g}"
        )

        metrics_str = f"    Positives: {pos_amt:.2f}, Total Recall: {total_recall:.2%}"
        if extra_metrics is not None and isinstance(extra_metrics_data, pd.DataFrame):
            for metric in extra_metrics:
                metrics_str += f", {metric}: {extra_metrics[metric](extra_metrics_data.loc[mask]):.4f}".rstrip(
                    "0"
                ).rstrip(
                    "."
                )

        print(metrics_str)

    def __repr__(self) -> str:
        """Return string representation of AsymmeTree parameters.

        Returns:
            str: String representation showing key configuration parameters.
        """
        return f"AsymmeTree(max_depth={self.max_depth}, max_cat_unique={self.max_cat_unique}, cat_value_min_recall={self.cat_value_min_recall}, num_bin={self.num_bin}, node_max_precision={self.node_max_precision}, node_min_recall={self.node_min_recall}, leaf_min_precision={self.leaf_min_precision}, feature_shown_num={self.feature_shown_num}, condition_shown_num={self.condition_shown_num}, sorted_by={self.sorted_by}, pos_weight={self.pos_weight}, beta={self.beta}, knot={self.knot}, factor={self.factor}, ignore_null={self.ignore_null}, show_metrics={self.show_metrics}, verbose={self.verbose})"


class ExitSplit(Exception):
    """Exception raised to exit node splitting process.

    Used to signal early termination of the splitting process due to
    various conditions like user intervention, insufficient data quality,
    or reaching stopping criteria.
    """

    pass
