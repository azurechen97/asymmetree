import ast
import re
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype

from handytree.metrics import *
from handytree.utils import *


class Node:
    def __init__(
        self,
        index=None,
        parent=None,
        children=None,
        split_feature=None,
        operator=None,
        split_values=None,
        is_leaf=True,
        prediction=None,
        metrics=None,
        depth=None,
        id=None,
    ):
        self.index = index
        self.parent = parent
        self.children = children
        self.split_feature = split_feature
        self.operator = operator
        self.split_values = split_values
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.metrics = metrics
        self.depth = depth
        self.id = id

    def __repr__(self):
        return f"Node(id={self.id}, depth={self.depth}, is_leaf={self.is_leaf}, split_feature={self.split_feature}, operator={self.operator}, split_values={self.split_values}, prediction={self.prediction}, metrics={self.metrics})"

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        d = self.__dict__.copy()
        del d["indices"]
        del d["metrics"]

        d["parent"] = self.parent.id if isinstance(self.parent, Node) else None
        d["children"] = (
            [child.id for child in self.children if isinstance(child, Node)]
            if isinstance(self.children, list)
            else None
        )

        return d

    def to_json(self):
        return json.dumps(self.to_dict())


class HandyTree:
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
        ignore_null=True,
        show_metrics=False,
        verbose=False,
    ):
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
        self.node_dict = None
        self.node_counter = None

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
        if isinstance(self.y, np.ndarray):
            self.y = pd.Series(self.y)
        if isinstance(self.y, pd.Series) and not self.y.index.equals(self.X.index):
            self.y.index = self.X.index

        X_columns = self.X.columns.tolist()
        str_columns = [col for col in X_columns if is_string_dtype(self.X[col])]
        self.cat_features = [
            col for col in X_columns if col in str_columns or col in self.cat_features
        ]

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
            self.node_dict = {0: self.tree}
            self.node_counter = 1
        else:
            self.reset_tree_data(node=self.tree)

    def reset_tree_data(self, node: Node = None, mask=None):
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
                left_child.split_values,
            )
            self.reset_tree_data(node=left_child, mask=left_mask & mask)
            self.reset_tree_data(node=right_child, mask=(~left_mask) & mask)

    def split(self, id: int = None, auto=False):
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
                feature, operator, values = self._parse_partial_sql(sql)
                _, _ = self._split_node_by_sql(
                    feature,
                    operator,
                    values,
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
        if id not in self.node_dict:
            raise ValueError(f"Node with id {id} not found")
        node = self.node_dict[id]
        if not node.is_leaf:
            raise ValueError(f"Node with id {id} is not a leaf")
        node.prediction = 1 - node.prediction

    def relabel(self, min_precision=None, min_recall=None):
        if min_precision is None:
            min_precision = self.leaf_min_precision
        if min_recall is None:
            min_recall = self.node_min_recall
        for id in self.node_dict:
            node = self.node_dict[id]
            precision = node.metrics["precision"]
            recall = node.metrics["recall"]
            if node.is_leaf:
                node.prediction = (
                    1 if precision >= min_precision and recall >= min_recall else 0
                )

    def prune(self, node: Node = None):
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
        predictions = self._predict_mask(X, self.tree).astype(int)
        return predictions

    def print(
        self, node: Node = None, current_node: Node = None, show_metrics=False, depth=0
    ):
        if node is None:
            node = self.tree

        prefix = "|   " * depth
        node_str = f"Node {node.id}"

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
        nodes = {}
        for id in self.node_dict:
            nodes[id] = self.node_dict[id].to_dict()
        return nodes

    def to_json(self):
        return json.dumps(self.to_dict())

    def from_dict(self, nodes: dict):
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
        nodes = json.loads(json_str)
        nodes = {int(k): nodes[k] for k in nodes}
        self.from_dict(nodes)

    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, file_path):
        with open(file_path, "r") as f:
            nodes = json.load(f)
        nodes = {int(k): nodes[k] for k in nodes}
        self.from_dict(nodes)

    def _best_splits(
        self, X, y, weights, cat_features, lt_only_features, gt_only_features
    ):
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
                            split_values = not_split[0]
                            operator = "<>"
                        else:
                            split_values = not_split.copy()
                            operator = "not in"
                    else:
                        if len(split) == 1:
                            split_values = split[0]
                            operator = "="
                        else:
                            split_values = split.copy()
                            operator = "in"
                    split_info = {
                        "feature": feature,
                        "operator": operator,
                        "split_values": split_values,
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
                            "split_values": threshold,
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
        pass

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
        pass

    def _split_node_by_sql(
        self,
        feature,
        operator,
        values,
        X,
        y,
        weights,
        extra_metrics,
        extra_metrics_data,
        node,
    ):
        pass

    def _create_children(self, feature, operator, values, X, y, weights, node):
        pass

    def _custom_split(self, feature, operator, values, X, y, weights):
        pass

    def _predict_mask(self, X, node):
        pass

    def _value_to_sql(self, value):
        pass

    def _node_to_sql(self, node):
        pass

    def _parse_partial_sql(self, sql):
        pass

    def _parse_sql(self, sql):
        pass

    def _split_to_mask(self, X, feature, operator, values):
        pass

    def _reverse_operator(self, operator):
        pass

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
        pass

    def __repr__(self) -> str:
        return f"HandyTree(max_depth={self.max_depth}, max_cat_unique={self.max_cat_unique}, cat_value_min_ratio={self.cat_value_min_ratio}, num_bin={self.num_bin}, node_max_precision={self.node_max_precision}, node_min_recall={self.node_min_recall}, leaf_min_precision={self.leaf_min_precision}, feature_shown_num={self.feature_shown_num}, condition_shown_num={self.condition_shown_num}, sorted_by={self.sorted_by}, ignore_null={self.ignore_null}, show_metrics={self.show_metrics}, verbose={self.verbose})"


class ExitSplit(Exception):
    pass
