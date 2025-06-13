"""
Metrics module for HandyTree decision tree evaluation.

This module provides various metrics for evaluating decision tree performance,
including information-theoretic measures and classification metrics.
All functions are optimized with Numba for fast computation.
"""

import numpy as np
import pandas as pd

from numba import njit


@njit
def _entropy(y, weights, pos_weight):
    """Calculate weighted entropy for binary classification.

    Args:
        y (np.array): Binary target values.
        weights (np.array): Sample weights.
        pos_weight (float): Weight for positive class.

    Returns:
        float: Weighted entropy value.
    """
    p = np.sum(y * weights) / np.sum(weights)
    entropy = -(
        pos_weight * p * np.log2(p if p > 0 else 1)
        + (1 - p) * np.log2(1 - p if 1 - p > 0 else 1)
    )
    return entropy


@njit
def _information_gain_njit(y, s0, s1, weights, pos_weight):
    """Calculate information gain using Numba-optimized computation.

    Args:
        y (np.array): Binary target values.
        s0 (np.array): Boolean mask for first subset.
        s1 (np.array): Boolean mask for second subset.
        weights (np.array): Sample weights.
        pos_weight (float): Weight for positive class.

    Returns:
        float: Information gain value.
    """
    n = np.sum(weights)

    y0 = y[s0]
    w0 = weights[s0]
    n0 = np.sum(w0)

    y1 = y[s1]
    w1 = weights[s1]
    n1 = np.sum(w1)

    subset_entropy = (n0 / n) * _entropy(y0, w0, pos_weight) if n0 > 0 else 0
    subset_entropy += (n1 / n) * _entropy(y1, w1, pos_weight) if n1 > 0 else 0
    gain = _entropy(y, weights, pos_weight) - subset_entropy

    return gain


def information_gain(y, subsets, weights=None, pos_weight=1):
    """Calculate information gain for a split.

    Information gain measures the reduction in entropy achieved by splitting
    the data according to the given subsets.

    Args:
        y (pd.Series or np.array): Binary target values.
        subsets (list): List of boolean masks defining the split.
        weights (pd.Series or np.array, optional): Sample weights. Defaults to equal weights.
        pos_weight (float): Weight for positive class. Defaults to 1.

    Returns:
        float: Information gain value (higher is better).
    """
    if weights is None:
        weights = np.ones(len(y), dtype="float64")

    if isinstance(y, pd.Series):
        y = y.to_numpy(dtype="int64")
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy(dtype="float64")
    for i, s in enumerate(subsets):
        if isinstance(s, pd.Series):
            subsets[i] = s.to_numpy(dtype="bool")

    return _information_gain_njit(y, subsets[0], subsets[1], weights, pos_weight)


@njit
def _intrinsic_information_njit(s0, s1, weights):
    """Calculate intrinsic information using Numba-optimized computation.

    Args:
        s0 (np.array): Boolean mask for first subset.
        s1 (np.array): Boolean mask for second subset.
        weights (np.array): Sample weights.

    Returns:
        float: Intrinsic information value.
    """
    n = np.sum(weights)

    w0 = weights[s0]
    n0 = np.sum(w0)

    w1 = weights[s1]
    n1 = np.sum(w1)

    ii = -n0 / n * (np.log2(n0) - np.log2(n)) if n0 > 0 else 0
    ii += -n1 / n * (np.log2(n1) - np.log2(n)) if n1 > 0 else 0
    return ii


def intrinsic_information(y, subsets, weights=None):
    """Calculate intrinsic information for a split.

    Intrinsic information measures how much information is used to split
    the data, independent of the target variable.

    Args:
        y (pd.Series or np.array): Target values (used for shape consistency).
        subsets (list): List of boolean masks defining the split.
        weights (pd.Series or np.array, optional): Sample weights. Defaults to equal weights.

    Returns:
        float: Intrinsic information value.
    """
    if weights is None:
        weights = np.ones(len(y), dtype="float64")

    if isinstance(y, pd.Series):
        y = y.to_numpy(dtype="int64")
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy(dtype="float64")
    for i, s in enumerate(subsets):
        if isinstance(s, pd.Series):
            subsets[i] = s.to_numpy(dtype="bool")

    return _intrinsic_information_njit(subsets[0], subsets[1], weights)


def information_gain_ratio(ig, ii):
    """Calculate information gain ratio.

    Information gain ratio normalizes information gain by intrinsic information
    to reduce bias toward splits with many outcomes.

    Args:
        ig (float): Information gain value.
        ii (float): Intrinsic information value.

    Returns:
        float: Information gain ratio (higher is better, 0 if ii=0).
    """
    if ii == 0:
        return 0
    return ig / ii


@njit
def _information_value_njit(y, s0, s1, weights, pos_weight):
    """Calculate information value using Numba-optimized computation.

    Args:
        y (np.array): Binary target values.
        s0 (np.array): Boolean mask for first subset.
        s1 (np.array): Boolean mask for second subset.
        weights (np.array): Sample weights.
        pos_weight (float): Weight for positive class.

    Returns:
        float: Information value.
    """
    total_pos = np.sum(weights[y == 1]) * pos_weight
    total_neg = np.sum(weights[y == 0])

    y0 = y[s0]
    w0 = weights[s0]
    pos0 = np.sum(w0[y0 == 1]) * pos_weight
    neg0 = np.sum(w0[y0 == 0])
    p_pos0 = pos0 / total_pos if total_pos > 0 and pos0 > 0 else 1e-10
    p_neg0 = neg0 / total_neg if total_neg > 0 and neg0 > 0 else 1e-10

    y1 = y[s1]
    w1 = weights[s1]
    pos1 = np.sum(w1[y1 == 1]) * pos_weight
    neg1 = np.sum(w1[y1 == 0])
    p_pos1 = pos1 / total_pos if total_pos > 0 and pos1 > 0 else 1e-10
    p_neg1 = neg1 / total_neg if total_neg > 0 and neg1 > 0 else 1e-10

    iv = (p_neg0 - p_pos0) * (np.log(p_neg0) - np.log(p_pos0))
    iv += (p_neg1 - p_pos1) * (np.log(p_neg1) - np.log(p_pos1))
    return iv


def information_value(y, subsets, weights=None, pos_weight=1):
    """Calculate information value for a split.

    Information value (Weight of Evidence) measures the strength of a feature
    in separating positive and negative classes.

    Args:
        y (pd.Series or np.array): Binary target values.
        subsets (list): List of boolean masks defining the split.
        weights (pd.Series or np.array, optional): Sample weights. Defaults to equal weights.
        pos_weight (float): Weight for positive class. Defaults to 1.

    Returns:
        float: Information value (higher absolute value indicates stronger separation).
    """
    if weights is None:
        weights = np.ones(len(y), dtype="float64")

    if isinstance(y, pd.Series):
        y = y.to_numpy(dtype="int64")
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy(dtype="float64")
    for i, s in enumerate(subsets):
        if isinstance(s, pd.Series):
            subsets[i] = s.to_numpy(dtype="bool")

    return _information_value_njit(y, subsets[0], subsets[1], weights, pos_weight)


@njit
def _get_precision_njit(y_pred, y_true, weights):
    """Calculate precision using Numba-optimized computation.

    Args:
        y_pred (np.array): Predicted binary values.
        y_true (np.array): True binary values.
        weights (np.array): Sample weights.

    Returns:
        float: Precision value.
    """
    return np.sum(y_pred * y_true * weights) / np.sum(y_pred * weights)


def get_precision(y_pred, y_true, weights=1):
    """Calculate precision for binary classification.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        y_pred (pd.Series or np.array): Predicted binary values.
        y_true (pd.Series or np.array): True binary values.
        weights (pd.Series, np.array, or scalar): Sample weights. Defaults to 1.

    Returns:
        float: Precision value between 0 and 1.
    """
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy(dtype="int64")
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy(dtype="int64")
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy(dtype="float64")
    return _get_precision_njit(y_pred, y_true, weights)


@njit
def _get_recall_njit(y_pred, y_true, weights, total_pos):
    """Calculate recall using Numba-optimized computation.

    Args:
        y_pred (np.array): Predicted binary values.
        y_true (np.array): True binary values.
        weights (np.array): Sample weights.
        total_pos (float): Total positive samples for recall calculation.

    Returns:
        float: Recall value.
    """
    return np.sum(y_pred * y_true * weights) / total_pos


def get_recall(y_pred, y_true, weights=1, total_pos=None):
    """Calculate recall for binary classification.

    Recall = True Positives / (True Positives + False Negatives)

    Args:
        y_pred (pd.Series or np.array): Predicted binary values.
        y_true (pd.Series or np.array): True binary values.
        weights (pd.Series, np.array, or scalar): Sample weights. Defaults to 1.
        total_pos (float, optional): Total positive samples. Calculated from data if None.

    Returns:
        float: Recall value between 0 and 1.
    """
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy(dtype="int64")
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy(dtype="int64")
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy(dtype="float64")
    if total_pos is None:
        total_pos = np.sum(y_true * weights)
    return _get_recall_njit(y_pred, y_true, weights, total_pos)


def get_f_score(precision, recall, beta=1, knot=1, factor=1):
    """Calculate F-score with optional precision adjustment.

    F-score is the harmonic mean of precision and recall, with optional
    beta weighting and precision scaling for values above a threshold.

    Args:
        precision (float): Precision value.
        recall (float): Recall value.
        beta (float): Beta parameter for F-beta score. Defaults to 1 (F1-score).
        knot (float): Threshold for precision scaling. Defaults to 1.
        factor (float): Scaling factor for precision above knot. Defaults to 1.

    Returns:
        float: F-score value (0 if precision or recall is 0).
    """
    if precision == 0 or recall == 0:
        return 0
    if precision > knot:
        precision = knot + (precision - knot) * factor

    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
