import numpy as np
import pandas as pd

from numba import njit


@njit
def _entropy(y, weights, pos_weight):
    p = np.sum(y * weights) / np.sum(weights)
    entropy = -(
        pos_weight * p * np.log2(p if p > 0 else 1)
        + (1 - p) * np.log2(1 - p if 1 - p > 0 else 1)
    )
    return entropy


@njit
def _information_gain_njit(y, s0, s1, weights, pos_weight):
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
    n = np.sum(weights)

    w0 = weights[s0]
    n0 = np.sum(w0)

    w1 = weights[s1]
    n1 = np.sum(w1)

    ii = -n0 / n * (np.log2(n0) - np.log2(n)) if n0 > 0 else 0
    ii += -n1 / n * (np.log2(n1) - np.log2(n)) if n1 > 0 else 0
    return ii


def intrinsic_information(y, subsets, weights=None):
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
    if ii == 0:
        return 0
    return ig / ii


@njit
def _information_value_njit(y, s0, s1, weights, pos_weight):
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
    return np.sum(y_pred * y_true * weights) / np.sum(y_pred * weights)


def get_precision(y_pred, y_true, weights=1):
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy(dtype="int64")
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy(dtype="int64")
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy(dtype="float64")
    return _get_precision_njit(y_pred, y_true, weights)


@njit
def _get_recall_njit(y_pred, y_true, weights, total_pos):
    return np.sum(y_pred * y_true * weights) / total_pos


def get_recall(y_pred, y_true, weights=1, total_pos=None):
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
    if precision == 0 or recall == 0:
        return 0
    if precision > knot:
        precision = knot + (precision - knot) * factor

    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
