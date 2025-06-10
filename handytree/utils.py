import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype


def get_precision(y, weights, total_pos):
    pos_count = (y * weights).sum()
    precision = pos_count / weights.sum() if weights.sum() > 0 else 0
    return precision


def get_recall(y, weights, total_pos):
    pos_count = (y * weights).sum()
    recall = pos_count / total_pos if total_pos > 0 else 0
    return recall
