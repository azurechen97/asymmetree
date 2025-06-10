from functools import singledispatch
import operator

import numpy as np
import pandas as pd

@singledispatch
def isin(x, value):
    return x in value

@isin.register(np.ndarray)
def _isin_ndarray(x, value):
    return np.isin(x, value)

@isin.register(pd.Series)
def _isin_series(x, value):
    return x.isin(value)

@singledispatch
def notin(x, value):
    if x is None:
        return False
    return x not in value

@notin.register(np.ndarray)
def _notin_ndarray(x, value):
    return (~np.isin(x, value))&(~np.isnan(x))

@notin.register(pd.Series)
def _notin_series(x, value):
    return (~x.isin(value))&(~x.isna())

operator_map = {
    "in": isin,
    "not in": notin,
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

operator_reverse_map = {
    "in": "not in",
    "not in": "in",
    "==": "!=",
    "!=": "==",
    ">": "<=",
    "<=": ">",
    "<": ">=",
    ">=": "<",
    "=": "<>",
    "<>": "=",
}