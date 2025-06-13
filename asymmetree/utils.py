"""
Utility functions for AsymmeTree decision tree operations.

This module provides operator handling functions and mappings for split conditions,
supporting both pandas Series and numpy arrays with polymorphic dispatch.
"""

from functools import singledispatch
import operator

import numpy as np
import pandas as pd


@singledispatch
def isin(x, value):
    """Check if values are contained in a collection (base implementation).

    Args:
        x: Value or collection to check.
        value: Collection to check membership against.

    Returns:
        bool or array: Boolean result of membership test.
    """
    return x in value


@isin.register(np.ndarray)
def _isin_ndarray(x, value):
    """Check if numpy array values are contained in a collection.

    Args:
        x (np.ndarray): Array of values to check.
        value: Collection to check membership against.

    Returns:
        np.ndarray: Boolean array indicating membership.
    """
    return np.isin(x, value)


@isin.register(pd.Series)
def _isin_series(x, value):
    """Check if pandas Series values are contained in a collection.

    Args:
        x (pd.Series): Series of values to check.
        value: Collection to check membership against.

    Returns:
        pd.Series: Boolean Series indicating membership.
    """
    return x.isin(value)


@singledispatch
def notin(x, value):
    """Check if values are NOT contained in a collection (base implementation).

    Args:
        x: Value or collection to check.
        value: Collection to check non-membership against.

    Returns:
        bool or array: Boolean result of non-membership test.
    """
    if x is None:
        return False
    return x not in value


@notin.register(np.ndarray)
def _notin_ndarray(x, value):
    """Check if numpy array values are NOT contained in a collection.

    Args:
        x (np.ndarray): Array of values to check.
        value: Collection to check non-membership against.

    Returns:
        np.ndarray: Boolean array indicating non-membership (excluding NaN values).
    """
    return (~np.isin(x, value)) & (~np.isnan(x))


@notin.register(pd.Series)
def _notin_series(x, value):
    """Check if pandas Series values are NOT contained in a collection.

    Args:
        x (pd.Series): Series of values to check.
        value: Collection to check non-membership against.

    Returns:
        pd.Series: Boolean Series indicating non-membership (excluding NA values).
    """
    return (~x.isin(value)) & (~x.isna())


# Operator mapping for split conditions
# Maps string operators to their corresponding functions
operator_map = {
    "in": isin,  # Membership test
    "not in": notin,  # Non-membership test
    "==": operator.eq,  # Equality
    "=": operator.eq,  # Equality (alternative syntax)
    "!=": operator.ne,  # Inequality
    "<>": operator.ne,  # Inequality (alternative syntax)
    ">": operator.gt,  # Greater than
    ">=": operator.ge,  # Greater than or equal
    "<": operator.lt,  # Less than
    "<=": operator.le,  # Less than or equal
}

# Reverse operator mapping for negating conditions
# Maps operators to their logical opposites
operator_reverse_map = {
    "in": "not in",  # in -> not in
    "not in": "in",  # not in -> in
    "==": "!=",  # equals -> not equals
    "!=": "==",  # not equals -> equals
    ">": "<=",  # greater than -> less than or equal
    "<=": ">",  # less than or equal -> greater than
    "<": ">=",  # less than -> greater than or equal
    ">=": "<",  # greater than or equal -> less than
    "=": "<>",  # equals (alt) -> not equals (alt)
    "<>": "=",  # not equals (alt) -> equals (alt)
}
