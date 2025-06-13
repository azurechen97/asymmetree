# API Reference

This document provides detailed information about AsymmeTree's classes and methods.

## AsymmeTree Class

### Constructor

```python
AsymmeTree(
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
    verbose=False
)
```

#### Parameters

**Tree Structure Parameters:**
- `max_depth` (int, default=5): Maximum depth of the decision tree
- `max_cat_unique` (int, default=50): Maximum unique values for categorical features
- `num_bin` (int, default=25): Number of bins for numerical feature discretization

**Stopping Criteria:**
- `node_max_precision` (float, default=0.3): Maximum precision threshold for node splitting
- `node_min_recall` (float, default=0.05): Minimum recall threshold for nodes
- `leaf_min_precision` (float, default=0.15): Minimum precision for positive leaf predictions
- `cat_value_min_recall` (float, default=0.005): Minimum recall threshold for categorical values

**Display Parameters:**
- `feature_shown_num` (int, default=5): Number of features shown in interactive mode
- `condition_shown_num` (int, default=5): Number of conditions shown in interactive mode
- `show_metrics` (bool, default=False): Whether to display metrics in tree visualization
- `verbose` (bool, default=False): Enable verbose output during training

**Optimization Parameters:**
- `sorted_by` (str, default="f_score"): Metric for sorting splits. Options:
  - `"f_score"`: F-score optimization (recommended for imbalanced data)
  - `"ig"`: Information gain
  - `"igr"`: Information gain ratio
  - `"iv"`: Information value
- `pos_weight` (float, default=1): Weight for positive class in calculations
- `beta` (float, default=1): Beta parameter for F-beta score
- `knot` (float, default=1): Threshold for precision scaling
- `factor` (float, default=1): Scaling factor for precision above knot
- `ignore_null` (bool, default=True): Whether to ignore null values in splits

## Core Methods

### Data Import and Training

#### import_data()
```python
import_data(
    X, y, 
    weights=None,
    cat_features=None,
    lt_only_features=None,
    gt_only_features=None,
    pinned_features=None,
    extra_metrics=None,
    extra_metrics_data=None,
    total_pos=None
)
```
Import and preprocess data for tree building.

**Parameters:**
- `X` (DataFrame): Feature matrix
- `y` (Series/array): Target variable (binary: 0/1)
- `weights` (Series/array, optional): Sample weights
- `cat_features` (list, optional): List of categorical feature names
- `lt_only_features` (list, optional): Features that can only use `<` operator
- `gt_only_features` (list, optional): Features that can only use `>` operator
- `pinned_features` (list, optional): Features to prioritize in splits
- `extra_metrics` (dict, optional): Custom metrics functions
- `extra_metrics_data` (DataFrame, optional): Data for custom metrics
- `total_pos` (int, optional): Total positive samples for validation

#### fit()
```python
fit(
    X=None, y=None,
    weights=None,
    cat_features=None,
    lt_only_features=None,
    gt_only_features=None,
    pinned_features=None,
    extra_metrics=None,
    extra_metrics_data=None,
    total_pos=None,
    auto=False
)
```
Train the AsymmeTree model.

**Parameters:**
- Same as `import_data()` plus:
- `auto` (bool, default=False): Whether to build tree automatically without interaction

**Returns:**
- `self`: Returns the fitted AsymmeTree instance

### Interactive Tree Building

#### split()
```python
split(id=None, auto=False)
```
Interactively split a node.

**Parameters:**
- `id` (int, optional): Node ID to split. If None, splits root node
- `auto` (bool, default=False): Whether to split automatically

#### quick_split()
```python
quick_split(id=None, sql=None, overwrite=False)
```
Split a node using SQL-like condition.

**Parameters:**
- `id` (int, optional): Node ID to split
- `sql` (str): SQL-like condition (e.g., "age >= 25")
- `overwrite` (bool, default=False): Whether to overwrite existing splits

**Example:**
```python
tree.quick_split(id=2, sql="income > 50000", overwrite=True)
```

#### continue_fit()
```python
continue_fit(id, auto=False)
```
Continue building tree from specific node.

**Parameters:**
- `id` (int): Node ID to continue from
- `auto` (bool, default=False): Whether to continue automatically

### Prediction and Evaluation

#### predict()
```python
predict(X)
```
Make predictions on new data.

**Parameters:**
- `X` (DataFrame): Feature matrix for prediction

**Returns:**
- `array`: Binary predictions (0/1)

#### metrics()
```python
metrics(
    pred=None, true=None,
    weights=None,
    extra_metrics=None,
    extra_metrics_data=None,
    total_pos=None
)
```
Calculate detailed performance metrics.

**Returns:**
- `dict`: Dictionary containing metrics like precision, recall, F-score

#### performance()
```python
performance(
    pred=None, true=None,
    weights=None,
    extra_metrics=None,
    extra_metrics_data=None,
    total_pos=None
)
```
Print performance summary.

### Tree Manipulation

#### toggle_prediction()
```python
toggle_prediction(id=None)
```
Toggle prediction (0↔1) for a leaf node.

**Parameters:**
- `id` (int, optional): Node ID to toggle

#### relabel()
```python
relabel(min_precision=None, min_recall=None)
```
Relabel leaf nodes based on precision/recall thresholds.

**Parameters:**
- `min_precision` (float, optional): Minimum precision for positive prediction
- `min_recall` (float, optional): Minimum recall threshold

#### prune()
```python
prune(node=None)
```
Prune tree branches.

**Parameters:**
- `node` (Node, optional): Node to prune from. If None, prunes from root

#### clear_children()
```python
clear_children(id=None)
```
Remove all children from a node, making it a leaf.

**Parameters:**
- `id` (int, optional): Node ID to clear children from

### Visualization and Export

#### print()
```python
print(node=None, current_node=None, show_metrics=False, depth=0)
```
Print tree structure.

**Parameters:**
- `node` (Node, optional): Starting node for printing
- `current_node` (Node, optional): Current node to highlight
- `show_metrics` (bool, default=False): Whether to show node metrics
- `depth` (int, default=0): Starting depth for display

#### to_sql()
```python
to_sql(node=None)
```
Export tree rules as SQL WHERE clause.

**Parameters:**
- `node` (Node, optional): Starting node for export

**Returns:**
- `str`: SQL WHERE clause representing tree rules

**Example Output:**
```sql
(age >= 25 AND income > 50000) OR (risk_score >= 0.8)
```

### Serialization

#### save()
```python
save(file_path)
```
Save tree to JSON file.

**Parameters:**
- `file_path` (str): Path to save file

#### load()
```python
load(file_path)
```
Load tree from JSON file.

**Parameters:**
- `file_path` (str): Path to load file

#### to_dict()
```python
to_dict()
```
Convert tree to dictionary representation.

**Returns:**
- `dict`: Dictionary representation of tree

#### to_json()
```python
to_json()
```
Convert tree to JSON string.

**Returns:**
- `str`: JSON representation of tree

## Node Class

### Attributes

- `id` (int): Unique node identifier
- `depth` (int): Node depth in tree (root = 0)
- `is_leaf` (bool): Whether node is a leaf
- `prediction` (int): Prediction for leaf nodes (0 or 1)
- `split_feature` (str): Feature used for splitting
- `operator` (str): Split operator (`<`, `>=`, `in`, `not in`)
- `split_value`: Split threshold or categorical values
- `parent` (Node): Parent node reference
- `children` (list): List of child nodes
- `metrics` (dict): Node performance metrics

### Methods

#### to_dict()
```python
to_dict()
```
Convert node to dictionary representation.

#### to_json()
```python
to_json()
```
Convert node to JSON string.

## Utility Functions

See `asymmetree.utils` and `asymmetree.metrics` modules for additional utility functions used internally by AsymmeTree. 