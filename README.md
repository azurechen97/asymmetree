# AsymmeTree: Interactive Asymmetric Decision Trees for Business-Ready Imbalanced Classification

**Authored by Aoxue Chen**

AsymmeTree is an interactive decision tree classifier specifically designed for highly imbalanced datasets. Unlike traditional decision trees that optimize for node purity, AsymmeTree focuses on maximizing precision while capturing sufficient recall, making it ideal for fraud detection, anomaly detection, and other rare event prediction tasks.

## 🚀 Key Features

- **Imbalanced-Optimized Algorithm**: Novel splitting strategy where left child = "positive node" (higher positive ratio), right child = "neutral node" (lower positive ratio)
- **F-Score Based Optimization**: Optimizes splits using F-score of the positive node for better precision-recall balance
- **Multiple Splitting Criteria**: Supports both imbalanced-focused (f_score) and traditional purity-based approaches (information gain, information gain ratio, information value)
- **Hybrid Interaction Modes**: Interactive, automatic, or hybrid modes allowing domain expertise integration
- **Business-Aligned Design**: Built with real-world business constraints and interpretability in mind
- **Comprehensive Feature Support**: Handles categorical and numerical features with custom operator constraints
- **Programmable Metrics**: Extensible metrics system for custom evaluation criteria

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/azurechen97/asymmetree.git
cd asymmetree

# Install dependencies
pip install -r requirements.txt

# Install asymmetree
pip install -e .
```

## 🔧 Quick Start

### Basic Usage

```python
import pandas as pd
from asymmetree import AsymmeTree

# Load your imbalanced dataset
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')['target']

# Initialize AsymmeTree
tree = AsymmeTree(
    max_depth=5,
    sorted_by='f_score',  # Optimized for imbalanced data
    node_min_recall=0.05,
    leaf_min_precision=0.15
)

# Fit the model
tree.fit(X, y, auto=True)  # Automatic mode

# Make predictions
predictions = tree.predict(X)

# View performance
tree.performance()

# Export to SQL for business rules
sql_rules = tree.to_sql()
print(sql_rules)
```

### Interactive Mode

```python
# Interactive tree building with domain expertise
tree = AsymmeTree(max_depth=5)
tree.import_data(X, y)

# Start interactive splitting
tree.split(id=0)  # Split root node interactively

# Continue building specific branches
tree.continue_fit(id=2)  # Continue from node 2

# Quick split with custom condition
tree.quick_split(id=4, sql="age >= 25", overwrite=True)
```

### Advanced Configuration

```python
# Configure for specific use case
tree = AsymmeTree(
    max_depth=7,
    max_cat_unique=20,
    cat_value_min_recall=0.005,  # Minimum recall for categorical values
    sorted_by='f_score',
    node_max_precision=0.3,
    node_min_recall=0.05,
    cat_features=['category', 'region'],
    lt_only_features=['age'],  # Age can only use < operator
    gt_only_features=['income'],  # Income can only use > operator
    pinned_features=['risk_score'],  # Prioritize this feature
    beta=1,  # F-beta score parameter
    knot=1,  # Precision scaling threshold
    factor=1,  # Scaling factor for precision above knot
    verbose=True
)
```

## 🎯 Use Cases

AsymmeTree excels in scenarios with significant class imbalance (< 5% positive class):

### Fraud Detection
```python
# Fraud detection with high precision requirements
fraud_tree = AsymmeTree(
    sorted_by='f_score',
    node_max_precision=0.2,  # Stop splitting at 20% precision
    leaf_min_precision=0.4,  # Require 40% precision for positive prediction
    pos_weight=10  # Weight positive cases more heavily
)
```

### Medical Diagnosis
```python
# Medical diagnosis with interpretable rules
# Patient outcome data aligned with feature matrix
patient_data = pd.DataFrame({
    'cost': patient_costs,
    'severity_score': severity_scores
}, index=X.index)

medical_tree = AsymmeTree(
    max_depth=4,  # Keep rules simple
    show_metrics=True,
    extra_metrics={
        'avg_cost': lambda x: x['cost'].mean(),
        'avg_severity': lambda x: x['severity_score'].mean()
    },
    extra_metrics_data=patient_data,
    verbose=True
)
```

### Quality Control
```python
# Manufacturing defect detection
qc_tree = AsymmeTree(
    sorted_by='igr',  # Use information gain ratio
    cat_features=['machine_id', 'shift', 'operator'],
    num_bin=20,  # Fine-grained numerical splits
    ignore_null=False  # Include null as category
)
```

## 📊 Performance Metrics

AsymmeTree provides comprehensive metrics for imbalanced classification:

```python
# Built-in metrics
metrics = tree.metrics()
print(f"Precision: {metrics['Precision']:.2%}")
print(f"Recall: {metrics['Recall']:.2%}")
print(f"Positives Captured: {metrics['Positives']:.0f}")

# Custom metrics (requires extra_metrics_data with same index as X)
def business_value(data):
    return data['revenue'].sum() - data['cost'].sum()

def avg_transaction_size(data):
    return data['amount'].mean()

# extra_metrics_data must have same index as X
extra_data = pd.DataFrame({
    'revenue': [...],  # Revenue values for each sample
    'cost': [...],     # Cost values for each sample  
    'amount': [...]    # Transaction amounts
}, index=X.index)

tree = AsymmeTree(
    extra_metrics={
        'business_value': business_value,
        'avg_transaction': avg_transaction_size
    },
    extra_metrics_data=extra_data
)
```

## 🔍 Tree Visualization and Export

```python
# Print tree structure
tree.print(show_metrics=True)

# Export to SQL for deployment
sql_where_clause = tree.to_sql()

# Save/load models
tree.save('fraud_model.json')
new_tree = AsymmeTree()
new_tree.load('fraud_model.json')

# Export to dictionary
tree_dict = tree.to_dict()

# Toggle leaf node predictions manually
tree.toggle_prediction(id=5)  # Toggle prediction for node 5

# Relabel nodes based on thresholds
tree.relabel(min_precision=0.2, min_recall=0.05)

# Prune tree to remove redundant splits
tree.prune()

# Clear children of a specific node
tree.clear_children(id=3)
```

## 🧠 Algorithm Details

### Imbalanced Splitting Strategy

1. **Positive Node Assignment**: Left child always gets higher positive ratio samples
2. **Neutral Node Assignment**: Right child gets lower positive ratio samples  
3. **F-Score Optimization**: Maximizes F-score of positive node for better precision-recall balance
4. **Adaptive Thresholds**: Uses configurable precision/recall thresholds to control splitting

### Supported Split Criteria

- `f_score`: F-score based (default for imbalanced data)
- `ig`: Information Gain
- `igr`: Information Gain Ratio  
- `iv`: Information Value (Weight of Evidence)

## 🛠️ API Reference

### Core Classes

- `AsymmeTree`: Main classifier class
- `Node`: Individual tree node with split conditions and metrics

### Key Methods

#### Model Training
- `fit(X, y, auto=False)`: Train the model
- `import_data(X, y, ...)`: Import training data with configuration
- `continue_fit(id, auto=False)`: Continue building from specific node

#### Interactive Splitting
- `split(id, auto=False)`: Interactive node splitting
- `quick_split(id, sql, overwrite=False)`: Quick split with SQL condition

#### Prediction and Evaluation
- `predict(X)`: Generate predictions
- `performance()`: Display model metrics
- `metrics()`: Return metrics dictionary

#### Tree Manipulation
- `toggle_prediction(id)`: Toggle leaf node prediction
- `relabel(min_precision, min_recall)`: Relabel nodes based on thresholds
- `prune()`: Remove redundant splits
- `clear_children(id)`: Remove all children of a node

#### Model Export/Import
- `to_sql()`: Export as SQL WHERE clause
- `to_dict()`: Export as dictionary
- `to_json()`: Export as JSON string
- `save(file_path)`: Save model to file
- `load(file_path)`: Load model from file
- `from_dict(nodes)`: Load from dictionary
- `from_json(json_str)`: Load from JSON string

#### Visualization
- `print(show_metrics=False)`: Display tree structure

### Configuration Parameters

- `max_depth` (int): Maximum tree depth (default: 5)
- `max_cat_unique` (int): Maximum unique values for categorical features (default: 50)
- `cat_value_min_recall` (float): Minimum recall threshold for categorical values (default: 0.005)
- `num_bin` (int): Number of bins for numerical discretization (default: 25)
- `node_max_precision` (float): Maximum precision threshold for splitting (default: 0.3)
- `node_min_recall` (float): Minimum recall threshold for nodes (default: 0.05)
- `leaf_min_precision` (float): Minimum precision for positive leaf prediction (default: 0.15)
- `feature_shown_num` (int): Number of features shown in interactive mode (default: 5)
- `condition_shown_num` (int): Number of conditions shown in interactive mode (default: 5)
- `sorted_by` (str): Split criterion - 'f_score', 'ig', 'igr', 'iv' (default: 'f_score')
- `pos_weight` (float): Weight for positive class in calculations (default: 1)
- `beta` (float): Beta parameter for F-beta score (default: 1)
- `knot` (float): Threshold for precision scaling (default: 1)
- `factor` (float): Scaling factor for precision above knot (default: 1)
- `ignore_null` (bool): Whether to ignore null values (default: True)
- `show_metrics` (bool): Whether to show metrics in tree display (default: False)
- `verbose` (bool): Whether to print verbose output (default: False)

### Feature Constraints

- `cat_features` (list): Categorical feature names
- `lt_only_features` (list): Features restricted to '<' operators  
- `gt_only_features` (list): Features restricted to '>' operators
- `pinned_features` (list): Features to prioritize in splitting

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use AsymmeTree in your research, please cite:

```bibtex
@software{asymmetree,
  author = {Aoxue Chen},
  title = {AsymmeTree: Interactive Asymmetric Decision Trees for Business-Ready Imbalanced Classification},
  url = {https://github.com/azurechen97/asymmetree},
  year = {2025}
}
```

## 🏆 Acknowledgments

- Thanks to the scikit-learn team for inspiration
- Built with NumPy, Pandas, and optimized for performance