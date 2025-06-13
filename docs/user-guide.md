# User Guide

This comprehensive guide walks you through using AsymmeTree for imbalanced classification tasks.

## Overview

AsymmeTree is designed for binary classification problems with severe class imbalance (typically < 5% positive class). It addresses the limitations of traditional decision trees by:

1. **Optimizing for F-score** instead of purity measures
2. **Prioritizing precision** while maintaining adequate recall
3. **Enabling interactive tree building** with domain expertise
4. **Providing business-ready SQL export** for production deployment

## Basic Workflow

### 1. Data Preparation

```python
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')['target']

# Check class balance
print(f"Positive class ratio: {y.mean():.2%}")
```

### 2. Model Configuration

```python
# Basic configuration for imbalanced data
tree = AsymmeTree(
    max_depth=5,
    sorted_by='f_score',  # Use F-score optimization
    node_min_recall=0.05,  # Stop if recall falls below 5%
    leaf_min_precision=0.15  # Require 15% precision for positive prediction
)
```

### 3. Training

```python
# Automatic training
tree.fit(X, y, auto=True)

# Interactive training (recommended for domain expertise)
tree.fit(X, y, auto=False)
```

### 4. Evaluation and Deployment

```python
# Evaluate performance
predictions = tree.predict(X)
tree.performance()

# Export for production
sql_rules = tree.to_sql()
print(sql_rules)
```

## Interactive Mode

Interactive mode is AsymmeTree's key differentiator, allowing domain experts to guide tree construction.

### Starting Interactive Training

```python
tree = AsymmeTree(max_depth=4, verbose=True)
tree.fit(X, y, auto=False)
```

The system will prompt you to:
1. Choose features to split on
2. Select split conditions
3. Decide whether to continue splitting

### Interactive Commands

During interactive sessions, you can:

- **Enter feature number**: Choose from displayed options
- **Type 'c'**: Continue with automatic splitting
- **Type 'q'**: Quit and stop building
- **Type 's'**: Skip current node

### Example Interactive Session

```
Node 0 (Root): 1000 samples, 20 positives (2.0% precision)

Top Features for Splitting:
1. income (F-score: 0.45)
2. age (F-score: 0.38)
3. credit_score (F-score: 0.31)

Enter feature number or command: 1

Top Conditions for 'income':
1. income >= 50000 (Left: 8.5% precision, Right: 1.2% precision)
2. income >= 75000 (Left: 12.1% precision, Right: 1.5% precision)

Enter condition number: 2
```

## Advanced Configuration

### Feature Constraints

```python
tree = AsymmeTree(max_depth=5, sorted_by='f_score')

# Feature constraints are specified in fit() method
tree.fit(
    X, y,
    cat_features=['category', 'region'],  # Categorical features
    lt_only_features=['age'],  # Age can only use < operator
    gt_only_features=['income'],  # Income can only use >= operator
    pinned_features=['risk_score'],  # Prioritize risk_score in splits
    auto=True
)
```

### Custom Metrics

Add business-specific metrics to tree evaluation:

```python
# Define custom metrics
def revenue_impact(data):
    return data['transaction_value'].sum()

def avg_customer_age(data):
    return data['customer_age'].mean()

# Extra data aligned with X index
extra_data = pd.DataFrame({
    'transaction_value': [...],
    'customer_age': [...]
}, index=X.index)

tree = AsymmeTree(show_metrics=True)

# Custom metrics are specified in fit() method
tree.fit(
    X, y,
    extra_metrics={
        'total_revenue': revenue_impact,
        'avg_age': avg_customer_age
    },
    extra_metrics_data=extra_data,
    auto=True
)
```

### Optimization Parameters

Fine-tune the algorithm for your specific use case:

```python
tree = AsymmeTree(
    # F-beta score configuration
    beta=2,  # Emphasize recall over precision
    pos_weight=5,  # Weight positive samples more heavily
    
    # Precision scaling
    knot=0.1,  # Scale precision above 10%
    factor=2,  # Double the scaling factor
    
    # Categorical handling
    max_cat_unique=30,  # Maximum categories per feature
    cat_value_min_recall=0.01,  # Minimum recall for category inclusion
    
    # Numerical binning
    num_bin=50,  # Fine-grained numerical splits
)
```

## Splitting Strategies

### F-Score Optimization (Recommended)

```python
tree = AsymmeTree(sorted_by='f_score')
```
- Optimizes F1-score of the positive child node
- Best for imbalanced datasets
- Balances precision and recall

### Traditional Information-Based Methods

```python
# Information Gain
tree = AsymmeTree(sorted_by='ig')

# Information Gain Ratio
tree = AsymmeTree(sorted_by='igr')

# Information Value
tree = AsymmeTree(sorted_by='iv')
```

## Tree Manipulation

### Post-Training Adjustments

```python
# View tree structure
tree.print(show_metrics=True)

# Toggle prediction for specific nodes
tree.toggle_prediction(id=5)

# Relabel based on thresholds
tree.relabel(min_precision=0.2, min_recall=0.05)

# Prune branches
tree.prune()

# Clear children from a node
tree.clear_children(id=3)
```

### Quick Splitting

Use SQL-like syntax for rapid tree construction:

```python
# Split root node
tree.quick_split(id=0, sql="income >= 50000")

# Split with categorical condition
tree.quick_split(id=1, sql="region in ('North', 'East')")

# Overwrite existing splits
tree.quick_split(id=2, sql="age < 30", overwrite=True)
```

### Continue Building

Resume tree construction from any node:

```python
# Continue interactively from node 3
tree.continue_fit(id=3, auto=False)

# Continue automatically from node 5
tree.continue_fit(id=5, auto=True)
```

## Production Deployment

### SQL Export

Export tree rules for database deployment:

```python
sql_where_clause = tree.to_sql()
print(sql_where_clause)
```

Example output:
```sql
(income >= 50000 AND age < 40 AND region in ('North','East')) OR 
(credit_score >= 750 AND debt_ratio < 0.3)
```

Use in production queries:
```sql
SELECT customer_id, 
       CASE WHEN (income >= 50000 AND age < 40 AND region in ('North','East')) OR 
                 (credit_score >= 750 AND debt_ratio < 0.3)
            THEN 1 ELSE 0 END as high_risk_flag
FROM customers;
```

### Model Persistence

```python
# Save trained model
tree.save('fraud_detection_model.json')

# Load for prediction
new_tree = AsymmeTree()
new_tree.load('fraud_detection_model.json')
predictions = new_tree.predict(new_data)
```

## Best Practices

### 1. Data Preparation
- Ensure proper handling of missing values
- Consider feature scaling for distance-based splits
- Validate feature types (categorical vs numerical)

### 2. Parameter Tuning
- Start with default parameters
- Adjust `node_min_recall` based on business requirements
- Tune `leaf_min_precision` for deployment thresholds

### 3. Interactive Building
- Use domain expertise to guide initial splits
- Review metrics at each step
- Consider business constraints in split decisions

### 4. Validation
- Use cross-validation for parameter selection
- Test on holdout data before deployment
- Monitor performance in production

### 5. Interpretability
- Keep tree depth reasonable (≤ 6 levels)
- Use meaningful feature names
- Document business logic behind splits

## Troubleshooting

### Common Issues

**Low Precision/Recall**
- Adjust stopping criteria (`node_min_recall`, `leaf_min_precision`)
- Try different splitting strategies
- Increase `max_depth` if underfitting

**Tree Too Complex**
- Reduce `max_depth`
- Increase stopping thresholds
- Use pruning after training

**Poor Categorical Handling**
- Adjust `max_cat_unique`
- Increase `cat_value_min_recall`
- Consider feature engineering

**Interactive Mode Confusion**
- Set `verbose=True` for detailed output
- Review feature importance before starting
- Practice with automatic mode first

## Performance Tips

1. **Use numba optimization**: Ensure numba is installed for faster computation
2. **Reduce num_bin for large datasets**: Fewer bins = faster training
3. **Limit categorical unique values**: Use `max_cat_unique` to control complexity
4. **Consider data sampling**: For extremely large datasets, sample for tree building

## Next Steps

- Explore [Real-world Examples](examples.md) for your domain
- Check [FAQ](faq.md) for common questions
- Review [API Reference](api-reference.md) for advanced usage 