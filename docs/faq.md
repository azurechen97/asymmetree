# Frequently Asked Questions

This document addresses common questions about using AsymmeTree.

## General Questions

### What makes AsymmeTree different from other decision tree algorithms?

AsymmeTree is specifically designed for highly imbalanced datasets where traditional decision trees fail. Key differences:

1. **F-score optimization**: Optimizes for F-score instead of information gain or Gini impurity
2. **Asymmetric node design**: Left child = "positive node" (higher positive ratio), right child = "neutral node"
3. **Interactive mode**: Allows domain experts to guide tree construction
4. **Business-ready export**: Direct SQL export for production deployment
5. **Precision-recall balance**: Built-in stopping criteria based on precision and recall thresholds

### When should I use AsymmeTree vs traditional decision trees?

Use AsymmeTree when:
- Your positive class is < 5% of the dataset
- Precision is more important than overall accuracy
- You need interpretable business rules
- Domain expertise can improve model performance
- You need to deploy rules in a database/SQL environment

Use traditional decision trees when:
- Classes are relatively balanced (e.g., 20-80% split)
- Overall accuracy is the primary metric
- You need a simple, automated solution

### What types of problems is AsymmeTree best suited for?

AsymmeTree excels in:
- **Fraud detection**: Credit card fraud, insurance fraud
- **Medical diagnosis**: Rare disease prediction, adverse event detection
- **Quality control**: Manufacturing defect detection
- **Marketing**: High-value customer identification, conversion prediction
- **Risk management**: Default prediction, anomaly detection

## Installation and Setup

### Why do I get "ImportError: No module named 'asymmetree'"?

This usually means the package isn't properly installed. Try:

1. Check if you're in the correct Python environment
2. Reinstall with `pip install -e .` from the project directory
3. Verify installation with `python -c "import asymmetree; print('Success')"`

### Why is training so slow?

Several factors can affect performance:

1. **Large categorical features**: Use `max_cat_unique` to limit unique values
2. **High `num_bin` values**: Reduce for faster training on large datasets
3. **Missing numba**: Install numba for optimized performance: `pip install numba`
4. **Deep trees**: Reduce `max_depth` for faster training

## Configuration and Parameters

### How do I choose the right parameters for my dataset?

Start with these guidelines:

**For high imbalance (< 1% positive):**
```python
AsymmeTree(
    max_depth=4,
    sorted_by='f_score',
    node_min_recall=0.01,
    leaf_min_precision=0.05
)
```

**For moderate imbalance (1-5% positive):**
```python
AsymmeTree(
    max_depth=5,
    sorted_by='f_score',
    node_min_recall=0.03,
    leaf_min_precision=0.10
)
```

### What's the difference between the splitting criteria?

- **`f_score`** (recommended): Optimizes F1-score of positive child node. Best for imbalanced data
- **`ig`**: Information gain. Traditional approach, good for balanced data
- **`igr`**: Information gain ratio. Handles categorical features better than IG
- **`iv`**: Information value. Good for feature selection in highly imbalanced data

### How do I handle categorical features with many unique values?

Use these strategies:

1. **Limit unique values**: Set `max_cat_unique=20` or lower
2. **Group rare categories**: Combine categories with low frequency
3. **Feature engineering**: Create higher-level categorical groupings
4. **Minimum recall threshold**: Use `cat_value_min_recall=0.01` to exclude rare values

### What do the stopping criteria parameters mean?

- **`node_max_precision`**: Stop splitting if node precision exceeds this (you've found pure positive regions)
- **`node_min_recall`**: Stop splitting if node recall falls below this (too few positives remain)
- **`leaf_min_precision`**: Minimum precision required to predict positive for a leaf
- **`max_depth`**: Maximum tree depth to prevent overfitting

## Training and Usage

### Should I use automatic or interactive mode?

**Use automatic mode (`auto=True`) when:**
- You have no domain expertise about the features
- You want a quick baseline model
- The dataset is well-understood and clean

**Use interactive mode (`auto=False`) when:**
- You have business knowledge about which features are important
- You want to incorporate domain constraints
- You need to understand the decision-making process
- You're building models for regulated industries

### How do I handle missing values?

AsymmeTree provides several options:

1. **Ignore nulls** (`ignore_null=True`, default): Treats nulls as missing data
2. **Include nulls** (`ignore_null=False`): Treats null as a separate category
3. **Preprocessing**: Fill nulls before training with appropriate values

### What's the best way to validate AsymmeTree models?

Use stratified cross-validation to maintain class balance:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # Train and validate here
```

Focus on precision, recall, and F1-score rather than accuracy for imbalanced data.

## Interactive Mode

### What commands can I use in interactive mode?

During interactive training:
- **Number (1-N)**: Select feature or condition by number
- **'c'**: Continue with automatic mode from current node
- **'q'**: Quit and stop building the tree
- **'s'**: Skip current node (make it a leaf)

### How do I know which features to choose?

The system shows features ranked by potential improvement. Consider:

1. **Business logic**: Choose features that make business sense
2. **F-score improvement**: Higher F-score indicates better separation
3. **Domain knowledge**: Features you know are important predictors
4. **Interpretability**: Features stakeholders will understand

### Can I modify the tree after training?

Yes! AsymmeTree provides several post-training modifications:

```python
# Toggle predictions for specific nodes
tree.toggle_prediction(id=5)

# Relabel based on new thresholds
tree.relabel(min_precision=0.15)

# Prune branches
tree.prune()

# Clear children from a node
tree.clear_children(id=3)
```

## Performance and Metrics

### Why is my precision/recall low?

Common causes and solutions:

**Low Precision:**
- Tree is too permissive. Increase `leaf_min_precision`
- Reduce `max_depth` to avoid overfitting
- Use stricter stopping criteria

**Low Recall:**
- Tree is too restrictive. Decrease `leaf_min_precision`
- Increase `max_depth` to capture more patterns
- Lower `node_min_recall` threshold

### How do I interpret the custom metrics?

Custom metrics help evaluate business impact:

```python
def revenue_impact(data):
    return data['revenue'].sum()

tree = AsymmeTree(
    extra_metrics={'revenue': revenue_impact},
    extra_metrics_data=revenue_data
)
```

These metrics show business value captured by each node, helping prioritize splits.

### What's a good F-score for imbalanced data?

F-score interpretation depends on your dataset:
- **F-score > 0.3**: Generally good for highly imbalanced data
- **F-score > 0.5**: Excellent performance
- **F-score < 0.1**: May need parameter tuning or more data

Compare against baseline precision (overall positive rate) to gauge improvement.

## Production Deployment

### How do I deploy AsymmeTree rules in production?

AsymmeTree provides SQL export for database deployment:

```python
sql_rules = tree.to_sql()
# Use in database queries:
# SELECT * FROM customers WHERE {sql_rules}
```

For application deployment:
```python
# Save model
tree.save('model.json')

# Load in production
production_tree = AsymmeTree()
production_tree.load('model.json')
predictions = production_tree.predict(new_data)
```

### How do I handle concept drift?

Monitor model performance and retrain when:
1. Precision/recall drops significantly
2. Business rules change
3. New data patterns emerge

Implement automated retraining pipelines and A/B testing for model updates.

### Can I use AsymmeTree with other ML models?

Yes! Common patterns:

**As a filter**: Use AsymmeTree to pre-screen high-precision cases, then apply complex models to remaining data

**Feature engineering**: Use tree rules as features for other models

**Ensemble**: Combine AsymmeTree predictions with other algorithms

## Troubleshooting

### The tree isn't splitting/stops early

Check these common issues:

1. **Insufficient positive samples**: Lower `node_min_recall`
2. **Too strict precision requirements**: Lower `leaf_min_precision`
3. **Categorical features with too many values**: Increase `max_cat_unique`
4. **All samples in one branch**: Check feature distributions and scaling

### Interactive mode is confusing

Tips for better interactive experience:

1. Set `verbose=True` for detailed output
2. Start with automatic mode to understand the data
3. Prepare by analyzing feature importance beforehand
4. Practice with sample datasets first

### Memory issues with large datasets

Solutions for memory problems:

1. **Reduce num_bin**: Use fewer bins for numerical features
2. **Limit categorical values**: Set lower `max_cat_unique`
3. **Sample data**: Train on a representative sample
4. **Feature selection**: Remove irrelevant features before training

### Getting unexpected predictions

Debug steps:

1. Print the tree structure: `tree.print(show_metrics=True)`
2. Check individual predictions: `tree._predict_mask(X, node)`
3. Verify data preprocessing matches training
4. Ensure feature names match exactly

## Best Practices

### How can I improve model interpretability?

1. **Limit tree depth**: Keep `max_depth <= 6` for human comprehension
2. **Use meaningful feature names**: Avoid technical abbreviations
3. **Document business logic**: Explain why certain splits make sense
4. **Visualize rules**: Use `tree.print()` to show decision paths

### What's the recommended workflow?

1. **Explore data**: Understand class balance and feature distributions
2. **Start simple**: Use default parameters with automatic mode
3. **Add domain knowledge**: Switch to interactive mode for important decisions
4. **Validate thoroughly**: Use cross-validation with appropriate metrics
5. **Test in production**: A/B test against existing rules
6. **Monitor and maintain**: Track performance and retrain as needed

### How do I explain AsymmeTree to stakeholders?

Focus on business benefits:
- **Interpretable rules**: Easy to understand "if-then" logic
- **SQL deployment**: Integrate directly with existing systems  
- **Domain expertise**: Incorporates business knowledge
- **Precision focus**: Reduces false positives in critical applications
- **Proven approach**: Successful in fraud detection and medical diagnosis

Need more help? Check the [User Guide](user-guide.md) for detailed examples or the [API Reference](api-reference.md) for technical details. 