# AsymmeTree Documentation

Welcome to the AsymmeTree documentation! AsymmeTree is an interactive decision tree classifier specifically designed for highly imbalanced datasets, making it ideal for fraud detection, anomaly detection, and other rare event prediction tasks.

## Table of Contents

- [Getting Started](getting-started.md) - Installation and quick start guide
- [User Guide](user-guide.md) - Comprehensive usage guide with examples
- [API Reference](api-reference.md) - Detailed API documentation
- [Tutorials](tutorials.md) - Step-by-step tutorials for common use cases
- [Examples](examples.md) - Real-world examples and use cases
- [FAQ](faq.md) - Frequently asked questions

## Key Features

- **Imbalanced-Optimized Algorithm**: Novel splitting strategy optimized for rare events
- **F-Score Based Optimization**: Maximizes precision while capturing sufficient recall
- **Interactive Mode**: Allows domain expertise integration during tree building
- **Business-Ready**: Exports to SQL for production deployment
- **Comprehensive Metrics**: Built-in evaluation metrics for imbalanced classification

## Quick Example

```python
import pandas as pd
from asymmetree import AsymmeTree

# Load your imbalanced dataset
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')['target']

# Initialize and train
tree = AsymmeTree(max_depth=5, sorted_by='f_score')
tree.fit(X, y, auto=True)

# Make predictions and view performance
predictions = tree.predict(X)
tree.performance()
```

## Use Cases

AsymmeTree excels in scenarios with significant class imbalance (< 5% positive class):
- Fraud detection
- Medical diagnosis
- Quality control
- Anomaly detection
- Risk assessment

## Getting Help

- Check the [FAQ](faq.md) for common questions
- Review the [Examples](examples.md) for practical use cases
- Consult the [API Reference](api-reference.md) for detailed parameter descriptions 