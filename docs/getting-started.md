# Getting Started

This guide will help you get up and running with AsymmeTree quickly.

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/azurechen97/asymmetree.git
cd asymmetree
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install AsymmeTree in development mode:
```bash
pip install -e .
```

### Dependencies

AsymmeTree has minimal dependencies:
- `numba>=0.61.2` - For performance optimization
- `numpy>=2.2.6` - For numerical computations
- `pandas>=2.3.0` - For data manipulation

## Quick Start

### Basic Usage

Here's a simple example to get you started:

```python
import pandas as pd
import numpy as np
from asymmetree import AsymmeTree

# Create sample imbalanced dataset
np.random.seed(42)
n_samples = 10000
n_features = 5

# Generate features
X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                 columns=[f'feature_{i}' for i in range(n_features)])

# Create imbalanced target (2% positive class)
y = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

# Initialize AsymmeTree
tree = AsymmeTree(
    max_depth=5,
    sorted_by='f_score',  # Optimized for imbalanced data
    node_min_recall=0.05,
    leaf_min_precision=0.15
)

# Fit the model
tree.fit(X, y, auto=True)

# Make predictions
predictions = tree.predict(X)

# View performance metrics
tree.performance()

# Print the tree structure
tree.print(show_metrics=True)
```

### Understanding the Output

After training, AsymmeTree provides:

1. **Performance Metrics**: Precision, recall, F-score optimized for imbalanced data
2. **Tree Structure**: Hierarchical rules with business-interpretable conditions
3. **SQL Export**: Ready-to-deploy business rules

### Next Steps

- Learn about [Interactive Mode](user-guide.md#interactive-mode) for incorporating domain expertise
- Explore [Advanced Configuration](user-guide.md#advanced-configuration) options
- Check out [Real-world Examples](examples.md) for your use case
- Review the [Complete API Reference](api-reference.md)

## Troubleshooting

### Common Issues

**ImportError: No module named 'asymmetree'**
- Make sure you've installed the package with `pip install -e .`
- Check that you're using the correct Python environment

**Performance Issues**
- Ensure numba is properly installed for optimized performance
- Consider reducing `max_depth` for very large datasets

**Memory Issues**
- Use smaller `num_bin` values for numerical features
- Consider data sampling for extremely large datasets

Need more help? Check the [FAQ](faq.md) or review the detailed [User Guide](user-guide.md). 