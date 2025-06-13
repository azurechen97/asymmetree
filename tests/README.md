# AsymmeTree Test Suite

This directory contains comprehensive unit tests for the AsymmeTree project using pytest.

## Test Files

### `test_node.py`
Tests for the `Node` class covering:
- Initialization with default and custom values
- String representation (`__repr__` and `__str__`)
- Dictionary and JSON serialization (`to_dict()`, `to_json()`)
- Parent-child relationships
- Edge cases and error handling

### `test_asymmetree.py`
Tests for the `AsymmeTree` class covering:
- Initialization with default and custom parameters
- Data import functionality with various configurations
- Prediction generation
- Metrics calculation
- Tree manipulation (toggle predictions, relabel nodes, clear children)
- Serialization and deserialization (dict, JSON, file I/O)
- SQL generation
- Error handling

### `test_integration.py`
Integration tests covering:
- Complete workflow from data import to prediction
- Tree structure manipulation
- Error handling for invalid operations
- End-to-end functionality

### `conftest.py`
Pytest configuration file with:
- Warning suppression for cleaner test output
- Test environment setup
- Global fixtures and settings

## Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test files:
```bash
python -m pytest tests/test_node.py -v
python -m pytest tests/test_asymmetree.py -v
python -m pytest tests/test_integration.py -v
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=asymmetree --cov-report=html
```

## Test Features

- **Fixtures**: Reusable test data and fitted tree instances
- **Parametrized tests**: Testing multiple scenarios efficiently
- **Error testing**: Comprehensive error condition coverage
- **Mocking**: Isolated testing of components
- **Integration testing**: End-to-end workflow validation

## Test Data

Tests use synthetic imbalanced datasets appropriate for AsymmeTree's use case:
- 5% positive class ratio (highly imbalanced)
- Mixed feature types (numerical, categorical)
- Various data sizes for different test scenarios

All tests are designed to be:
- **Fast**: Complete in under 1 second
- **Isolated**: No dependencies between tests
- **Deterministic**: Consistent results using fixed random seeds
- **Comprehensive**: Cover major functionality and edge cases 