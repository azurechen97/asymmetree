import pytest
import warnings


def pytest_configure(config):
    """Configure pytest with custom settings."""
    import pandas as pd

    # Suppress pandas performance warnings during testing
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    # Suppress other common warnings that don't affect functionality
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment settings."""
    import pandas as pd
    import numpy as np

    # Set random seed for reproducible tests
    np.random.seed(42)

    # Configure pandas for testing
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    yield

    # Cleanup after tests
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
