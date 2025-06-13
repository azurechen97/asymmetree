# AsymmeTree Makefile
# Python project automation for development, testing, and deployment

.PHONY: help install install-dev test lint format clean build publish run dev-run check-deps update-deps venv

# Default target
help:
	@echo "AsymmeTree Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  help        Show this help message"
	@echo "  install     Install project dependencies"
	@echo "  install-dev Install development dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and isort"
	@echo "  clean       Clean up build artifacts and cache files"
	@echo "  build       Build the package"
	@echo "  publish     Publish to PyPI"
	@echo "  run         Run the main application"
	@echo "  dev-run     Run in development mode"
	@echo "  check-deps  Check for outdated dependencies"
	@echo "  update-deps Update dependencies"
	@echo "  venv        Create virtual environment"

# Variables
PYTHON := python3
PIP := pip
PACKAGE_NAME := asymmetree
VENV_DIR := .venv

# Virtual environment creation
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

# Install project dependencies
install:
	@echo "Installing project dependencies..."
	$(PIP) install -e .

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -e .[dev]
	$(PIP) install pytest pytest-cov black isort flake8 mypy pre-commit

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term

# Run linting checks
lint:
	@echo "Running linting checks..."
	flake8 $(PACKAGE_NAME)/ main.py
	mypy $(PACKAGE_NAME)/ main.py
	@echo "Linting completed!"

# Format code
format:
	@echo "Formatting code..."
	black $(PACKAGE_NAME)/ main.py
	isort $(PACKAGE_NAME)/ main.py
	@echo "Code formatting completed!"

# Clean up build artifacts and cache
clean:
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type f -name ".coverage" -delete
	@echo "Cleanup completed!"

# Build the package
build: clean
	@echo "Building package..."
	$(PYTHON) -m build
	@echo "Package built successfully!"

# Publish to PyPI
publish: build
	@echo "Publishing to PyPI..."
	$(PYTHON) -m twine upload dist/*
	@echo "Package published!"

# Run the main application
run:
	@echo "Running AsymmeTree..."
	$(PYTHON) main.py

# Run in development mode with verbose output
dev-run:
	@echo "Running AsymmeTree in development mode..."
	$(PYTHON) -v main.py

# Check for outdated dependencies
check-deps:
	@echo "Checking for outdated dependencies..."
	$(PIP) list --outdated

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	$(PIP) install --upgrade -r requirements.txt
	@echo "Dependencies updated!"

# Run all quality checks
quality: lint test
	@echo "All quality checks completed!"

# Setup development environment
setup-dev: venv install-dev
	@echo "Development environment setup completed!"
	@echo "Remember to activate the virtual environment: source $(VENV_DIR)/bin/activate"

# Quick development cycle: format, lint, and test
dev-cycle: format lint test
	@echo "Development cycle completed!" 