# Unit Tests for MNIST CNN Project

This directory contains comprehensive unit tests for the MNIST CNN classification project.

## Test Files

### `test_train_cnn.py`
Tests for training functionality including:
- **Model Architecture Tests**: Verify CNN layers, dimensions, and structure
- **Training Functions**: Test training epoch and evaluation functions
- **Model Persistence**: Test model saving and loading
- **Robustness Tests**: Edge cases and invalid inputs

### `test_predict_cnn.py`
Tests for prediction and visualisation functionality including:
- **Model Loading**: Test checkpoint loading and device placement
- **Prediction Functions**: Test sample predictions and single image inference
- **Visualisation**: Test prediction visualisations
- **Confusion Matrix**: Test confusion matrix generation
- **Integration Tests**: End-to-end prediction pipeline

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_train_cnn.py
pytest tests/test_predict_cnn.py
```

### Run Specific Test Class
```bash
pytest tests/test_train_cnn.py::TestMNISTCNN
pytest tests/test_predict_cnn.py::TestModelLoading
```

### Run Specific Test Function
```bash
pytest tests/test_train_cnn.py::TestMNISTCNN::test_model_initialization
```

### Run with Coverage Report
```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` to view the coverage report.

### Run with Verbose Output
```bash
pytest -v
```

### Run Tests Matching Pattern
```bash
pytest -k "model"  # Run tests with "model" in the name
pytest -k "not slow"  # Skip slow tests
```

### Run with Output Capture Disabled
```bash
pytest -s  # Show print statements
```

## Test Markers

Tests can be marked with custom markers:
- `@pytest.mark.slow` - Marks slow-running tests
- `@pytest.mark.integration` - Marks integration tests
- `@pytest.mark.unit` - Marks unit tests

Run specific marked tests:
```bash
pytest -m unit  # Run only unit tests
pytest -m "not slow"  # Skip slow tests
```

## Requirements

Install test dependencies:
```bash
pip install -r requirements.txt
```

Core testing packages:
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `scikit-learn>=1.3.0` - For confusion matrix tests (optional)
- `seaborn>=0.12.0` - For confusion matrix visualisation (optional)

## Test Coverage

Current test coverage includes:
- Model architecture validation
- Forward pass correctness
- Training loop functionality
- Gradient flow verification
- Model persistence (save/load)
- Prediction accuracy
- Visualisation generation
- Edge cases and error handling

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They:
- Use non-interactive matplotlib backend
- Create temporary files that are cleaned up
- Mock large datasets to reduce test time
- Run without GPU requirements

## Notes

- Tests use CPU by default for compatibility
- Large dataset downloads are avoided using synthetic data
- Matplotlib figures are automatically closed to prevent memory leaks
- Temporary files are cleaned up after each test
