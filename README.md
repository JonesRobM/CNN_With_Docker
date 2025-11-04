# MNIST CNN Classification with Docker

A reproducible containerised deep learning workflow for classifying handwritten digits using a Convolutional Neural Network on the MNIST dataset.

## Features

- 🧠 Small CNN architecture optimised for MNIST
- 🐳 Fully containerised with Docker for reproducibility
- 📊 Training metrics and visualisation
- 🧪 Comprehensive unit tests
- 📦 Installable Python package
- 🎯 ~98-99% accuracy on MNIST test set

## Quick Start

### Installation

#### Option 1: Install from source
```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-cnn-docker.git
cd mnist-cnn-docker

# Install the package
pip install .

# Or install with development dependencies
pip install -e ".[dev]"
```

#### Option 2: Install dependencies only
```bash
pip install -r requirements.txt
```

### Usage

#### Local Training
```bash
# Train the model
python train_cnn.py

# Or use the installed command
mnist-train
```

#### Local Prediction
```bash
# Run predictions on test set
python predict_cnn.py

# Or use the installed command
mnist-predict
```

#### Docker Training
```bash
# Build the container
docker build -t mnist-cnn .

# Run training in container
docker run --rm -v $(pwd)/outputs:/app/outputs mnist-cnn

# Interactive mode
docker run --rm -it mnist-cnn /bin/bash
```

## Model Architecture

- **Input**: 28×28 grayscale images
- **Conv1**: 1→16 channels, 3×3 kernel, ReLU, MaxPool 2×2
- **Conv2**: 16→32 channels, 3×3 kernel, ReLU, MaxPool 2×2
- **FC**: Flatten → 800 → 64 neurons → ReLU → 10 output neurons
- **Loss**: CrossEntropyLoss
- **Optimiser**: Adam (lr=0.001)
- **Training**: 5-10 epochs, batch size 64

## Dataset

- **Source**: MNIST via `torchvision.datasets`
- **Training**: 60,000 images
- **Testing**: 10,000 images
- **Preprocessing**: Normalised to [0,1]
- **Download**: Automatic on first run

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_train_cnn.py
```

## Project Structure

```
.
├── train_cnn.py           # Training script
├── predict_cnn.py         # Prediction script
├── Dockerfile             # Container definition
├── requirements.txt       # Dependencies
├── setup.py              # Installation script
├── pyproject.toml        # Modern Python packaging
├── tests/                # Unit tests
│   ├── test_train_cnn.py
│   └── test_predict_cnn.py
├── data/                 # MNIST dataset (auto-downloaded)
└── outputs/              # Training outputs
    ├── model_*.pt        # Saved models
    └── *.png            # Visualisations
```

## Development

### Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
```

### Build Distribution
```bash
python -m build
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- matplotlib 3.7+
- numpy 1.24+

Optional:
- pytest (for testing)
- scikit-learn (for confusion matrix)
- seaborn (for visualisation)

## Outputs

- `model.pt` - Trained model checkpoint
- `outputs/model_YYYYMMDD_HHMMSS.pt` - Timestamped model
- `outputs/training_history_*.png` - Training curves
- `outputs/predictions.png` - Sample predictions
- `outputs/confusion_matrix.png` - Confusion matrix

## License

Apache License - See LICENSE file for details

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mnist_cnn_docker,
  title={MNIST CNN Classification with Docker},
  author={MNIST CNN Project},
  year={2024},
  url={https://github.com/JonesRobM/mnist-cnn-docker}
}
```
