"""
MNIST CNN Classification Project
A reproducible containerised deep learning workflow for handwritten digit recognition.
"""

__version__ = '1.0.0'
__author__ = 'MNIST CNN Project'

# Import main components for easy access
try:
    from train_cnn import MNIST_CNN, train_epoch, evaluate
    from predict_cnn import load_model, predict_samples, visualise_predictions, predict_single_image
except ImportError:
    # Handle case where dependencies aren't installed yet
    pass

__all__ = [
    'MNIST_CNN',
    'train_epoch',
    'evaluate',
    'load_model',
    'predict_samples',
    'visualise_predictions',
    'predict_single_image',
]
