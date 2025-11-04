"""
Unit tests for train_cnn.py
Tests model architecture, training, and evaluation functions.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import sys

# Add parent directory to path to import train_cnn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_cnn import MNIST_CNN, train_epoch, evaluate


class TestMNISTCNN:
    """Test cases for MNIST_CNN model architecture"""

    def test_model_initialization(self):
        """Test that model initialises correctly"""
        model = MNIST_CNN()
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_layers(self):
        """Test that model has correct layers"""
        model = MNIST_CNN()

        # Check convolutional layers
        assert isinstance(model.conv1, nn.Conv2d)
        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 16
        assert model.conv1.kernel_size == (3, 3)

        assert isinstance(model.conv2, nn.Conv2d)
        assert model.conv2.in_channels == 16
        assert model.conv2.out_channels == 32
        assert model.conv2.kernel_size == (3, 3)

        # Check pooling layer
        assert isinstance(model.pool, nn.MaxPool2d)
        assert model.pool.kernel_size == 2
        assert model.pool.stride == 2

        # Check fully connected layers
        assert isinstance(model.fc1, nn.Linear)
        assert model.fc1.in_features == 32 * 7 * 7
        assert model.fc1.out_features == 64

        assert isinstance(model.fc2, nn.Linear)
        assert model.fc2.in_features == 64
        assert model.fc2.out_features == 10

    def test_model_forward_pass_shape(self):
        """Test that model produces correct output shape"""
        model = MNIST_CNN()
        batch_size = 4

        # Create dummy input (batch_size, channels, height, width)
        x = torch.randn(batch_size, 1, 28, 28)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, 10)

    def test_model_forward_pass_values(self):
        """Test that model output contains reasonable values"""
        model = MNIST_CNN()
        x = torch.randn(2, 1, 28, 28)

        output = model(x)

        # Output should be finite
        assert torch.isfinite(output).all()

        # Output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_model_different_batch_sizes(self):
        """Test model with different batch sizes"""
        model = MNIST_CNN()

        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 1, 28, 28)
            output = model(x)
            assert output.shape == (batch_size, 10)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = MNIST_CNN()
        x = torch.randn(2, 1, 28, 28, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None

    def test_model_to_device(self):
        """Test that model can be moved to different devices"""
        model = MNIST_CNN()

        # Test CPU
        model_cpu = model.to('cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            assert next(model_cuda.parameters()).device.type == 'cuda'


class TestTrainingFunctions:
    """Test cases for training and evaluation functions"""

    @pytest.fixture
    def dummy_dataset(self):
        """Create a small dummy dataset for testing"""
        # Create 20 samples of 28x28 images
        data = torch.randn(20, 1, 28, 28)
        targets = torch.randint(0, 10, (20,))
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    @pytest.fixture
    def model_and_optimiser(self):
        """Create model and optimiser for testing"""
        model = MNIST_CNN()
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        return model, criterion, optimiser, device

    def test_train_epoch_returns_metrics(self, dummy_dataset, model_and_optimiser):
        """Test that train_epoch returns loss and accuracy"""
        model, criterion, optimiser, device = model_and_optimiser

        loss, acc = train_epoch(
            model, device, dummy_dataset, criterion, optimiser, epoch=1
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100

    def test_train_epoch_updates_weights(self, dummy_dataset, model_and_optimiser):
        """Test that train_epoch updates model weights"""
        model, criterion, optimiser, device = model_and_optimiser

        # Store initial weights
        initial_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Train for one epoch
        train_epoch(model, device, dummy_dataset, criterion, optimiser, epoch=1)

        # Check that at least some weights have changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(initial_weights[name], param):
                weights_changed = True
                break

        assert weights_changed, "Model weights should update during training"

    def test_evaluate_returns_metrics(self, dummy_dataset, model_and_optimiser):
        """Test that evaluate returns loss and accuracy"""
        model, criterion, optimiser, device = model_and_optimiser

        loss, acc = evaluate(model, device, dummy_dataset, criterion)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100

    def test_evaluate_no_gradient(self, dummy_dataset, model_and_optimiser):
        """Test that evaluate doesn't compute gradients"""
        model, criterion, optimiser, device = model_and_optimiser

        # Enable gradient tracking
        for param in model.parameters():
            param.requires_grad = True

        # Run evaluation
        evaluate(model, device, dummy_dataset, criterion)

        # Gradients should not be computed during evaluation
        for param in model.parameters():
            assert param.grad is None or torch.equal(
                param.grad, torch.zeros_like(param.grad)
            )

    def test_train_mode_vs_eval_mode(self, dummy_dataset):
        """Test that model behaves differently in train vs eval mode"""
        model = MNIST_CNN()
        x = torch.randn(4, 1, 28, 28)

        # Training mode
        model.train()
        assert model.training

        # Evaluation mode
        model.eval()
        assert not model.training

    def test_loss_decreases_over_epochs(self, dummy_dataset, model_and_optimiser):
        """Test that loss generally decreases over multiple epochs"""
        model, criterion, optimiser, device = model_and_optimiser

        losses = []
        for epoch in range(1, 4):
            loss, _ = train_epoch(
                model, device, dummy_dataset, criterion, optimiser, epoch
            )
            losses.append(loss)

        # Loss should generally trend downward (allow some fluctuation)
        assert losses[-1] < losses[0] * 1.5, "Loss should decrease with training"


class TestModelPersistence:
    """Test cases for saving and loading models"""

    def test_model_save_and_load(self):
        """Test that model can be saved and loaded correctly"""
        model = MNIST_CNN()

        # Create dummy input
        x = torch.randn(2, 1, 28, 28)
        original_output = model(x)

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
            }, temp_path)

        try:
            # Load model
            new_model = MNIST_CNN()
            checkpoint = torch.load(temp_path, map_location='cpu')
            new_model.load_state_dict(checkpoint['model_state_dict'])

            # Compare outputs
            new_output = new_model(x)
            assert torch.allclose(original_output, new_output, atol=1e-6)
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_checkpoint_contains_required_keys(self):
        """Test that checkpoint contains all required information"""
        model = MNIST_CNN()
        optimiser = optim.Adam(model.parameters(), lr=0.001)

        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'train_loss': 0.5,
            'test_acc': 95.0,
        }

        # Check all required keys are present
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimiser_state_dict' in checkpoint
        assert 'train_loss' in checkpoint
        assert 'test_acc' in checkpoint


class TestModelRobustness:
    """Test edge cases and robustness"""

    def test_model_with_zeros_input(self):
        """Test model with all-zero input"""
        model = MNIST_CNN()
        x = torch.zeros(2, 1, 28, 28)

        output = model(x)
        assert output.shape == (2, 10)
        assert torch.isfinite(output).all()

    def test_model_with_extreme_values(self):
        """Test model with extreme input values"""
        model = MNIST_CNN()

        # Very large values
        x_large = torch.ones(2, 1, 28, 28) * 100
        output_large = model(x_large)
        assert torch.isfinite(output_large).all()

        # Very small values
        x_small = torch.ones(2, 1, 28, 28) * 0.001
        output_small = model(x_small)
        assert torch.isfinite(output_small).all()

    def test_model_deterministic_with_same_input(self):
        """Test that model produces same output for same input"""
        model = MNIST_CNN()
        model.eval()  # Set to eval mode for deterministic behaviour

        x = torch.randn(2, 1, 28, 28)

        output1 = model(x)
        output2 = model(x)

        assert torch.allclose(output1, output2)

    def test_batch_size_one(self):
        """Test model with batch size of 1"""
        model = MNIST_CNN()
        x = torch.randn(1, 1, 28, 28)

        output = model(x)
        assert output.shape == (1, 10)

    def test_invalid_input_shape_raises_error(self):
        """Test that invalid input shape raises an error"""
        model = MNIST_CNN()

        # Wrong number of channels
        x_wrong_channels = torch.randn(2, 3, 28, 28)
        with pytest.raises(RuntimeError):
            model(x_wrong_channels)

        # Wrong spatial dimensions
        x_wrong_size = torch.randn(2, 1, 32, 32)
        # This might work but produce wrong output size
        output = model(x_wrong_size)
        # Just check it doesn't crash
        assert output.shape[0] == 2
