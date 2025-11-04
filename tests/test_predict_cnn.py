"""
Unit tests for predict_cnn.py
Tests model loading, prediction, and visualisation functions.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_cnn import MNIST_CNN
from predict_cnn import (
    load_model,
    predict_samples,
    visualise_predictions,
    predict_single_image,
    compute_confusion_matrix
)


class TestModelLoading:
    """Test cases for loading trained models"""

    @pytest.fixture
    def saved_model_path(self):
        """Create a temporary saved model for testing"""
        model = MNIST_CNN()
        optimiser = optim.Adam(model.parameters(), lr=0.001)

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name
            torch.save({
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'train_loss': 0.234,
                'test_acc': 95.5,
            }, temp_path)

        yield temp_path

        # Clean up after test
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_load_model_success(self, saved_model_path):
        """Test that model loads successfully"""
        model = load_model(saved_model_path, device='cpu')

        assert model is not None
        assert isinstance(model, MNIST_CNN)

    def test_loaded_model_in_eval_mode(self, saved_model_path):
        """Test that loaded model is in evaluation mode"""
        model = load_model(saved_model_path, device='cpu')

        assert not model.training

    def test_loaded_model_produces_output(self, saved_model_path):
        """Test that loaded model can produce predictions"""
        model = load_model(saved_model_path, device='cpu')

        x = torch.randn(4, 1, 28, 28)
        output = model(x)

        assert output.shape == (4, 10)
        assert torch.isfinite(output).all()

    def test_load_model_nonexistent_file(self):
        """Test that loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.pt', device='cpu')

    def test_load_model_different_devices(self, saved_model_path):
        """Test loading model on different devices"""
        # CPU
        model_cpu = load_model(saved_model_path, device='cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'

        # CUDA if available
        if torch.cuda.is_available():
            model_cuda = load_model(saved_model_path, device='cuda')
            assert next(model_cuda.parameters()).device.type == 'cuda'


class TestPredictSamples:
    """Test cases for predict_samples function"""

    @pytest.fixture
    def dummy_test_loader(self):
        """Create dummy test data loader"""
        data = torch.randn(50, 1, 28, 28)
        targets = torch.randint(0, 10, (50,))
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=8, shuffle=False)

    @pytest.fixture
    def trained_model(self):
        """Create a model for testing"""
        model = MNIST_CNN()
        model.eval()
        return model

    def test_predict_samples_returns_correct_count(
        self, trained_model, dummy_test_loader
    ):
        """Test that predict_samples returns requested number of samples"""
        device = torch.device('cpu')
        num_samples = 20

        images, true_labels, pred_labels = predict_samples(
            trained_model, device, dummy_test_loader, num_samples
        )

        assert len(images) == num_samples
        assert len(true_labels) == num_samples
        assert len(pred_labels) == num_samples

    def test_predict_samples_output_types(self, trained_model, dummy_test_loader):
        """Test that predict_samples returns correct types"""
        device = torch.device('cpu')

        images, true_labels, pred_labels = predict_samples(
            trained_model, device, dummy_test_loader, num_samples=10
        )

        assert all(isinstance(img, torch.Tensor) for img in images)
        assert all(isinstance(label, int) for label in true_labels)
        assert all(isinstance(label, int) for label in pred_labels)

    def test_predict_samples_label_range(self, trained_model, dummy_test_loader):
        """Test that predicted labels are in valid range"""
        device = torch.device('cpu')

        _, true_labels, pred_labels = predict_samples(
            trained_model, device, dummy_test_loader, num_samples=15
        )

        assert all(0 <= label < 10 for label in true_labels)
        assert all(0 <= label < 10 for label in pred_labels)

    def test_predict_samples_image_shapes(self, trained_model, dummy_test_loader):
        """Test that images have correct shape"""
        device = torch.device('cpu')

        images, _, _ = predict_samples(
            trained_model, device, dummy_test_loader, num_samples=5
        )

        for img in images:
            assert img.shape == (1, 28, 28)

    def test_predict_samples_with_small_dataset(self, trained_model):
        """Test predict_samples with dataset smaller than requested samples"""
        device = torch.device('cpu')

        # Create very small dataset
        data = torch.randn(5, 1, 28, 28)
        targets = torch.randint(0, 10, (5,))
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        images, true_labels, pred_labels = predict_samples(
            trained_model, device, loader, num_samples=20
        )

        # Should return only available samples
        assert len(images) == 5
        assert len(true_labels) == 5
        assert len(pred_labels) == 5


class TestVisualisePredictions:
    """Test cases for visualise_predictions function"""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data"""
        images = [torch.randn(1, 28, 28) for _ in range(10)]
        true_labels = [i % 10 for i in range(10)]
        pred_labels = [(i + 1) % 10 for i in range(10)]
        return images, true_labels, pred_labels

    def test_visualise_predictions_creates_file(self, sample_predictions):
        """Test that visualisation creates output file"""
        images, true_labels, pred_labels = sample_predictions

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name

        try:
            visualise_predictions(
                images, true_labels, pred_labels, save_path=temp_path
            )

            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close('all')

    def test_visualise_predictions_with_different_counts(self):
        """Test visualisation with different numbers of samples"""
        for num_samples in [1, 5, 10, 20]:
            images = [torch.randn(1, 28, 28) for _ in range(num_samples)]
            true_labels = list(range(num_samples))
            pred_labels = list(range(num_samples))

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                temp_path = f.name

            try:
                visualise_predictions(
                    images, true_labels, pred_labels, save_path=temp_path
                )
                assert os.path.exists(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                plt.close('all')

    def test_visualise_predictions_handles_correct_incorrect(self):
        """Test visualisation with mix of correct and incorrect predictions"""
        images = [torch.randn(1, 28, 28) for _ in range(6)]
        true_labels = [0, 1, 2, 3, 4, 5]
        pred_labels = [0, 1, 2, 9, 9, 9]  # First 3 correct, last 3 wrong

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            temp_path = f.name

        try:
            visualise_predictions(
                images, true_labels, pred_labels, save_path=temp_path
            )
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close('all')


class TestPredictSingleImage:
    """Test cases for predict_single_image function"""

    @pytest.fixture
    def model(self):
        """Create model for testing"""
        model = MNIST_CNN()
        model.eval()
        return model

    def test_predict_single_image_returns_prediction(self, model):
        """Test that predict_single_image returns valid prediction"""
        device = torch.device('cpu')
        image = torch.randn(1, 28, 28)

        prediction, probabilities = predict_single_image(model, device, image)

        assert isinstance(prediction, int)
        assert 0 <= prediction < 10
        assert isinstance(probabilities, object)  # numpy array
        assert len(probabilities) == 10

    def test_predict_single_image_probabilities_sum_to_one(self, model):
        """Test that probabilities sum to approximately 1"""
        device = torch.device('cpu')
        image = torch.randn(1, 28, 28)

        _, probabilities = predict_single_image(model, device, image)

        # Probabilities should sum to approximately 1
        assert abs(probabilities.sum() - 1.0) < 1e-5

    def test_predict_single_image_probabilities_positive(self, model):
        """Test that all probabilities are positive"""
        device = torch.device('cpu')
        image = torch.randn(1, 28, 28)

        _, probabilities = predict_single_image(model, device, image)

        assert all(p >= 0 for p in probabilities)
        assert all(p <= 1 for p in probabilities)

    def test_predict_single_image_deterministic(self, model):
        """Test that same image produces same prediction"""
        device = torch.device('cpu')
        image = torch.randn(1, 28, 28)

        pred1, prob1 = predict_single_image(model, device, image)
        pred2, prob2 = predict_single_image(model, device, image)

        assert pred1 == pred2
        assert (prob1 == prob2).all()


class TestConfusionMatrix:
    """Test cases for confusion matrix computation"""

    @pytest.fixture
    def dummy_test_loader(self):
        """Create dummy test data loader"""
        data = torch.randn(100, 1, 28, 28)
        targets = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=10, shuffle=False)

    @pytest.fixture
    def model(self):
        """Create model for testing"""
        model = MNIST_CNN()
        model.eval()
        return model

    def test_compute_confusion_matrix_runs(self, model, dummy_test_loader):
        """Test that confusion matrix computation runs without errors"""
        device = torch.device('cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create outputs directory
                os.makedirs('outputs', exist_ok=True)

                # This requires scikit-learn and seaborn
                try:
                    compute_confusion_matrix(model, device, dummy_test_loader)
                    assert os.path.exists('outputs/confusion_matrix.png')
                except ImportError:
                    pytest.skip("scikit-learn or seaborn not installed")
            finally:
                os.chdir(original_dir)
                plt.close('all')


class TestIntegration:
    """Integration tests for the prediction pipeline"""

    def test_full_prediction_pipeline(self):
        """Test complete prediction workflow"""
        # Create and save a model
        model = MNIST_CNN()
        optimiser = optim.Adam(model.parameters(), lr=0.001)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            model_path = f.name
            torch.save({
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'train_loss': 0.5,
                'test_acc': 90.0,
            }, model_path)

        try:
            # Load the model
            loaded_model = load_model(model_path, device='cpu')

            # Create test data
            data = torch.randn(20, 1, 28, 28)
            targets = torch.randint(0, 10, (20,))
            dataset = TensorDataset(data, targets)
            loader = DataLoader(dataset, batch_size=4, shuffle=False)

            # Make predictions
            images, true_labels, pred_labels = predict_samples(
                loaded_model, torch.device('cpu'), loader, num_samples=10
            )

            # Verify results
            assert len(images) == 10
            assert len(true_labels) == 10
            assert len(pred_labels) == 10

            # Create visualisation
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                viz_path = f.name

            try:
                visualise_predictions(
                    images, true_labels, pred_labels, save_path=viz_path
                )
                assert os.path.exists(viz_path)
            finally:
                if os.path.exists(viz_path):
                    os.unlink(viz_path)
                plt.close('all')

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_prediction_consistency(self):
        """Test that predictions are consistent across calls"""
        model = MNIST_CNN()
        model.eval()
        device = torch.device('cpu')

        # Create test data
        data = torch.randn(10, 1, 28, 28)
        targets = torch.randint(0, 10, (10,))
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=5, shuffle=False)

        # Make predictions twice
        _, _, pred_labels_1 = predict_samples(model, device, loader, num_samples=10)

        # Reset loader
        loader = DataLoader(dataset, batch_size=5, shuffle=False)
        _, _, pred_labels_2 = predict_samples(model, device, loader, num_samples=10)

        # Predictions should be identical
        assert pred_labels_1 == pred_labels_2
