"""
MNIST CNN Prediction Script
Loads trained model and visualises predictions on test samples.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from train_cnn import MNIST_CNN


def load_model(model_path='model.pt', device='cpu'):
    """Load trained model from checkpoint"""
    model = MNIST_CNN().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Model loaded from {model_path}')
    print(f'Model was trained for {checkpoint["epoch"]} epochs')
    print(f'Final test accuracy: {checkpoint["test_acc"]:.2f}%')

    return model


def predict_samples(model, device, test_loader, num_samples=20):
    """Make predictions on sample images"""
    model.eval()

    images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output.data, 1)

            # Collect samples
            for i in range(len(data)):
                if len(images) >= num_samples:
                    break

                images.append(data[i].cpu())
                true_labels.append(target[i].cpu().item())
                pred_labels.append(predicted[i].cpu().item())

            if len(images) >= num_samples:
                break

    return images, true_labels, pred_labels


def visualise_predictions(images, true_labels, pred_labels, save_path='outputs/predictions.png'):
    """Visualise predictions with true labels"""
    num_samples = len(images)
    cols = 5
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        ax = axes[i]

        # Display image
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')

        # Set title with true and predicted labels
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        color = 'green' if true_label == pred_label else 'red'

        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        ax.axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nPrediction visualisation saved to {save_path}')
    plt.close()


def compute_confusion_matrix(model, device, test_loader):
    """Compute and display confusion matrix"""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    print('Confusion matrix saved to outputs/confusion_matrix.png')
    plt.close()

    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_targets, all_preds, target_names=[str(i) for i in range(10)]))


def predict_single_image(model, device, image_tensor):
    """Predict a single image"""
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)

    return predicted.item(), probabilities.cpu().numpy()[0]


def main():
    # Configuration
    MODEL_PATH = 'model.pt'
    NUM_SAMPLES = 20

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f'Error: Model file "{MODEL_PATH}" not found.')
        print('Please train the model first using train_cnn.py')
        return

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # Load model
    model = load_model(MODEL_PATH, device)

    # Data transforms (same as training, normalise)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load test dataset
    print('\nLoading MNIST test dataset...')
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # Make predictions on sample images
    print(f'\nGenerating predictions for {NUM_SAMPLES} sample images...')
    images, true_labels, pred_labels = predict_samples(model, device, test_loader, NUM_SAMPLES)

    # Calculate accuracy on samples
    correct = sum([1 for t, p in zip(true_labels, pred_labels) if t == p])
    accuracy = 100 * correct / len(true_labels)
    print(f'Sample accuracy: {correct}/{len(true_labels)} ({accuracy:.2f}%)')

    # Visualise predictions
    visualise_predictions(images, true_labels, pred_labels)

    # Compute confusion matrix (requires scikit-learn)
    try:
        print('\nComputing confusion matrix...')
        compute_confusion_matrix(model, device, test_loader)
    except ImportError:
        print('\nNote: Install scikit-learn and seaborn to generate confusion matrix')
        print('  pip install scikit-learn seaborn')

    print('\n' + '='*50)
    print('Prediction complete!')
    print('='*50)


if __name__ == '__main__':
    main()
