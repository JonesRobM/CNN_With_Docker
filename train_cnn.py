"""
MNIST CNN Training Script
Trains a small CNN on MNIST dataset and saves the trained model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime


class MNIST_CNN(nn.Module):
    """
    Small CNN for MNIST classification
    Architecture:
    - Conv1: 1->16 channels, 3x3 kernel, ReLU, MaxPool 2x2
    - Conv2: 16->32 channels, 3x3 kernel, ReLU, MaxPool 2x2
    - FC: Flatten -> 800 -> 64 -> 10
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        # With 32 channels: 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 32 * 7 * 7)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def train_epoch(model, device, train_loader, criterion, optimiser, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimiser.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimise
        loss.backward()
        optimiser.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, device, test_loader, criterion):
    """Evaluate the model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += criterion(output, target).item()

            # Get predictions
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total

    return test_loss, test_acc


def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std (normalise)
    ])

    # Load datasets
    print('Loading MNIST dataset...')
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Use 0 for Windows compatibility
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    # Initialize model
    model = MNIST_CNN().to(device)
    print('\nModel Architecture:')
    print(model)

    # Loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f'\nStarting training for {EPOCHS} epochs...\n')
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimiser, epoch)

        # Evaluate
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n')

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'outputs/model_{timestamp}.pt'
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'train_loss': train_loss,
        'test_acc': test_acc,
    }, model_path)

    # Also save as model.pt for easy reference
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'train_loss': train_loss,
        'test_acc': test_acc,
    }, 'model.pt')

    print(f'\nModel saved to {model_path} and model.pt')

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, EPOCHS + 1), test_accs, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = f'outputs/training_history_{timestamp}.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Training history plot saved to {plot_path}')

    # Final results
    print('\n' + '='*50)
    print('Training Complete!')
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    print('='*50)


if __name__ == '__main__':
    main()
