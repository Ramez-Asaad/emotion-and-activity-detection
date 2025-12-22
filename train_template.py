"""
Training Template for Custom CNN
=================================

This is a template showing how to train the CustomCNN model.
Modify this for your specific dataset and training requirements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_cnn import CustomCNN


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Training loop for the Custom CNN model.
    
    Args:
        model: CustomCNN model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print("\n")
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model


def test_model(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained CustomCNN model
        test_loader: DataLoader for test data
    
    Returns:
        float: Test accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc


def load_checkpoint(model, checkpoint_path):
    """
    Load a saved model checkpoint.
    
    Args:
        model: CustomCNN model instance
        checkpoint_path: Path to checkpoint file
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    return model


if __name__ == "__main__":
    """
    Example usage - modify this section for your dataset
    """
    
    print("=" * 70)
    print("Custom CNN Training Template")
    print("=" * 70)
    print("\nThis is a template. You need to:")
    print("1. Load your dataset (FER-2013 or UCF101)")
    print("2. Create DataLoaders")
    print("3. Call train_model() with your data")
    print("\nExample code structure:\n")
    
    print("""
# Example for FER-2013 (7 emotion classes)
from torchvision import transforms, datasets

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('path/to/train', transform=transform)
val_dataset = datasets.ImageFolder('path/to/val', transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = CustomCNN(num_classes=7)  # 7 for FER-2013

# Train
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    learning_rate=0.001
)

# Test
test_dataset = datasets.ImageFolder('path/to/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_acc = test_model(trained_model, test_loader)
    """)
    
    print("\n" + "=" * 70)
    print("For UCF101 (5 classes), simply change num_classes=5")
    print("=" * 70)
