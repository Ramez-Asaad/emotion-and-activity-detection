"""
Base CNN Architecture
====================

Reusable CustomCNN architecture for image classification tasks.
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for image classification tasks.
    
    The model consists of 5 convolutional blocks followed by global average
    pooling and a linear classification head. The architecture is designed
    to be simple, stable, and fast to train.
    
    Args:
        num_classes (int): Number of output classes for classification.
    
    Architecture Details:
        - Block 1: Conv(3→32) + BatchNorm + ReLU + MaxPool
        - Block 2: Conv(32→64) + BatchNorm + ReLU + MaxPool
        - Block 3: Conv(64→128) + BatchNorm + ReLU + MaxPool
        - Block 4: Conv(128→256) + BatchNorm + ReLU
        - Block 5: Conv(256→256) + BatchNorm + ReLU
        - Global Average Pooling
        - Fully Connected: 256 → num_classes
    
    Input Shape: (batch_size, 3, 224, 224)
    Output Shape: (batch_size, num_classes)
    """
    
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Block 1: 3 → 32 channels
        # Output: (batch, 32, 112, 112) after pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 → 112
        )
        
        # Block 2: 32 → 64 channels
        # Output: (batch, 64, 56, 56) after pooling
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 → 56
        )
        
        # Block 3: 64 → 128 channels
        # Output: (batch, 128, 28, 28) after pooling
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 → 28
        )
        
        # Block 4: 128 → 256 channels
        # Output: (batch, 256, 28, 28) - no pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 5: 256 → 256 channels
        # Output: (batch, 256, 28, 28) - no pooling
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling
        # Reduces spatial dimensions to 1x1
        # Output: (batch, 256, 1, 1) → (batch, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classification Head
        # Output: (batch, num_classes)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        # Pass through convolutional blocks
        x = self.block1(x)  # (batch, 32, 112, 112)
        x = self.block2(x)  # (batch, 64, 56, 56)
        x = self.block3(x)  # (batch, 128, 28, 28)
        x = self.block4(x)  # (batch, 256, 28, 28)
        x = self.block5(x)  # (batch, 256, 28, 28)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 256, 1, 1)
        
        # Flatten for fully connected layer
        x = torch.flatten(x, 1)  # (batch, 256)
        
        # Classification head - returns raw logits
        x = self.fc(x)  # (batch, num_classes)
        
        return x


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
