"""
Base Configuration
==================

Base configuration class with common hyperparameters.
"""

import torch
from pathlib import Path


class BaseConfig:
    """Base configuration for training."""
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of data loading workers
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    LR_MIN = 1e-6
    
    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    
    # Image preprocessing
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Paths (relative to project root)
    PROJECT_ROOT = Path(__file__).parent.parent
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    DATASET_DIR = PROJECT_ROOT / 'datasets'
    
    # Checkpoint settings
    SAVE_BEST_ONLY = True
    SAVE_FREQUENCY = 5  # Save every N epochs
    
    # Logging
    PRINT_FREQUENCY = 10  # Print every N batches
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def display(cls):
        """Display current configuration."""
        print("=" * 70)
        print("Configuration")
        print("=" * 70)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and key.isupper():
                print(f"{key}: {value}")
        print("=" * 70)
