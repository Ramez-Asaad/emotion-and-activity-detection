
"""
Train Emotion Recognition Model
===============================

Training script for FER-2013 emotion recognition task.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from models.emotion_model import EmotionModel
from configs.emotion_config import EmotionConfig
from data.transforms import get_train_transforms, get_val_transforms
from utils.training import Trainer, set_seed
from utils.checkpoint import save_checkpoint, save_best_model
from utils.visualization import plot_training_history


def main():
    """Main training function."""
    
    # Configuration
    config = EmotionConfig()
    config.create_dirs()
    config.display()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    print("\n" + "=" * 70)
    print("Loading FER-2013 Dataset")
    print("=" * 70)
    
    # Data transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Load datasets
    try:
        from data.emotion_dataset import FER2013Dataset
        
        print("Loading dataset using FER2013Dataset...")
        train_dataset = FER2013Dataset(
            root_dir=config.TRAIN_DIR,
            transform=train_transform
        )
        val_dataset = FER2013Dataset(
            root_dir=config.VAL_DIR,
            transform=val_transform
        )
        
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        
        # Show class distribution
        train_dist = train_dataset.get_class_distribution()
        print(f"\nTraining set class distribution:")
        for emotion, count in train_dist.items():
            print(f"  {emotion}: {count} samples")
        
    except FileNotFoundError:
        print("\n⚠ Dataset not found!")
        print(f"Please organize your FER-2013 dataset in:")
        print(f"  Train: {config.TRAIN_DIR}")
        print(f"  Val:   {config.VAL_DIR}")
        print("\nDataset structure should be:")
        print("  FER2013/")
        print("    train/")
        print("      Angry/")
        print("        image1.jpg")
        print("        image2.jpg")
        print("      Disgust/")
        print("        ...")
        print("      Fear/")
        print("      Happy/")
        print("      Sad/")
        print("      Surprise/")
        print("      Neutral/")
        print("    val/")
        print("      (same structure)")
        return
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)
    
    # Model
    model = EmotionModel()
    print(f"✓ Model created: EmotionModel (7 emotion classes)")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.LR_MIN
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=config.DEVICE
    )
    
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("=" * 70 + "\n")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = trainer.train_epoch(epoch)
        
        # Validate
        val_loss, val_acc = trainer.validate()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        save_best_model(model, metrics, config, metric_name='val_acc')
        
        # Save checkpoint periodically
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics, config,
                filename=f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # Early stopping
        if trainer.check_early_stopping(val_acc):
            break
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    
    # Plot training history
    plot_path = config.RESULTS_DIR / 'training_history.png'
    plot_training_history(history, save_path=plot_path)
    
    print(f"\n✓ Results saved to: {config.RESULTS_DIR}")
    print(f"✓ Best model saved to: {config.CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
