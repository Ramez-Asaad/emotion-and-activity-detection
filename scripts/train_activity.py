"""
Train Activity Recognition Model
================================

Training script for UCF101 activity recognition task.
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

from models.activity_model import ActivityModel
from configs.activity_config import ActivityConfig
from data.transforms import get_train_transforms, get_val_transforms
from utils.training import Trainer, set_seed
from utils.checkpoint import save_checkpoint, save_best_model
from utils.visualization import plot_training_history


def main():
    """Main training function."""
    
    # Configuration
    config = ActivityConfig()
    config.create_dirs()
    config.display()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    print("\n" + "=" * 70)
    print("Loading UCF101 Dataset")
    print("=" * 70)
    
    # Data transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Load datasets
    # Three options for loading UCF101:
    # 1. UCF101ImageFolder - Simple image folder (recommended for pre-extracted frames)
    # 2. UCF101Dataset - Advanced loader with video support
    # 3. ImageFolder - Standard PyTorch loader
    
    try:
        from data.activity_dataset import UCF101ImageFolder, UCF101Dataset
        
        # Use UCF101Dataset which handles video subfolders
        # Our structure: UCF101/train/Walking/WalkingWithDog_v_.../frame_0000.jpg
        print("Loading dataset using UCF101Dataset...")
        train_dataset = UCF101Dataset(
            root_dir=config.TRAIN_DIR,
            transform=train_transform,
            frames_per_video=1,  # Load 1 frame per video folder
            frame_sampling='center',  # Use center frame
            use_videos=False  # We have extracted frames, not videos
        )
        val_dataset = UCF101Dataset(
            root_dir=config.VAL_DIR,
            transform=val_transform,
            frames_per_video=1,
            frame_sampling='center',
            use_videos=False
        )
        
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        
        # Update class names from dataset
        class_names = train_dataset.classes
        ActivityModel.update_activity_labels({i: name for i, name in enumerate(class_names)})
        config.update_class_names(class_names)
        print(f"✓ Activity classes: {class_names}")
        
    except FileNotFoundError:
        print("\n⚠ Dataset not found!")
        print(f"Please organize your UCF101 dataset in:")
        print(f"  Train: {config.TRAIN_DIR}")
        print(f"  Val:   {config.VAL_DIR}")
        print("\nDataset structure options:")
        print("\nOption 1 - Pre-extracted frames (RECOMMENDED):")
        print("  UCF101/")
        print("    train/")
        print("      class1/")
        print("        image1.jpg")
        print("        image2.jpg")
        print("      class2/")
        print("        ...")
        print("    val/")
        print("      (same structure)")
        print("\nOption 2 - Video files:")
        print("  UCF101/")
        print("    train/")
        print("      class1/")
        print("        video1.avi")
        print("        video2.avi")
        print("      class2/")
        print("        ...")
        print("\nOption 3 - Frames in video folders:")
        print("  UCF101/")
        print("    train/")
        print("      class1/")
        print("        video1/")
        print("          frame_001.jpg")
        print("          frame_002.jpg")
        print("        video2/")
        print("          ...")
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
    model = ActivityModel()
    print(f"✓ Model created: ActivityModel (5 activity classes)")
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
