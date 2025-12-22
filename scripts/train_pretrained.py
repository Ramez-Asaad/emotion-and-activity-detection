"""
Train Pretrained Models
========================

Unified training script for all pretrained models with transfer learning.

Usage:
    python scripts/train_pretrained.py --model resnet18 --task emotion --epochs 20
    python scripts/train_pretrained.py --model mobilenet_v2 --task activity --epochs 20
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.pretrained_models import create_pretrained_model, list_available_models
from utils.experiment_logger import ExperimentLogger
from configs.emotion_config import EmotionConfig
from configs.activity_config import ActivityConfig


def get_config(task):
    """Get configuration for task."""
    if task == 'emotion':
        return EmotionConfig()
    elif task == 'activity':
        return ActivityConfig()
    else:
        raise ValueError(f"Unknown task: {task}")


def get_dataloaders(config, task):
    """Get data loaders for the task."""
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip() if config.HORIZONTAL_FLIP else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if task == 'emotion':
        from data.emotion_dataset import FER2013Dataset
        
        train_dataset = FER2013Dataset(
            root_dir=config.TRAIN_DIR,
            transform=train_transform
        )
        val_dataset = FER2013Dataset(
            root_dir=config.VAL_DIR,
            transform=val_transform
        )
    else:  # activity
        from data.activity_dataset import UCF101Dataset
        
        train_dataset = UCF101Dataset(
            root_dir=config.TRAIN_DIR,
            transform=train_transform,
            frames_per_video=1,
            frame_sampling='center',
            use_videos=False
        )
        val_dataset = UCF101Dataset(
            root_dir=config.VAL_DIR,
            transform=val_transform,
            frames_per_video=1,
            frame_sampling='center',
            use_videos=False
        )
    
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
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train pretrained models')
    parser.add_argument('--model', type=str, required=True,
                       choices=list_available_models(),
                       help='Model architecture')
    parser.add_argument('--task', type=str, required=True,
                       choices=['emotion', 'activity'],
                       help='Task to train on')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                       help='Number of epochs to freeze backbone')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Training {args.model.upper()} on {args.task.upper()}")
    print("=" * 70)
    
    # Get config
    config = get_config(args.task)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print(f"\nCreating model...")
    model = create_pretrained_model(
        args.model,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    model = model.to(device)
    
    # Freeze backbone initially
    model.freeze_backbone()
    
    # Get data loaders
    print(f"\nLoading {args.task} dataset...")
    train_loader, val_loader = get_dataloaders(config, args.task)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    print(f"\n{'=' * 70}")
    print("Starting Training")
    print(f"{'=' * 70}")
    print(f"Total epochs: {args.epochs}")
    print(f"Freeze epochs: {args.freeze_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {config.BATCH_SIZE}\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Unfreeze after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            print("\n🔓 Unfreezing backbone for fine-tuning...")
            model.unfreeze_all()
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
            print(f"  New learning rate: {args.lr * 0.1}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint_dir = config.CHECKPOINT_DIR / 'pretrained'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'{args.model}_{args.task}.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                },
                'history': history
            }, checkpoint_path)
            
            print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    avg_epoch_time = total_time / args.epochs
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Total Time: {total_time:.1f}s ({avg_epoch_time:.1f}s/epoch)")
    
    # Log experiment
    print("\nLogging experiment...")
    logger = ExperimentLogger()
    
    experiment_data = {
        'model_name': f'{args.model}_pretrained',
        'task': args.task,
        'architecture': args.model,
        'pretrained': True,
        'freeze_epochs': args.freeze_epochs,
        'total_epochs': args.epochs,
        'train_acc': history['train_acc'][-1],
        'val_acc': best_val_acc,
        'test_acc': 0.0,
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][best_epoch-1],
        'test_loss': 0.0,
        'total_params': model.total_params,
        'trainable_params': model.trainable_params,
        'train_time_seconds': total_time,
        'avg_epoch_time': avg_epoch_time,
        'best_epoch': best_epoch,
        'learning_rate': args.lr,
        'batch_size': config.BATCH_SIZE,
        'notes': f'Transfer learning: {args.freeze_epochs} frozen epochs',
        'checkpoint_path': str(checkpoint_path)
    }
    
    logger.log_experiment(experiment_data)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
