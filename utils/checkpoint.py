"""
Checkpoint Utilities
===================

Model saving and loading utilities.
"""

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, config, filename='checkpoint.pth'):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics (loss, accuracy, etc.)
        config: Configuration object
        filename: Checkpoint filename
    """
    checkpoint_path = config.CHECKPOINT_DIR / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'task_name': config.TASK_NAME,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load model to
    
    Returns:
        tuple: (model, optimizer, epoch, metrics)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✓ Checkpoint loaded from epoch {epoch}")
    print(f"  Metrics: {metrics}")
    
    return model, optimizer, epoch, metrics


def save_best_model(model, metrics, config, metric_name='val_acc'):
    """
    Save model if it's the best so far.
    
    Args:
        model: PyTorch model
        metrics: Current metrics dictionary
        config: Configuration object
        metric_name: Metric to track for best model
    
    Returns:
        bool: True if model was saved
    """
    best_checkpoint = config.CHECKPOINT_DIR / 'best_model.pth'
    
    # Check if this is the best model
    current_metric = metrics.get(metric_name, 0)
    
    if best_checkpoint.exists():
        best_checkpoint_data = torch.load(best_checkpoint)
        best_metric = best_checkpoint_data['metrics'].get(metric_name, 0)
        
        if current_metric <= best_metric:
            return False
    
    # Save as best model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'task_name': config.TASK_NAME,
        }
    }
    
    torch.save(checkpoint, best_checkpoint)
    print(f"✓ Best model saved! ({metric_name}: {current_metric:.4f})")
    
    return True
