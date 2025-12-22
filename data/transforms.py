"""
Data Transforms
==============

Data augmentation and preprocessing transforms.
"""

from torchvision import transforms


def get_train_transforms(config):
    """
    Get training data transforms with augmentation.
    
    Args:
        config: Configuration object with augmentation settings
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    transform_list = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]
    
    # Add augmentation if enabled
    if hasattr(config, 'USE_AUGMENTATION') and config.USE_AUGMENTATION:
        if hasattr(config, 'HORIZONTAL_FLIP') and config.HORIZONTAL_FLIP:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if hasattr(config, 'ROTATION_DEGREES') and config.ROTATION_DEGREES:
            transform_list.append(
                transforms.RandomRotation(degrees=config.ROTATION_DEGREES)
            )
        
        if hasattr(config, 'COLOR_JITTER') and config.COLOR_JITTER:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        config: Configuration object
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])


def get_test_transforms(config):
    """
    Get test data transforms (same as validation).
    
    Args:
        config: Configuration object
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return get_val_transforms(config)
