"""
Activity Recognition Configuration
==================================

UCF101 specific configuration.
"""

from .base_config import BaseConfig


class ActivityConfig(BaseConfig):
    """Configuration for UCF101 activity recognition task."""
    
    # Task-specific settings
    TASK_NAME = 'activity_recognition'
    NUM_CLASSES = 5
    # Daily human activity classes
    CLASS_NAMES = [
        'Walking',
        'Running',
        'Sitting',
        'Standing',
        'Jumping'
    ]
    
    # Dataset paths
    DATASET_NAME = 'UCF101'
    TRAIN_DIR = BaseConfig.DATASET_DIR / 'UCF101' / 'train'
    VAL_DIR = BaseConfig.DATASET_DIR / 'UCF101' / 'val'
    TEST_DIR = BaseConfig.DATASET_DIR / 'UCF101' / 'test'
    
    # Task-specific paths
    CHECKPOINT_DIR = BaseConfig.CHECKPOINT_DIR / 'activity'
    RESULTS_DIR = BaseConfig.RESULTS_DIR / 'activity'
    
    # Training hyperparameters (can override base config)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # Data augmentation settings
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP = True
    ROTATION_DEGREES = 10
    COLOR_JITTER = True
    
    # Video-specific settings (if using video frames)
    FRAMES_PER_VIDEO = 1  # Number of frames to sample per video
    FRAME_SAMPLING = 'uniform'  # 'uniform', 'random', or 'center'
    
    # Class weights (if dataset is imbalanced)
    USE_CLASS_WEIGHTS = False
    CLASS_WEIGHTS = None  # Will be computed from dataset if needed
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        cls.VAL_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def update_class_names(cls, class_names):
        """
        Update class names with actual UCF101 subset classes.
        
        Args:
            class_names (list): List of activity class names
        """
        cls.CLASS_NAMES = class_names
        cls.NUM_CLASSES = len(class_names)
