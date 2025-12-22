"""
Emotion Recognition Configuration
=================================

FER-2013 specific configuration.
"""

from .base_config import BaseConfig


class EmotionConfig(BaseConfig):
    """Configuration for FER-2013 emotion recognition task."""
    
    # Task specific
    TASK_NAME = 'emotion_recognition'
    NUM_CLASSES = 7
    CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Dataset paths
    DATASET_NAME = 'FER2013'
    TRAIN_DIR = BaseConfig.DATASET_DIR / 'FER2013' / 'train'
    VAL_DIR = BaseConfig.DATASET_DIR / 'FER2013' / 'val'
    TEST_DIR = BaseConfig.DATASET_DIR / 'FER2013' / 'test'
    
    # Task-specific paths
    CHECKPOINT_DIR = BaseConfig.CHECKPOINT_DIR / 'emotion'
    RESULTS_DIR = BaseConfig.RESULTS_DIR / 'emotion'
    
    # Training hyperparameters (can override base config)
    BATCH_SIZE = 64  # FER-2013 images are smaller, can use larger batch
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # Data augmentation settings
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP = True
    ROTATION_DEGREES = 15
    COLOR_JITTER = True
    
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
