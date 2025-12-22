"""
Test Module Imports
===================

Quick test to verify all modules can be imported correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Testing Module Imports")
print("=" * 70)

try:
    # Test models
    from models.base_cnn import CustomCNN, count_parameters
    from models.emotion_model import EmotionModel
    from models.activity_model import ActivityModel
    print("✓ Models imported successfully")
    
    # Test configs
    from configs.base_config import BaseConfig
    from configs.emotion_config import EmotionConfig
    from configs.activity_config import ActivityConfig
    print("✓ Configs imported successfully")
    
    # Test data
    from data.transforms import get_train_transforms, get_val_transforms
    print("✓ Data transforms imported successfully")
    
    # Test utils
    from utils.training import Trainer, set_seed
    from utils.evaluation import evaluate_model
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from utils.visualization import plot_training_history
    print("✓ Utils imported successfully")
    
    print("\n" + "=" * 70)
    print("Creating Model Instances")
    print("=" * 70)
    
    # Create models
    emotion_model = EmotionModel()
    activity_model = ActivityModel()
    
    print(f"✓ EmotionModel created: {emotion_model.num_classes} classes")
    print(f"  Emotion labels: {list(EmotionModel.EMOTION_LABELS.values())}")
    print(f"  Parameters: {count_parameters(emotion_model):,}")
    
    print(f"\n✓ ActivityModel created: {activity_model.num_classes} classes")
    print(f"  Activity labels: {list(ActivityModel.ACTIVITY_LABELS.values())}")
    print(f"  Parameters: {count_parameters(activity_model):,}")
    
    print("\n" + "=" * 70)
    print("Testing Configurations")
    print("=" * 70)
    
    emotion_config = EmotionConfig()
    print(f"✓ EmotionConfig loaded")
    print(f"  Task: {emotion_config.TASK_NAME}")
    print(f"  Classes: {emotion_config.NUM_CLASSES}")
    print(f"  Batch size: {emotion_config.BATCH_SIZE}")
    print(f"  Learning rate: {emotion_config.LEARNING_RATE}")
    
    activity_config = ActivityConfig()
    print(f"\n✓ ActivityConfig loaded")
    print(f"  Task: {activity_config.TASK_NAME}")
    print(f"  Classes: {activity_config.NUM_CLASSES}")
    print(f"  Batch size: {activity_config.BATCH_SIZE}")
    print(f"  Learning rate: {activity_config.LEARNING_RATE}")
    
    print("\n" + "=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)
    print("\nYour project structure is ready to use!")
    print("Next steps:")
    print("  1. Organize your datasets in datasets/FER2013 and datasets/UCF101")
    print("  2. Run: python scripts/train_emotion.py")
    print("  3. Run: python scripts/train_activity.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
