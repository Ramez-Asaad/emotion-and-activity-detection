"""
FER2013 Dataset Organizer
=========================

Script to organize the downloaded FER2013 dataset into the project structure.
"""

import os
import shutil
from pathlib import Path

# Source: Downloaded dataset location
SOURCE_DIR = Path(r"C:\Users\Ramoz\.cache\kagglehub\datasets\msambare\fer2013\versions\1")

# Destination: Project dataset location
PROJECT_ROOT = Path(__file__).parent.parent
DEST_DIR = PROJECT_ROOT / "datasets" / "FER2013"

def explore_dataset_structure():
    """Explore the downloaded dataset structure."""
    print("=" * 70)
    print("Exploring Downloaded Dataset Structure")
    print("=" * 70)
    print(f"\nSource directory: {SOURCE_DIR}")
    
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return False
    
    print("\nDirectory contents:")
    for item in SOURCE_DIR.rglob("*"):
        if item.is_dir():
            num_files = len(list(item.glob("*")))
            print(f"  📁 {item.relative_to(SOURCE_DIR)} ({num_files} items)")
        elif item.suffix in ['.jpg', '.png', '.jpeg']:
            print(f"  🖼️  {item.relative_to(SOURCE_DIR)}")
    
    return True

def organize_dataset():
    """Organize the FER2013 dataset into train/val/test structure."""
    print("\n" + "=" * 70)
    print("Organizing FER2013 Dataset")
    print("=" * 70)
    
    # Check source
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return False
    
    # Create destination directories
    print(f"\nDestination: {DEST_DIR}")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset has train/test folders already
    source_train = SOURCE_DIR / "train"
    source_test = SOURCE_DIR / "test"
    
    if not source_train.exists():
        print(f"\n❌ Training folder not found: {source_train}")
        return False
    
    if not source_test.exists():
        print(f"\n❌ Test folder not found: {source_test}")
        return False
    
    print("\n✓ Dataset has train/test split")
    
    # Copy train folder
    print("\n📂 Copying training data...")
    dest_train = DEST_DIR / "train"
    if dest_train.exists() and any(dest_train.iterdir()):
        print(f"  ⚠ {dest_train} already exists with data, skipping...")
    else:
        dest_train.mkdir(parents=True, exist_ok=True)
        # Copy each emotion folder
        for emotion_folder in source_train.iterdir():
            if emotion_folder.is_dir():
                dest_emotion = dest_train / emotion_folder.name
                if not dest_emotion.exists():
                    print(f"  Copying {emotion_folder.name}...")
                    shutil.copytree(emotion_folder, dest_emotion)
        print(f"  ✓ Training data copied to {dest_train}")
    
    # Copy test folder as validation
    print("\n📂 Copying test data as validation...")
    dest_val = DEST_DIR / "val"
    if dest_val.exists() and any(dest_val.iterdir()):
        print(f"  ⚠ {dest_val} already exists with data, skipping...")
    else:
        dest_val.mkdir(parents=True, exist_ok=True)
        # Copy each emotion folder
        for emotion_folder in source_test.iterdir():
            if emotion_folder.is_dir():
                dest_emotion = dest_val / emotion_folder.name
                if not dest_emotion.exists():
                    print(f"  Copying {emotion_folder.name}...")
                    shutil.copytree(emotion_folder, dest_emotion)
        print(f"  ✓ Validation data copied to {dest_val}")
    
    # Create test folder (copy from test for now)
    print("\n📂 Creating test set (copy of test data)...")
    dest_test = DEST_DIR / "test"
    if dest_test.exists() and any(dest_test.iterdir()):
        print(f"  ⚠ {dest_test} already exists with data, skipping...")
    else:
        dest_test.mkdir(parents=True, exist_ok=True)
        # Copy each emotion folder
        for emotion_folder in source_test.iterdir():
            if emotion_folder.is_dir():
                dest_emotion = dest_test / emotion_folder.name
                if not dest_emotion.exists():
                    print(f"  Copying {emotion_folder.name}...")
                    shutil.copytree(emotion_folder, dest_emotion)
        print(f"  ✓ Test data copied to {dest_test}")
    
    # Count samples
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        split_dir = DEST_DIR / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            total = 0
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                    print(f"  {class_dir.name}: {count:,} images")
                    total += count
            print(f"  {'─' * 30}")
            print(f"  TOTAL: {total:,} images")
    
    print("\n" + "=" * 70)
    print("✅ Dataset organization complete!")
    print("=" * 70)
    print(f"\nDataset location: {DEST_DIR}")
    print("\nNext steps:")
    print("  1. Verify the dataset:")
    print("     python scripts/train_emotion.py")
    print("  2. Start training:")
    print("     python scripts/train_emotion.py")
    
    return True

if __name__ == "__main__":
    # First explore the structure
    if explore_dataset_structure():
        print("\n")
        # Then organize it
        organize_dataset()
