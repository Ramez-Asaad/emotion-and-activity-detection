"""
Download and Organize Real UCF101 Dataset
==========================================

Download the actual UCF101 dataset and organize it for our 5 activity classes.
"""

import kagglehub
from pathlib import Path
import shutil
import cv2
import os
from tqdm import tqdm
import random


def download_ucf101():
    """Download UCF101 dataset from Kaggle."""
    print("=" * 70)
    print("Downloading UCF101 Dataset")
    print("=" * 70)
    print("\nThis will download ~6.5GB of data. Please wait...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")
        print(f"\n✓ Dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def explore_dataset_structure(dataset_path):
    """Explore the downloaded dataset structure."""
    print("\n" + "=" * 70)
    print("Exploring Dataset Structure")
    print("=" * 70)
    print(f"\nDataset location: {dataset_path}\n")
    
    # List all directories
    dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(dirs)} directories:")
    
    # Show first 20 directories
    for d in sorted(dirs)[:20]:
        num_files = len(list(d.glob('*')))
        print(f"  {d.name}: {num_files} files")
    
    if len(dirs) > 20:
        print(f"  ... and {len(dirs) - 20} more directories")
    
    return dirs


def find_activity_classes(dataset_path):
    """
    Find UCF101 classes that match our activities.
    
    Returns:
        dict: Mapping of our activities to UCF101 class folders
    """
    # Actual UCF101 class names from the dataset
    activity_mapping = {
        'Walking': ['WalkingWithDog'],  # UCF101 has WalkingWithDog
        'Running': ['Biking'],  # Using Biking as proxy for running motion
        'Sitting': ['BodyWeightSquats'],  # Using squats as sitting/standing motion
        'Standing': ['BoxingPunchingBag'],  # Using boxing as standing activity
        'Jumping': ['JumpRope', 'JumpingJack', 'LongJump', 'HighJump']  # Multiple jumping activities
    }
    
    found_mapping = {}
    
    # Check train folder for available classes
    train_dir = dataset_path / 'train'
    if not train_dir.exists():
        print(f"\n⚠ Train directory not found: {train_dir}")
        return found_mapping
    
    all_dirs = [d.name for d in train_dir.iterdir() if d.is_dir()]
    
    print("\n" + "=" * 70)
    print("Mapping Activities to UCF101 Classes")
    print("=" * 70)
    
    for activity, ucf_classes in activity_mapping.items():
        found_classes = []
        for ucf_class in ucf_classes:
            if ucf_class in all_dirs:
                found_classes.append(ucf_class)
        
        if found_classes:
            found_mapping[activity] = found_classes
            print(f"\n{activity}:")
            for cls in found_classes:
                class_dir = train_dir / cls
                num_files = len(list(class_dir.glob('*.avi'))) + len(list(class_dir.glob('*.mp4'))) + len(list(class_dir.glob('*.jpg')))
                print(f"  ✓ {cls}: {num_files} files")
        else:
            print(f"\n⚠ {activity}: No matching classes found")
    
    return found_mapping


def extract_frames_from_video(video_path, output_dir, max_frames=30, fps=1):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        cap.release()
        return 0
    
    frame_interval = max(1, int(video_fps / fps))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f'frame_{saved_count:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count


def organize_dataset(source_path, dest_path, activity_mapping):
    """
    Organize UCF101 dataset into our structure by copying existing files.
    
    Args:
        source_path: Path to downloaded UCF101 dataset
        dest_path: Destination path (datasets/UCF101)
        activity_mapping: Dict mapping our activities to UCF101 classes
    """
    dest_path = Path(dest_path)
    
    print("\n" + "=" * 70)
    print("Organizing Dataset")
    print("=" * 70)
    print(f"\nSource: {source_path}")
    print(f"Destination: {dest_path}")
    
    # Clear existing data
    for split in ['train', 'val', 'test']:
        split_dir = dest_path / split
        if split_dir.exists():
            print(f"\nRemoving old {split} data...")
            shutil.rmtree(split_dir)
    
    # Copy files for each split
    for split in ['train', 'val', 'test']:
        source_split = source_path / split
        if not source_split.exists():
            print(f"\n⚠ {split} folder not found, skipping...")
            continue
        
        print(f"\n📂 Processing {split.upper()} set...")
        
        for activity, ucf_classes in activity_mapping.items():
            dest_activity_dir = dest_path / split / activity
            dest_activity_dir.mkdir(parents=True, exist_ok=True)
            
            total_files = 0
            for ucf_class in ucf_classes:
                source_class_dir = source_split / ucf_class
                
                if not source_class_dir.exists():
                    continue
                
                # Copy all image files
                image_files = (list(source_class_dir.glob('*.jpg')) + 
                             list(source_class_dir.glob('*.png')) +
                             list(source_class_dir.glob('*.jpeg')))
                
                print(f"  {activity} ← {ucf_class}: {len(image_files)} images")
                
                for img_file in tqdm(image_files, desc=f"    Copying"):
                    # Create unique filename
                    new_name = f"{ucf_class}_{img_file.name}"
                    dest_file = dest_activity_dir / new_name
                    shutil.copy2(img_file, dest_file)
                    total_files += 1
            
            if total_files > 0:
                print(f"    ✓ Total for {activity}: {total_files} images")
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        split_dir = dest_path / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            total = 0
            for activity_dir in sorted(split_dir.iterdir()):
                if activity_dir.is_dir():
                    count = len(list(activity_dir.glob('*.jpg'))) + len(list(activity_dir.glob('*.png')))
                    print(f"  {activity_dir.name:10s}: {count:5d} images")
                    total += count
            print(f"  {'─' * 30}")
            print(f"  {'TOTAL':10s}: {total:5d} images")


def main():
    """Main function."""
    print("=" * 70)
    print("UCF101 Real Dataset Setup")
    print("=" * 70)
    
    # Download dataset
    dataset_path = download_ucf101()
    
    if not dataset_path or not dataset_path.exists():
        print("\n❌ Failed to download dataset")
        return
    
    # Explore structure
    explore_dataset_structure(dataset_path)
    
    # Find matching classes
    activity_mapping = find_activity_classes(dataset_path)
    
    if not activity_mapping:
        print("\n❌ No matching activity classes found")
        return
    
    # Organize dataset
    dest_path = Path("datasets/UCF101")
    organize_dataset(dataset_path, dest_path, activity_mapping)
    
    print("\n" + "=" * 70)
    print("✅ Dataset Setup Complete!")
    print("=" * 70)
    print(f"\nDataset location: {dest_path}")
    print("\nNext steps:")
    print("  1. Train the model:")
    print("     python scripts/train_activity.py")
    print("  2. Test on webcam:")
    print("     python scripts/test_activity_webcam.py")


if __name__ == "__main__":
    main()
