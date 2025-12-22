"""
UCF101 Dataset Downloader and Organizer
========================================

Download and prepare UCF101 dataset for activity recognition.
"""

import os
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, fps=1, max_frames=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract (None = all)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        print(f"⚠ Invalid FPS for {video_path}, skipping...")
        cap.release()
        return 0
    
    frame_interval = max(1, int(video_fps / fps))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if max_frames and saved_count >= max_frames:
                break
            
            frame_path = output_dir / f'frame_{saved_count:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count


def organize_ucf101_from_videos(source_dir, dest_dir, selected_classes, 
                                 train_split=0.7, val_split=0.15, fps=1):
    """
    Organize UCF101 videos into train/val/test splits with frame extraction.
    
    Args:
        source_dir: Directory containing UCF101 videos organized by class
        dest_dir: Destination directory for organized dataset
        selected_classes: List of class names to include
        train_split: Proportion for training (default: 0.7)
        val_split: Proportion for validation (default: 0.15)
        fps: Frames per second to extract
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    print("=" * 70)
    print("UCF101 Dataset Organization")
    print("=" * 70)
    print(f"\nSource: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Selected classes: {selected_classes}")
    print(f"Splits: Train={train_split}, Val={val_split}, Test={1-train_split-val_split}")
    print(f"FPS: {fps}")
    
    for class_name in selected_classes:
        class_dir = source_dir / class_name
        
        if not class_dir.exists():
            print(f"\n⚠ Class directory not found: {class_dir}")
            continue
        
        # Get all video files
        video_files = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
        
        if not video_files:
            print(f"\n⚠ No videos found in {class_dir}")
            continue
        
        print(f"\n📂 Processing {class_name}: {len(video_files)} videos")
        
        # Shuffle and split
        import random
        random.shuffle(video_files)
        
        n_train = int(len(video_files) * train_split)
        n_val = int(len(video_files) * val_split)
        
        train_videos = video_files[:n_train]
        val_videos = video_files[n_train:n_train + n_val]
        test_videos = video_files[n_train + n_val:]
        
        # Process each split
        for split_name, videos in [('train', train_videos), 
                                    ('val', val_videos), 
                                    ('test', test_videos)]:
            if not videos:
                continue
            
            split_dir = dest_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {split_name}: {len(videos)} videos")
            
            total_frames = 0
            for video_file in tqdm(videos, desc=f"  Extracting {split_name}"):
                video_output_dir = split_dir / video_file.stem
                frames = extract_frames_from_video(video_file, video_output_dir, fps=fps)
                total_frames += frames
            
            print(f"    ✓ Extracted {total_frames} frames")
    
    print("\n" + "=" * 70)
    print("✅ Dataset organization complete!")
    print("=" * 70)
    print(f"\nDataset location: {dest_dir}")


def create_sample_dataset(dest_dir, num_samples_per_class=100):
    """
    Create a small sample dataset for testing (using dummy data).
    
    Args:
        dest_dir: Destination directory
        num_samples_per_class: Number of sample images per class
    """
    import numpy as np
    
    dest_dir = Path(dest_dir)
    
    # Sample classes
    classes = ['Basketball', 'Biking', 'Typing', 'WalkingWithDog', 'YoYo']
    
    print("=" * 70)
    print("Creating Sample UCF101 Dataset")
    print("=" * 70)
    print(f"\nClasses: {classes}")
    print(f"Samples per class: {num_samples_per_class}")
    
    for split in ['train', 'val', 'test']:
        n_samples = num_samples_per_class if split == 'train' else num_samples_per_class // 5
        
        print(f"\nCreating {split} set ({n_samples} samples per class)...")
        
        for class_name in classes:
            class_dir = dest_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(n_samples):
                # Create dummy image (random noise)
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_path = class_dir / f'sample_{i:04d}.jpg'
                cv2.imwrite(str(img_path), img)
    
    print("\n✅ Sample dataset created!")
    print(f"Location: {dest_dir}")
    print("\nNote: This is dummy data for testing. Replace with real UCF101 data.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare UCF101 dataset')
    parser.add_argument('--mode', type=str, choices=['organize', 'sample'], 
                       default='sample',
                       help='Mode: organize (from videos) or sample (create dummy data)')
    parser.add_argument('--source', type=str, default=None,
                       help='Source directory with UCF101 videos (for organize mode)')
    parser.add_argument('--dest', type=str, 
                       default='datasets/UCF101',
                       help='Destination directory')
    parser.add_argument('--classes', nargs='+', 
                       default=['Basketball', 'Biking', 'Typing', 'WalkingWithDog', 'YoYo'],
                       help='Classes to include')
    parser.add_argument('--fps', type=int, default=1,
                       help='Frames per second to extract')
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per class for sample mode')
    
    args = parser.parse_args()
    
    if args.mode == 'organize':
        if args.source is None:
            print("❌ Error: --source required for organize mode")
            print("\nUsage:")
            print("  python prepare_ucf101.py --mode organize --source path/to/UCF101/videos")
            return
        
        organize_ucf101_from_videos(
            source_dir=args.source,
            dest_dir=args.dest,
            selected_classes=args.classes,
            fps=args.fps
        )
    
    elif args.mode == 'sample':
        create_sample_dataset(
            dest_dir=args.dest,
            num_samples_per_class=args.samples
        )
    
    print("\n📝 Next steps:")
    print("  1. Verify dataset structure:")
    print(f"     ls {args.dest}")
    print("  2. Update configs/activity_config.py with your classes")
    print("  3. Train the model:")
    print("     python scripts/train_activity.py")


if __name__ == "__main__":
    main()
