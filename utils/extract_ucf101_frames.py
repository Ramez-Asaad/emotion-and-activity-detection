"""
Extract Frames from UCF101 Videos
==================================

Extract frames from the downloaded UCF101 video files.
"""

import cv2
from pathlib import Path
import shutil
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, max_frames=30, fps=1):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  ⚠ Could not open: {video_path.name}")
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


def extract_all_frames():
    """Extract frames from all UCF101 videos."""
    
    source_path = Path(r"C:\Users\Ramoz\.cache\kagglehub\datasets\matthewjansen\ucf101-action-recognition\versions\4")
    dest_path = Path("datasets/UCF101")
    
    # Activity mapping
    activity_mapping = {
        'Walking': ['WalkingWithDog'],
        'Running': ['Biking'],
        'Sitting': ['BodyWeightSquats'],
        'Standing': ['BoxingPunchingBag'],
        'Jumping': ['JumpRope', 'JumpingJack', 'LongJump', 'HighJump']
    }
    
    print("=" * 70)
    print("Extracting Frames from UCF101 Videos")
    print("=" * 70)
    print(f"\nSource: {source_path}")
    print(f"Destination: {dest_path}")
    print(f"Frames per video: 30")
    
    # Clear existing data
    for split in ['train', 'val', 'test']:
        split_dir = dest_path / split
        if split_dir.exists():
            print(f"\nRemoving old {split} data...")
            shutil.rmtree(split_dir)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        source_split = source_path / split
        if not source_split.exists():
            print(f"\n⚠ {split} folder not found")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Processing {split.upper()} Set")
        print(f"{'=' * 70}")
        
        for activity, ucf_classes in activity_mapping.items():
            dest_activity_dir = dest_path / split / activity
            dest_activity_dir.mkdir(parents=True, exist_ok=True)
            
            total_frames = 0
            total_videos = 0
            
            for ucf_class in ucf_classes:
                source_class_dir = source_split / ucf_class
                
                if not source_class_dir.exists():
                    continue
                
                # Get all video files
                video_files = list(source_class_dir.glob('*.avi')) + list(source_class_dir.glob('*.mp4'))
                
                if not video_files:
                    continue
                
                print(f"\n{activity} ← {ucf_class}: {len(video_files)} videos")
                
                for video_file in tqdm(video_files, desc=f"  Extracting"):
                    # Create output directory for this video
                    video_output_dir = dest_activity_dir / f"{ucf_class}_{video_file.stem}"
                    
                    # Extract frames
                    frames = extract_frames_from_video(video_file, video_output_dir, max_frames=30, fps=1)
                    
                    if frames > 0:
                        total_frames += frames
                        total_videos += 1
                
                if total_videos > 0:
                    print(f"  ✓ Extracted {total_frames} frames from {total_videos} videos")
    
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
                    # Count all frames in all video folders
                    count = 0
                    for video_folder in activity_dir.iterdir():
                        if video_folder.is_dir():
                            count += len(list(video_folder.glob('*.jpg')))
                    print(f"  {activity_dir.name:10s}: {count:5d} frames")
                    total += count
            print(f"  {'─' * 30}")
            print(f"  {'TOTAL':10s}: {total:5d} frames")
    
    print("\n" + "=" * 70)
    print("✅ Frame Extraction Complete!")
    print("=" * 70)
    print(f"\nDataset location: {dest_path}")
    print("\nNext steps:")
    print("  python scripts/train_activity.py")


if __name__ == "__main__":
    extract_all_frames()
