# UCF101 Activity Recognition - Quick Setup Guide

## 📦 Dataset Preparation

### Option 1: Download UCF101 Dataset

The UCF101 dataset contains 101 action classes. For this project, we'll use 5 classes.

#### Recommended 5 Classes:
1. **Basketball** - Sports activity
2. **Biking** - Outdoor activity
3. **Typing** - Indoor activity
4. **WalkingWithDog** - Daily activity
5. **YoYo** - Recreational activity

### Option 2: Use Kaggle Dataset

```python
# Create a dataset downloader script
import kagglehub

# Download UCF101 subset
path = kagglehub.dataset_download("pevogam/ucf101")
print("Path to dataset files:", path)
```

---

## 🗂️ Dataset Structure

Organize your UCF101 dataset as follows:

```
datasets/UCF101/
├── train/
│   ├── Basketball/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── Biking/
│   ├── Typing/
│   ├── WalkingWithDog/
│   └── YoYo/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

---

## 🎬 Extracting Frames from Videos

If you have video files (.avi), use this script:

```python
import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, fps=1):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        fps: Frames per second to extract (1 = 1 frame/second)
    """
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f'frame_{saved_count:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path.name}")

# Example usage
video_dir = Path("UCF101_videos/Basketball")
output_dir = Path("datasets/UCF101/train/Basketball")

for video_file in video_dir.glob("*.avi"):
    extract_frames_from_video(video_file, output_dir, fps=1)
```

---

## ⚙️ Configuration

Update `configs/activity_config.py` with your selected classes:

```python
# Number of classes
NUM_CLASSES = 5

# Class names (update these!)
CLASS_NAMES = [
    'Basketball',
    'Biking', 
    'Typing',
    'WalkingWithDog',
    'YoYo'
]

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
```

---

## 🚀 Training

Once your dataset is ready:

```bash
# Train the activity recognition model
python scripts/train_activity.py
```

Expected output:
```
======================================================================
Loading UCF101 Dataset
======================================================================
Loading dataset using UCF101ImageFolder...
✓ Training samples: XXXX
✓ Validation samples: XXXX
✓ Activity classes: ['Basketball', 'Biking', 'Typing', 'WalkingWithDog', 'YoYo']

======================================================================
Initializing Model
======================================================================
✓ Model created: ActivityModel (5 activity classes)
✓ Total parameters: 981,767

======================================================================
Starting Training
======================================================================
Device: cuda
Epochs: 50
Batch size: 32
Learning rate: 0.001
```

---

## 📊 After Training

### 1. Analyze Performance
```bash
python scripts/analyze_activity.py
```

### 2. Test on Webcam (if applicable)
```bash
python scripts/test_activity_webcam.py
```

### 3. Compare Models
```bash
python scripts/compare_activity_models.py
```

---

## 💡 Quick Tips

1. **Start Small**: Test with 100 frames per class first
2. **Balance Classes**: Ensure similar number of frames per activity
3. **Frame Rate**: 1 FPS is usually sufficient for action recognition
4. **Augmentation**: Enabled by default in config
5. **GPU**: Make sure CUDA is available for faster training

---

## 🔧 Troubleshooting

### Dataset not found?
- Check paths in `configs/activity_config.py`
- Ensure folder structure matches expected format

### Out of memory?
- Reduce `BATCH_SIZE` in config
- Reduce image resolution

### Low accuracy?
- Train for more epochs
- Collect more diverse examples
- Try different activities (some are easier to classify)

---

## 📝 Recommended Activity Classes

**Easy to distinguish:**
- Basketball (ball + court)
- Biking (bike + motion)
- Typing (keyboard + sitting)

**Medium difficulty:**
- WalkingWithDog (person + dog)
- YoYo (hand motion)

**Alternative classes:**
- PushUps
- PullUps
- JumpingJack
- Drumming
- PlayingGuitar

Choose classes that are visually distinct for better results!

---

**Ready to train? Let's go! 🚀**
