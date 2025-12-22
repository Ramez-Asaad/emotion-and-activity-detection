# Dataset Preparation Guide

## 📦 Overview

This guide explains how to prepare your datasets for the two tasks:
1. **FER-2013** - Facial Emotion Recognition
2. **UCF101** - Human Activity Recognition

---

## 🎭 FER-2013 Dataset Preparation

### Dataset Structure

Organize your FER-2013 dataset in the following structure:

```
datasets/FER2013/
├── train/
│   ├── Angry/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── Disgust/
│   │   └── ...
│   ├── Fear/
│   │   └── ...
│   ├── Happy/
│   │   └── ...
│   ├── Sad/
│   │   └── ...
│   ├── Surprise/
│   │   └── ...
│   └── Neutral/
│       └── ...
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

### Steps to Prepare FER-2013

1. **Download FER-2013 Dataset**
   - Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
   - Or use the official FER-2013 dataset

2. **Extract and Organize**
   ```bash
   # Create directories
   mkdir -p datasets/FER2013/{train,val,test}/{Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral}
   ```

3. **Split the Data** (if not already split)
   - Training: 70-80%
   - Validation: 10-15%
   - Test: 10-15%

4. **Verify Structure**
   ```python
   python scripts/train_emotion.py
   ```
   The script will show class distribution if loaded correctly.

### Expected Output

When properly loaded, you should see:
```
Loading dataset using FER2013Dataset...
Loaded XXXXX samples from 7 classes
Classes: ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
✓ Training samples: XXXXX
✓ Validation samples: XXXXX

Training set class distribution:
  Angry: XXX samples
  Disgust: XXX samples
  Fear: XXX samples
  Happy: XXX samples
  Sad: XXX samples
  Surprise: XXX samples
  Neutral: XXX samples
```

---

## 🎬 UCF101 Dataset Preparation

### Three Options for UCF101

The UCF101 dataset loader supports three different formats:

#### **Option 1: Pre-extracted Frames** (RECOMMENDED ⭐)

Best for training speed and simplicity.

```
datasets/UCF101/
├── train/
│   ├── class1/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**How to extract frames:**
```python
import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """Extract frames from video at specified frame rate."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Example usage
extract_frames('video.avi', 'output_folder/', frame_rate=1)
```

#### **Option 2: Video Files**

Load directly from .avi or .mp4 files.

```
datasets/UCF101/
├── train/
│   ├── class1/
│   │   ├── video_001.avi
│   │   ├── video_002.avi
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**To use this option**, modify `scripts/train_activity.py`:
```python
train_dataset = UCF101Dataset(
    root_dir=config.TRAIN_DIR,
    transform=train_transform,
    frames_per_video=1,
    frame_sampling='center',  # or 'random', 'uniform'
    use_videos=True  # ← Set to True
)
```

#### **Option 3: Frames in Video Folders**

Frames organized by video.

```
datasets/UCF101/
├── train/
│   ├── class1/
│   │   ├── video1/
│   │   │   ├── frame_001.jpg
│   │   │   ├── frame_002.jpg
│   │   │   └── ...
│   │   ├── video2/
│   │   │   └── ...
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Steps to Prepare UCF101

1. **Download UCF101 Dataset**
   - Download from: https://www.crcv.ucf.edu/data/UCF101.php
   - Or use a subset with 5 classes

2. **Select Your 5 Classes**
   
   Example classes:
   - Basketball
   - Biking
   - Diving
   - GolfSwing
   - TennisSwing

3. **Extract Frames** (if using Option 1)
   ```bash
   # Use the extract_frames function above
   # Or use ffmpeg:
   ffmpeg -i video.avi -vf fps=1 frame_%04d.jpg
   ```

4. **Split the Data**
   - Training: 70%
   - Validation: 15%
   - Test: 15%

5. **Update Class Names**
   
   The dataset loader will automatically detect class names, but you can also update them in `models/activity_model.py`:
   ```python
   ACTIVITY_LABELS = {
       0: 'Basketball',
       1: 'Biking',
       2: 'Diving',
       3: 'GolfSwing',
       4: 'TennisSwing'
   }
   ```

### Frame Sampling Options

When using video files or video folders, you can choose how to sample frames:

- **`center`**: Extract the middle frame (fastest, good for single-frame models)
- **`random`**: Random frame each epoch (data augmentation)
- **`uniform`**: Uniformly spaced frames (for multi-frame models)

Example:
```python
train_dataset = UCF101Dataset(
    root_dir=config.TRAIN_DIR,
    transform=train_transform,
    frames_per_video=1,      # Number of frames to extract
    frame_sampling='center',  # Sampling strategy
    use_videos=False
)
```

---

## 🔍 Verification

### Check Dataset Loading

Run the test import script:
```bash
python test_imports.py
```

### Check Training Scripts

Try running the training scripts (they will show dataset info):
```bash
# For FER-2013
python scripts/train_emotion.py

# For UCF101
python scripts/train_activity.py
```

Both scripts will display:
- Number of samples loaded
- Class names
- Class distribution (for FER-2013)

---

## 📊 Dataset Statistics

### Recommended Sizes

**FER-2013:**
- Training: ~28,000 images
- Validation: ~3,500 images
- Test: ~3,500 images

**UCF101 (5 classes):**
- Training: ~500-1000 videos/frames per class
- Validation: ~100-200 videos/frames per class
- Test: ~100-200 videos/frames per class

---

## 🐛 Troubleshooting

### Issue: "Dataset not found"
**Solution**: Check that your dataset path matches the config:
```python
from configs.emotion_config import EmotionConfig
config = EmotionConfig()
print(f"Expected path: {config.TRAIN_DIR}")
```

### Issue: "No samples loaded"
**Solution**: 
- Check image file extensions (.jpg, .png, .jpeg)
- Verify folder structure matches expected format
- Ensure images are in class subfolders, not root

### Issue: "Class mismatch"
**Solution**: 
- For FER-2013: Use exact class names (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- For UCF101: Any class names work, they'll be auto-detected

### Issue: Video loading is slow
**Solution**: 
- Use Option 1 (pre-extracted frames) for faster training
- Reduce `frames_per_video` if using videos
- Use SSD storage for datasets

---

## 💡 Tips

1. **Start Small**: Test with a small subset first (100 images per class)
2. **Balance Classes**: Try to have similar numbers of samples per class
3. **Image Quality**: Ensure images are clear and properly labeled
4. **Augmentation**: The training scripts include data augmentation automatically
5. **Storage**: Keep original videos/images separate from processed datasets

---

## 📝 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Create dataset directories
mkdir -p datasets/FER2013/{train,val,test}
mkdir -p datasets/UCF101/{train,val,test}

# Verify GPU
python scripts/check_gpu.py

# Test dataset loading
python scripts/train_emotion.py  # Will show dataset info then exit if no data
python scripts/train_activity.py

# Start training (once datasets are ready)
python scripts/train_emotion.py
python scripts/train_activity.py
```

---

**Need Help?** Check the main [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md) for more information.
