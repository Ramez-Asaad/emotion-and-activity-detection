# Human Activity Recognition - Quick Start

## ✅ What's Ready

Your activity recognition model is configured for **5 daily human activities**:
1. 🚶 **Walking**
2. 🏃 **Running**
3. 🪑 **Sitting**
4. 🧍 **Standing**
5. 🤸 **Jumping**

---

## 📦 Sample Dataset Created

A sample dataset has been created at `datasets/UCF101/`:
- **Train**: 35 samples per activity
- **Val**: 7 samples per activity
- **Test**: 8 samples per activity

⚠ **Note**: This is synthetic data for testing. For real activity recognition, you'll need actual video frames or images.

---

## 🚀 Quick Start

### 1. Train the Model

```bash
python scripts/train_activity.py
```

This will:
- Load the sample dataset
- Train for the configured epochs
- Save the best model to `checkpoints/activity/best_model.pth`
- Generate training history plots

### 2. Test on Webcam

```bash
python scripts/test_activity_webcam.py
```

Features:
- Real-time activity detection
- Temporal smoothing for stable predictions
- Probability bars for all activities
- Color-coded activity display

**Controls**:
- `q` or `ESC` - Quit
- `s` - Save screenshot
- `t` - Toggle temporal smoothing

---

## 📊 Expected Training Output

```
======================================================================
Loading UCF101 Dataset
======================================================================
Loading dataset using UCF101ImageFolder...
✓ Training samples: 175
✓ Validation samples: 35
✓ Activity classes: ['Walking', 'Running', 'Sitting', 'Standing', 'Jumping']

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

## 🎯 Using Real Data

### Option 1: Record Your Own Videos

1. **Record videos** of yourself performing each activity
2. **Extract frames**:
   ```python
   import cv2
   
   def extract_frames(video_path, output_dir, fps=1):
       cap = cv2.VideoCapture(video_path)
       # ... (see UCF101_SETUP.md for full code)
   ```
3. **Organize** into `datasets/UCF101/train/Walking/`, etc.

### Option 2: Use Public Datasets

- **UCF101**: Full action recognition dataset
- **Kinetics**: Large-scale human action dataset
- **HMDB51**: Human motion database

---

## 📈 After Training

### Analyze Performance
```bash
# Create analysis script for activities
python scripts/analyze_activity.py
```

### Compare Models
```bash
python scripts/compare_activity_models.py
```

---

## 💡 Tips for Better Results

1. **Collect Real Data**: The synthetic dataset is just for testing
2. **Balance Classes**: Ensure similar number of samples per activity
3. **Diverse Examples**: Include different people, angles, lighting
4. **Clear Actions**: Make sure activities are visually distinct
5. **Temporal Context**: Consider using multiple frames (video clips)

---

## 🔧 Configuration

Edit `configs/activity_config.py` to adjust:
- Number of epochs
- Batch size
- Learning rate
- Data augmentation settings

---

## 📝 Next Steps

1. ✅ Sample dataset created
2. ⏳ Train the model: `python scripts/train_activity.py`
3. ⏳ Test on webcam: `python scripts/test_activity_webcam.py`
4. ⏳ Collect real data for better accuracy
5. ⏳ Compare with emotion model

---

**Ready to train! 🚀**
