# Quick Start Guide

## 🚀 Getting Started with Your Modular CNN Project

### Step 1: Verify GPU Setup

```bash
python scripts/check_gpu.py
```

Expected output:
```
PyTorch version: 2.7.1+cu118
CUDA available: True
GPU device: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

---

### Step 2: Install Missing Dependencies

```bash
pip install matplotlib seaborn scikit-learn tqdm
```

---

### Step 3: Organize Your Datasets

#### For FER-2013 (Emotion Recognition):

1. Download FER-2013 dataset
2. Organize into this structure:

```
datasets/FER2013/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   ├── Surprise/
│   └── Neutral/
├── val/
│   └── (same 7 folders)
└── test/
    └── (same 7 folders)
```

#### For UCF101 (Activity Recognition):

1. Download UCF101 dataset
2. Extract frames from videos (or use pre-extracted frames)
3. Organize into this structure:

```
datasets/UCF101/
├── train/
│   ├── Activity_1/
│   ├── Activity_2/
│   ├── Activity_3/
│   ├── Activity_4/
│   └── Activity_5/
├── val/
│   └── (same 5 folders)
└── test/
    └── (same 5 folders)
```

---

### Step 4: Train Your First Model

#### Option A: Train Emotion Recognition Model

```bash
python scripts/train_emotion.py
```

#### Option B: Train Activity Recognition Model

```bash
python scripts/train_activity.py
```

---

### Step 5: Monitor Training

During training, you'll see:
- Progress bars for each epoch
- Training and validation metrics
- Automatic checkpoint saving
- Best model tracking

Example output:
```
Epoch 1/50 [Train]: 100%|████████| 500/500 [02:15<00:00, loss: 1.8234, acc: 45.23%]
Validating: 100%|████████| 100/100 [00:30<00:00, loss: 1.6543, acc: 52.10%]

Epoch 1/50 Summary:
  Train Loss: 1.8234 | Train Acc: 45.23%
  Val Loss:   1.6543 | Val Acc:   52.10%
✓ Best model saved! (val_acc: 52.1000)
```

---

### Step 6: View Results

After training, check:
- **Checkpoints**: `checkpoints/emotion/` or `checkpoints/activity/`
- **Training curves**: `results/emotion/training_history.png`
- **Best model**: `checkpoints/emotion/best_model.pth`

---

## 🎯 Quick Examples

### Example 1: Test Model Import

```python
from models.emotion_model import EmotionModel
from models.activity_model import ActivityModel

# Create models
emotion_model = EmotionModel()
activity_model = ActivityModel()

print(f"Emotion model classes: {emotion_model.num_classes}")
print(f"Activity model classes: {activity_model.num_classes}")
```

### Example 2: Load Configuration

```python
from configs.emotion_config import EmotionConfig

config = EmotionConfig()
config.display()
```

### Example 3: Check Data Transforms

```python
from configs.emotion_config import EmotionConfig
from data.transforms import get_train_transforms, get_val_transforms

config = EmotionConfig()
train_transform = get_train_transforms(config)
val_transform = get_val_transforms(config)

print("Train transforms:", train_transform)
print("Val transforms:", val_transform)
```

---

## 📁 Project Structure Overview

```
project/
├── models/          ← Your CNN architectures
├── configs/         ← Training configurations
├── data/            ← Data loading & transforms
├── utils/           ← Helper functions
├── scripts/         ← Training & testing scripts
├── checkpoints/     ← Saved models (created during training)
├── results/         ← Training plots & logs (created during training)
└── datasets/        ← Your datasets (you need to add these)
```

---

## ⚙️ Customizing Training

### Modify Hyperparameters

Edit `configs/emotion_config.py` or `configs/activity_config.py`:

```python
class EmotionConfig(BaseConfig):
    BATCH_SIZE = 64        # Increase for faster training (if GPU allows)
    LEARNING_RATE = 0.0005 # Lower for more stable training
    NUM_EPOCHS = 100       # Train longer
    
    # Enable/disable augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP = True
    ROTATION_DEGREES = 15
```

### Early Stopping

```python
class EmotionConfig(BaseConfig):
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
```

---

## 🐛 Common Issues

### Issue: "Dataset not found"
**Solution**: Make sure your dataset is in the correct folder structure under `datasets/`

### Issue: "CUDA out of memory"
**Solution**: Reduce `BATCH_SIZE` in the config file

### Issue: "No module named 'models'"
**Solution**: Make sure you're running scripts from the project root directory

---

## 📊 Next Steps

1. ✅ Train both models
2. ✅ Compare performance
3. ✅ Experiment with hyperparameters
4. ✅ Try different data augmentation
5. ✅ Visualize results with confusion matrices

---

## 💡 Tips

- **Start small**: Try training for 5-10 epochs first to verify everything works
- **Monitor GPU**: Use `nvidia-smi` to check GPU usage
- **Save often**: Checkpoints are saved automatically, but you can adjust frequency
- **Experiment**: This is a research baseline - try different configurations!

---

**Happy Training! 🎉**
