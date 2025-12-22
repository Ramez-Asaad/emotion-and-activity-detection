# Deep Learning Project: CNN for Image Classification

A modular deep learning project implementing custom CNN architectures for two image classification tasks:
1. **Facial Emotion Recognition** (FER-2013, 7 classes)
2. **Human Activity Recognition** (UCF101 subset, 5 classes)

## 🏗️ Project Structure

```
project/
├── models/              # Model architectures
│   ├── base_cnn.py     # Base CustomCNN architecture
│   ├── emotion_model.py    # FER-2013 specific model
│   └── activity_model.py   # UCF101 specific model
│
├── configs/            # Configuration files
│   ├── base_config.py      # Base configuration
│   ├── emotion_config.py   # FER-2013 config
│   └── activity_config.py  # UCF101 config
│
├── data/               # Data handling
│   └── transforms.py       # Data augmentation
│
├── utils/              # Utility functions
│   ├── training.py         # Training loop
│   ├── evaluation.py       # Metrics & evaluation
│   ├── checkpoint.py       # Model saving/loading
│   └── visualization.py    # Plotting functions
│
├── scripts/            # Training & testing scripts
│   ├── train_emotion.py    # Train FER-2013 model
│   ├── train_activity.py   # Train UCF101 model
│   └── check_gpu.py        # GPU verification
│
├── checkpoints/        # Saved model checkpoints
│   ├── emotion/
│   └── activity/
│
├── results/            # Training results & logs
│   ├── emotion/
│   └── activity/
│
└── datasets/           # Dataset storage
    ├── FER2013/
    └── UCF101/
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
.\venv\Scripts\activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python scripts/check_gpu.py
```

### 2. Prepare Datasets

#### FER-2013 (Emotion Recognition)
Organize your FER-2013 dataset:
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
│   └── (same structure)
└── test/
    └── (same structure)
```

#### UCF101 (Activity Recognition)
Organize your UCF101 subset:
```
datasets/UCF101/
├── train/
│   ├── Activity_1/
│   ├── Activity_2/
│   ├── Activity_3/
│   ├── Activity_4/
│   └── Activity_5/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### 3. Train Models

#### Train Emotion Recognition Model
```bash
python scripts/train_emotion.py
```

#### Train Activity Recognition Model
```bash
python scripts/train_activity.py
```

## 🎯 Model Architecture

### Base CNN Architecture
- **Input**: 3-channel RGB images (224×224)
- **5 Convolutional Blocks**:
  - Block 1: 3 → 32 channels + MaxPool
  - Block 2: 32 → 64 channels + MaxPool
  - Block 3: 64 → 128 channels + MaxPool
  - Block 4: 128 → 256 channels
  - Block 5: 256 → 256 channels
- **Global Average Pooling**
- **Fully Connected Layer**: 256 → num_classes
- **Total Parameters**: ~981K (0.98M)

### Task-Specific Models

#### EmotionModel (7 classes)
```python
from models.emotion_model import EmotionModel

model = EmotionModel()
# Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
```

#### ActivityModel (5 classes)
```python
from models.activity_model import ActivityModel

model = ActivityModel()
# Classes: Configurable based on your UCF101 subset
```

## ⚙️ Configuration

Each task has its own configuration file in `configs/`:

- **Base Config** (`base_config.py`): Common hyperparameters
- **Emotion Config** (`emotion_config.py`): FER-2013 specific settings
- **Activity Config** (`activity_config.py`): UCF101 specific settings

### Key Hyperparameters

```python
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
```

Modify these in the respective config files to customize training.

## 📊 Training Features

- ✅ **Automatic GPU detection** and usage
- ✅ **Data augmentation** (rotation, flip, color jitter)
- ✅ **Learning rate scheduling** (ReduceLROnPlateau)
- ✅ **Early stopping** to prevent overfitting
- ✅ **Checkpoint saving** (best model + periodic saves)
- ✅ **Training visualization** (loss/accuracy curves)
- ✅ **Progress bars** with tqdm

## 📈 Evaluation & Visualization

The project includes comprehensive evaluation utilities:

```python
from utils.evaluation import evaluate_model, print_evaluation_results
from utils.visualization import plot_confusion_matrix

# Evaluate model
results = evaluate_model(model, test_loader, device, class_names)

# Print results
print_evaluation_results(results)

# Plot confusion matrix
plot_confusion_matrix(results['confusion_matrix'], class_names)
```

## 💻 Usage Examples

### Training with Custom Configuration

```python
from configs.emotion_config import EmotionConfig

# Modify configuration
config = EmotionConfig()
config.BATCH_SIZE = 64
config.LEARNING_RATE = 0.0005
config.NUM_EPOCHS = 100

# Train with custom config
# (see scripts/train_emotion.py for full example)
```

### Loading a Trained Model

```python
from models.emotion_model import EmotionModel
from utils.checkpoint import load_checkpoint

model = EmotionModel()
model, _, epoch, metrics = load_checkpoint(
    model, 
    'checkpoints/emotion/best_model.pth',
    device='cuda'
)
```

### Making Predictions

```python
import torch
from models.emotion_model import EmotionModel

model = EmotionModel()
model.load_state_dict(torch.load('checkpoints/emotion/best_model.pth')['model_state_dict'])
model.eval()

# Predict emotion
predicted_class, emotion_label, probabilities = model.predict_emotion(image_tensor)
print(f"Predicted emotion: {emotion_label}")
```

## 🔧 System Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA 11.8
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ recommended
- **Storage**: ~5GB for datasets + checkpoints

## 📦 Dependencies

Core dependencies:
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Metrics and evaluation
- `tqdm` - Progress bars
- `Pillow` - Image processing

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🎓 Project Organization Benefits

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Shared utilities prevent code duplication
3. **Scalability**: Easy to add new tasks or models
4. **Maintainability**: Clear structure makes code easy to navigate
5. **Professional**: Industry-standard project organization

## 📝 Adding New Tasks

To add a new classification task:

1. Create a new model in `models/your_task_model.py`
2. Create a new config in `configs/your_task_config.py`
3. Create training script in `scripts/train_your_task.py`
4. Organize dataset in `datasets/YourDataset/`
5. Run training!

## 🐛 Troubleshooting

### GPU Not Detected
```bash
python scripts/check_gpu.py
```
If CUDA is not available, reinstall PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Not Found
Ensure your dataset is organized in the correct structure under `datasets/` folder.

### Out of Memory
Reduce `BATCH_SIZE` in the config file.

## 📄 License

This project is for educational and research purposes.

## 👥 Authors

Deep Learning Project - Semester 7 (Fall 2025-26)  
Alamein International University

---

**GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU  
**Last Updated**: December 2025
