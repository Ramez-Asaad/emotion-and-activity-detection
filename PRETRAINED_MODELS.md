# Pretrained Models - Quick Start Guide

## 🎯 Overview

Train and compare 5 pretrained models on both emotion and activity recognition tasks:
- **ResNet-18** (~11.7M params)
- **ResNet-50** (~25.6M params)
- **MobileNet-V2** (~3.5M params)
- **EfficientNet-B0** (~5.3M params)
- **VGG-16** (~138.4M params)

---

## 🚀 Quick Start

### 1. Train a Pretrained Model

```bash
# Emotion recognition with ResNet-18
python scripts/train_pretrained.py --model resnet18 --task emotion --epochs 20

# Activity recognition with MobileNet-V2
python scripts/train_pretrained.py --model mobilenet_v2 --task activity --epochs 20
```

**Available models:**
- `resnet18`
- `resnet50`
- `mobilenet_v2`
- `efficientnet_b0`
- `vgg16`

**Available tasks:**
- `emotion` (FER-2013)
- `activity` (UCF101)

---

## 📊 Transfer Learning Strategy

### Phase 1: Freeze Backbone (Epochs 1-5)
- Load ImageNet pretrained weights
- **Freeze** all backbone layers
- Train only the final classification layer
- Learning rate: 0.001

### Phase 2: Fine-tune (Epochs 6-20)
- **Unfreeze** all layers
- Fine-tune entire network
- Learning rate: 0.0001 (10x lower)

This prevents catastrophic forgetting of ImageNet features!

---

## 📈 Experiment Tracking

All results are automatically logged to `experiments/results.csv`:

| model_name | task | val_acc | total_params | train_time |
|------------|------|---------|--------------|------------|
| CustomCNN | emotion | 64.22% | 981,767 | - |
| resnet18_pretrained | emotion | TBD | 11,689,512 | TBD |
| ... | ... | ... | ... | ... |

---

## 📓 View Results in Jupyter Notebook

The Jupyter notebook `experiments/model_comparison.ipynb` provides:
- ✅ Side-by-side model comparison
- ✅ Accuracy vs parameters plots
- ✅ Training time analysis
- ✅ Best model identification

**Note**: The notebook is in the `experiments/` directory which is gitignored. You can create it manually or run the cells from the implementation plan.

---

## 🎓 Training All Models

### Emotion Recognition (5 models × 20 epochs each)

```bash
python scripts/train_pretrained.py --model resnet18 --task emotion --epochs 20
python scripts/train_pretrained.py --model resnet50 --task emotion --epochs 20
python scripts/train_pretrained.py --model mobilenet_v2 --task emotion --epochs 20
python scripts/train_pretrained.py --model efficientnet_b0 --task emotion --epochs 20
python scripts/train_pretrained.py --model vgg16 --task emotion --epochs 20
```

### Activity Recognition (5 models × 20 epochs each)

```bash
python scripts/train_pretrained.py --model resnet18 --task activity --epochs 20
python scripts/train_pretrained.py --model resnet50 --task activity --epochs 20
python scripts/train_pretrained.py --model mobilenet_v2 --task activity --epochs 20
python scripts/train_pretrained.py --model efficientnet_b0 --task activity --epochs 20
python scripts/train_pretrained.py --model vgg16 --task activity --epochs 20
```

**Total**: 10 experiments (5 models × 2 tasks)

---

## 📁 File Structure

```
project/
├── models/
│   └── pretrained_models.py       # Model wrapper
├── scripts/
│   └── train_pretrained.py        # Training script
├── utils/
│   └── experiment_logger.py       # Experiment tracking
├── experiments/
│   ├── results.csv                # All experiment results
│   ├── summary.txt                # Text summary
│   ├── model_comparison.ipynb     # Jupyter notebook
│   └── plots/                     # Generated plots
│       ├── emotion_comparison.png
│       ├── activity_comparison.png
│       └── overall_comparison.png
└── checkpoints/
    ├── emotion/pretrained/        # Emotion model checkpoints
    └── activity/pretrained/       # Activity model checkpoints
```

---

## 🔍 Compare Results

### View CSV Results
```bash
cat experiments/results.csv
```

### View Summary
```bash
cat experiments/summary.txt
```

### Python Script
```python
from utils.experiment_logger import ExperimentLogger

logger = ExperimentLogger()

# Load all results
df = logger.load_results()
print(df)

# Get best model for emotion
best = logger.get_best_model('emotion')
print(f"Best emotion model: {best['model_name']} ({best['val_acc']:.2f}%)")

# Compare all models for activity
comparison = logger.compare_models('activity')
print(comparison)
```

---

## ⚙️ Advanced Options

### Custom Learning Rate
```bash
python scripts/train_pretrained.py --model resnet18 --task emotion --lr 0.0001
```

### Custom Freeze Epochs
```bash
python scripts/train_pretrained.py --model resnet50 --task activity --freeze-epochs 10
```

### CPU Training
```bash
python scripts/train_pretrained.py --model mobilenet_v2 --task emotion --device cpu
```

---

## 🎯 Expected Results

### Emotion Recognition (FER-2013)
- **Baseline (Custom CNN)**: 64.22% val acc
- **Expected improvement**: 5-15% with pretrained models
- **Best expected**: ResNet-50 or EfficientNet-B0 (~75-80%)

### Activity Recognition (UCF101)
- **Baseline (Custom CNN)**: 84.62% val acc
- **Expected improvement**: 3-10% with pretrained models
- **Best expected**: ResNet-50 (~90-95%)

---

## 🏆 Model Selection Guide

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **ResNet-18** | Balanced | Good accuracy, fast | Medium size |
| **ResNet-50** | Accuracy | Best accuracy | Slower, larger |
| **MobileNet-V2** | Speed | Very fast, small | Lower accuracy |
| **EfficientNet-B0** | Efficiency | Best accuracy/size ratio | Medium speed |
| **VGG-16** | Legacy | Simple architecture | Very large, slow |

---

## ✅ Next Steps

1. **Train all models** (10 experiments)
2. **Compare results** in Jupyter notebook
3. **Identify best model** for each task
4. **Deploy best models** for webcam testing
5. **Document findings** in final report

---

**Ready to start training!** 🚀
