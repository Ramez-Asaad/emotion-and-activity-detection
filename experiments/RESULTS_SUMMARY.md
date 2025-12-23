# Pretrained Model Experiments - Results Summary

**Generated:** 2025-12-23 02:52 AM

## Overview

This document summarizes the results from training 5 pretrained models on both emotion recognition (FER-2013) and activity recognition (UCF101) tasks using transfer learning.

## Key Findings

### 🏆 Best Models by Task

| Task | Model | Val Accuracy | Parameters | Training Time |
|------|-------|--------------|------------|---------------|
| **Emotion Recognition** | VGG-16 | **66.51%** | 134.3M | 25.7 min |
| **Activity Recognition** | MobileNet-V2 | **82.05%** | 2.2M | 0.5 min |

### Emotion Recognition Results (FER-2013)

| Rank | Model | Val Acc | Train Acc | Parameters | Train Time | Overfitting Gap |
|------|-------|---------|-----------|------------|------------|-----------------|
| 1 | **VGG-16** | 66.51% | 75.97% | 134.3M | 1541s (25.7m) | 9.46% |
| 2 | **MobileNet-V2** | 66.02% | 91.33% | 2.2M | 842s (14.0m) | 25.31% ⚠️ |

**Key Insights:**
- VGG-16 achieved the best validation accuracy with less overfitting
- MobileNet-V2 shows significant overfitting (25% gap) despite being lightweight
- Only 2 models completed training for emotion task
- Emotion recognition is more challenging (66% vs 82% for activity)

### Activity Recognition Results (UCF101)

| Rank | Model | Val Acc | Train Acc | Parameters | Train Time | Overfitting Gap |
|------|-------|---------|-----------|------------|------------|-----------------|
| 1 | **MobileNet-V2** | 82.05% | 100.00% | 2.2M | 31s | 17.95% |
| 2 | **ResNet-50** | 78.21% | 99.86% | 23.5M | 80s | 21.65% |
| 3 | **EfficientNet-B0** | 76.92% | 100.00% | 4.0M | 56s | 23.08% |
| 4 | **ResNet-18** | 75.64% | 100.00% | 11.2M | 35s | 24.36% |
| 5 | **VGG-16** | 71.79% | 100.00% | 134.3M | 178s | 28.21% ⚠️ |

**Key Insights:**
- MobileNet-V2 wins with best accuracy AND fastest training
- All models achieve 100% training accuracy (overfitting present)
- VGG-16 has worst performance despite being largest (134M params)
- Efficiency sweet spot: MobileNet-V2 (2.2M params, 31s training)

## Performance Analysis

### Accuracy vs Efficiency Trade-offs

**Best Overall:** MobileNet-V2
- Activity: 82.05% accuracy, 2.2M params, 31s training ✅
- Emotion: 66.02% accuracy, 2.2M params, 842s training
- **Winner for deployment:** Lightweight, fast, competitive accuracy

**Best for Emotion:** VGG-16
- 66.51% validation accuracy
- Less overfitting than MobileNet-V2
- Trade-off: 60x more parameters, 2x slower training

**Most Efficient:** MobileNet-V2
- Smallest model (2.2M params)
- Fastest training (31s for activity)
- Best accuracy-to-size ratio

**Least Efficient:** VGG-16
- Largest model (134M params)
- Slowest training (178s for activity)
- Worst activity accuracy (71.79%)

### Overfitting Analysis

All models show signs of overfitting (train acc >> val acc):

**Emotion Task:**
- VGG-16: 9.46% gap ✅ (best generalization)
- MobileNet-V2: 25.31% gap ⚠️ (significant overfitting)

**Activity Task:**
- MobileNet-V2: 17.95% gap (best)
- ResNet-50: 21.65% gap
- EfficientNet-B0: 23.08% gap
- ResNet-18: 24.36% gap
- VGG-16: 28.21% gap ⚠️ (worst)

**Recommendations:**
- Add more regularization (dropout, weight decay)
- Use data augmentation more aggressively
- Consider early stopping based on validation loss
- Reduce model capacity for activity task

## Transfer Learning Strategy

All models used the same transfer learning approach:

1. **Phase 1 (Epochs 1-5):** Backbone frozen, train classification head only
2. **Phase 2 (Epochs 6+):** Backbone unfrozen, fine-tune entire model with lower LR

**Results:**
- Strategy works well for activity recognition (75-82% accuracy)
- Emotion recognition more challenging (66% accuracy)
- All models converged successfully

## Recommendations

### For Production Deployment

**Activity Recognition:**
- **Primary:** MobileNet-V2 (best accuracy + efficiency)
- **Alternative:** ResNet-50 (if accuracy is critical, +4% but 10x larger)

**Emotion Recognition:**
- **Primary:** VGG-16 (best accuracy, less overfitting)
- **Alternative:** MobileNet-V2 (if speed/size matters, -0.5% accuracy)

### For Further Improvement

1. **Complete Emotion Training:**
   - Train ResNet-18, ResNet-50, EfficientNet-B0 on emotion task
   - May find better models than current VGG-16/MobileNet-V2

2. **Reduce Overfitting:**
   - Increase dropout rates
   - Add more data augmentation
   - Use mixup/cutmix techniques
   - Implement early stopping

3. **Optimize Hyperparameters:**
   - Tune learning rates per model
   - Experiment with different freeze epochs
   - Try different optimizers (AdamW, SGD with momentum)

4. **Ensemble Methods:**
   - Combine predictions from top 2-3 models
   - Could improve accuracy by 2-5%

## Visualizations

All comparison plots are available in `experiments/plots/`:

1. **1_accuracy_comparison.png** - Side-by-side accuracy comparison
2. **2_train_vs_val_accuracy.png** - Overfitting analysis
3. **3_efficiency_analysis.png** - Parameters vs accuracy scatter
4. **4_training_time.png** - Training time comparison
5. **5_comprehensive_analysis.png** - 4-panel comprehensive view
6. **6_best_models_summary.png** - Best model summary table

## Experiment Details

- **Datasets:** FER-2013 (emotion), UCF101 (activity)
- **Framework:** PyTorch with torchvision pretrained models
- **Transfer Learning:** ImageNet weights → task-specific fine-tuning
- **Training:** CUDA-accelerated on GPU
- **Tracking:** Automated logging to `experiments/results.csv`

## Next Steps

- [ ] Complete remaining emotion model training
- [ ] Implement overfitting reduction techniques
- [ ] Create ensemble models
- [ ] Deploy best models for real-time inference
- [ ] Document best practices in final report
