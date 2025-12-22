# Model Analysis & Comparison Guide

## 📊 Analyzing Your Model Performance

### Run Complete Analysis

```bash
python scripts/analyze_model.py
```

This will generate:
- **Confusion Matrix** - See which emotions are confused
- **Per-Class Metrics** - Precision, Recall, F1 for each emotion
- **Error Analysis** - Top 10 misclassifications
- **Detailed Report** - Text file with all metrics

### Output Files

All results saved in `results/emotion/`:
- `confusion_matrix.png` - Normalized confusion matrix heatmap
- `per_class_metrics.png` - Bar charts for each emotion
- `error_analysis.png` - Most common mistakes
- `analysis_results.txt` - Complete text report

---

## 🔬 Comparing with Pretrained Models

### Run Model Comparison

```bash
python scripts/compare_models.py
```

This compares your custom CNN against:
- **ResNet18** (~11M parameters)
- **ResNet34** (~21M parameters)
- **MobileNetV2** (~3.5M parameters)

### What It Shows

- Accuracy comparison
- F1-Score comparison
- Parameter count (model size)
- Inference time

### Example Output

```
MODEL COMPARISON
======================================================================
Model                Accuracy     F1-Score     Params       Time (s)
----------------------------------------------------------------------
Custom CNN             64.22%       0.6234      0.98M        12.45
ResNet18               58.31%       0.5621     11.18M        15.23
ResNet34               61.45%       0.5892     21.28M        18.67
MobileNetV2            55.12%       0.5234      3.50M        10.89
======================================================================

🏆 Best Model: Custom CNN
   Accuracy: 64.22%
   Parameters: 0.98M
```

---

## 📈 Understanding the Results

### Confusion Matrix

Shows which emotions are confused with each other:
- **Diagonal** = Correct predictions
- **Off-diagonal** = Misclassifications
- **Darker blue** = Higher proportion

Common confusions:
- Fear ↔ Surprise (similar facial expressions)
- Sad ↔ Neutral (subtle differences)
- Angry ↔ Disgust (both negative emotions)

### Per-Class Metrics

- **Precision**: Of all predicted X, how many were actually X?
- **Recall**: Of all actual X, how many did we find?
- **F1-Score**: Harmonic mean of precision and recall

### Error Analysis

Top misclassifications help you understand:
- Which emotions are hardest to distinguish
- Where to focus data collection efforts
- Potential model improvements

---

## 🎯 Improving Performance

### Based on Analysis Results

1. **Low accuracy on specific emotion?**
   - Collect more training data for that emotion
   - Check data quality for that class
   - Consider class weights in training

2. **High confusion between two emotions?**
   - Add more distinctive examples
   - Use data augmentation
   - Consider ensemble methods

3. **Overall low accuracy?**
   - Train for more epochs
   - Try different learning rates
   - Use pretrained model as baseline
   - Increase model capacity

### Quick Improvements

```python
# In configs/emotion_config.py

# 1. Train longer
NUM_EPOCHS = 50  # Increase from 10

# 2. Adjust learning rate
LEARNING_RATE = 0.0005  # Try lower

# 3. Use class weights (if imbalanced)
USE_CLASS_WEIGHTS = True

# 4. More augmentation
ROTATION_DEGREES = 20  # Increase from 15
```

---

## 🚀 Next Steps

### 1. Analyze Current Model
```bash
python scripts/analyze_model.py
```

### 2. Compare with Pretrained
```bash
python scripts/compare_models.py
```

### 3. Based on Results

**If custom model is best:**
- Continue training for more epochs
- Fine-tune hyperparameters
- Add more data

**If pretrained model is better:**
- Consider fine-tuning ResNet/MobileNet
- Transfer learning approach
- Ensemble custom + pretrained

### 4. Iterate

1. Analyze results
2. Identify weaknesses
3. Make improvements
4. Retrain
5. Repeat

---

## 💡 Tips

- **Run analysis after every training** to track progress
- **Save all results** for comparison
- **Focus on F1-score** for imbalanced datasets
- **Check per-class metrics** to find weak spots
- **Compare inference time** if deploying to production

---

## 📝 Example Workflow

```bash
# 1. Train model
python scripts/train_emotion.py

# 2. Analyze performance
python scripts/analyze_model.py

# 3. Test on webcam
python scripts/test_emotion_webcam.py

# 4. Compare with pretrained
python scripts/compare_models.py

# 5. Make improvements based on analysis

# 6. Retrain and repeat
```

---

**Happy analyzing! 📊**
