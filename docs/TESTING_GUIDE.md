# Testing Your Emotion Detection Model

## Real-Time Webcam Testing

Test your trained emotion model on live webcam feed with face detection!

### Quick Start

```bash
python scripts/test_emotion_webcam.py
```

### Features

- **Real-time face detection** using Haar Cascade
- **Emotion prediction** with confidence scores
- **Color-coded bounding boxes** for each emotion
- **Probability bars** showing all emotion scores
- **FPS counter** and face count display
- **Screenshot capture** functionality

### Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit the application |
| `p` | Toggle probability bars on/off |
| `s` | Save screenshot |

### Emotion Colors

- 🔴 **Angry** - Red
- 🟢 **Happy** - Green
- 🔵 **Sad** - Blue
- 🟣 **Fear** - Purple
- 🟡 **Surprise** - Yellow
- 🟤 **Disgust** - Dark Green
- ⚪ **Neutral** - Gray

---

## Single Image Testing

Test on a single image file with detailed probability visualization.

### Usage

```bash
# Test on an image
python scripts/test_emotion_image.py path/to/image.jpg

# Specify custom model
python scripts/test_emotion_image.py image.jpg --model checkpoints/emotion/checkpoint_epoch_5.pth

# Use CPU instead of GPU
python scripts/test_emotion_image.py image.jpg --device cpu
```

### Output

- Console output with all emotion probabilities
- Visualization saved as `emotion_prediction.png`
- Interactive matplotlib window

---

## Requirements

Make sure you have trained a model first:

```bash
python scripts/train_emotion.py
```

The best model will be saved at:
```
checkpoints/emotion/best_model.pth
```

---

## Troubleshooting

### Webcam not opening

**Issue**: `Error: Could not open webcam`

**Solutions**:
1. Check if another application is using the webcam
2. Try different camera ID:
   ```python
   # In test_emotion_webcam.py, change camera_id
   detector.run(camera_id=1)  # Try 1, 2, etc.
   ```
3. Check camera permissions in Windows settings

### Face not detected

**Solutions**:
1. Ensure good lighting
2. Face the camera directly
3. Move closer to the camera
4. Adjust `minSize` in the code:
   ```python
   faces = self.face_cascade.detectMultiScale(
       gray, scaleFactor=1.1, minNeighbors=5, 
       minSize=(50, 50)  # Reduce from (100, 100)
   )
   ```

### Low FPS / Slow performance

**Solutions**:
1. Use GPU if available (automatic)
2. Reduce image resolution
3. Close other applications
4. Disable probability bars (press `p`)

### Model not found

**Error**: `Model checkpoint not found!`

**Solution**: Train the model first:
```bash
python scripts/train_emotion.py
```

---

## Tips for Best Results

1. **Lighting**: Ensure your face is well-lit
2. **Distance**: Stay 1-2 feet from the camera
3. **Expression**: Make clear facial expressions
4. **Background**: Use a simple, uncluttered background
5. **Camera**: Use a good quality webcam

---

## Understanding the Output

### Webcam Mode

```
┌─────────────────────────────┐
│  Happy (87.3%)              │  ← Predicted emotion + confidence
├─────────────────────────────┤
│  [Green bounding box]       │  ← Color-coded by emotion
└─────────────────────────────┘

Probability Bars (if enabled):
Angry:    12.5% ████
Disgust:   2.1% █
Fear:      5.3% ██
Happy:    87.3% ████████████████████████████████
Sad:       1.8% █
Surprise:  8.2% ████
Neutral:   2.8% █
```

### Image Mode

Console output:
```
Predicted Emotion: Happy
Confidence: 87.34%

All Probabilities:
----------------------------------------
Angry      12.50% ██████
Disgust     2.10% █
Fear        5.30% ██
Happy      87.34% ███████████████████████████████████████████
Sad         1.80% █
Surprise    8.20% ████
Neutral     2.80% █
```

Plus a saved visualization image showing:
- Original image with prediction
- Bar chart of all emotion probabilities

---

## Advanced Usage

### Custom Model Path

```python
from scripts.test_emotion_webcam import EmotionDetector

detector = EmotionDetector(
    model_path='path/to/your/model.pth',
    device='cuda'
)
detector.run(camera_id=0, show_probabilities=True)
```

### Batch Testing

Test multiple images:
```bash
for img in images/*.jpg; do
    python scripts/test_emotion_image.py "$img"
done
```

---

## Example Workflow

1. **Train the model**:
   ```bash
   python scripts/train_emotion.py
   ```

2. **Test on webcam**:
   ```bash
   python scripts/test_emotion_webcam.py
   ```

3. **Test on sample images**:
   ```bash
   python scripts/test_emotion_image.py datasets/FER2013/test/happy/image.jpg
   ```

4. **Save screenshots** of good predictions (press `s` in webcam mode)

5. **Analyze results** and iterate on training if needed

---

**Enjoy testing your emotion detection model!**
