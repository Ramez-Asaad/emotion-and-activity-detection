"""
Dual-Model Real-Time Detection
================================

Test both emotion and activity recognition models simultaneously on webcam.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import torch
import numpy as np
from torchvision import transforms

from models.emotion_model import EmotionModel
from models.activity_model import ActivityModel
from configs.emotion_config import EmotionConfig
from configs.activity_config import ActivityConfig


class DualModelDetector:
    """Dual-model detector for emotion and activity recognition."""
    
    def __init__(self, emotion_model_path, activity_model_path, device='cuda'):
        """Initialize both models."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load emotion model
        print("\nLoading Emotion Model...")
        self.emotion_model = EmotionModel()
        self.load_model(self.emotion_model, emotion_model_path, "Emotion")
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        
        # Load activity model
        print("\nLoading Activity Model...")
        self.activity_model = ActivityModel()
        self.load_model(self.activity_model, activity_model_path, "Activity")
        self.activity_model.to(self.device)
        self.activity_model.eval()
        
        # Labels
        self.emotions = list(EmotionModel.EMOTION_LABELS.values())
        self.activities = list(ActivityModel.ACTIVITY_LABELS.values())
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Face detector for emotion
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Colors
        self.emotion_colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 128, 0), 'Fear': (128, 0, 128),
            'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Surprise': (0, 255, 255),
            'Neutral': (128, 128, 128)
        }
        
        self.activity_colors = {
            'Walking': (0, 255, 0), 'Running': (0, 165, 255), 'Sitting': (255, 0, 0),
            'Standing': (128, 128, 128), 'Jumping': (0, 255, 255)
        }
    
    def load_model(self, model, model_path, model_name):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"  ✓ {model_name} Model: Val Acc = {metrics.get('val_acc', 'N/A'):.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print(f"  ✓ {model_name} Model loaded")
    
    def detect_emotion(self, face_img):
        """Detect emotion from face image."""
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        
        img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.emotion_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion_idx = predicted.item()
            emotion_label = self.emotions[emotion_idx]
            confidence_val = confidence.item()
            probs = probabilities.cpu().numpy()[0]
        
        return emotion_label, confidence_val, probs
    
    def detect_activity(self, frame):
        """Detect activity from frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.activity_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            activity_idx = predicted.item()
            activity_label = self.activities[activity_idx]
            confidence_val = confidence.item()
            probs = probabilities.cpu().numpy()[0]
        
        return activity_label, confidence_val, probs
    
    def draw_overlay(self, frame, emotion_data, activity_data, faces):
        """Draw both model predictions overlaid on the same frame."""
        h, w = frame.shape[:2]
        display = frame.copy()
        
        emotion_label, emotion_conf, emotion_probs = emotion_data
        activity_label, activity_conf, activity_probs = activity_data
        
        # Draw face detection boxes with emotion labels
        for (x, y, fw, fh) in faces:
            color = self.emotion_colors.get(emotion_label, (255, 255, 255))
            cv2.rectangle(display, (x, y), (x+fw, y+fh), color, 3)
            
            # Emotion label above face
            label = f"Emotion: {emotion_label} ({emotion_conf*100:.0f}%)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display, (x, y-35), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(display, label, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top-left: Activity detection
        act_color = self.activity_colors.get(activity_label, (255, 255, 255))
        
        # Semi-transparent background for activity
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
        
        # Activity text
        cv2.putText(display, "Activity:", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, activity_label, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, act_color, 3)
        cv2.putText(display, f"{activity_conf*100:.1f}%", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bottom: Probability bars for both models
        bar_start_y = h - 250
        bar_height = 18
        bar_width = 300
        
        # Emotion probabilities
        cv2.putText(display, "Emotions:", (20, bar_start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        for i, (emotion, prob) in enumerate(zip(self.emotions, emotion_probs)):
            y = bar_start_y + i * (bar_height + 3)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Background
            cv2.rectangle(display, (20, y), (20 + bar_width, y + bar_height),
                         (50, 50, 50), -1)
            # Filled
            filled = int(bar_width * prob)
            cv2.rectangle(display, (20, y), (20 + filled, y + bar_height),
                         color, -1)
            # Text
            cv2.putText(display, f"{emotion}: {prob*100:.0f}%", (25, y + 13),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Activity probabilities (right side)
        cv2.putText(display, "Activities:", (w - 320, bar_start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        for i, (activity, prob) in enumerate(zip(self.activities, activity_probs)):
            y = bar_start_y + i * (bar_height + 3)
            color = self.activity_colors.get(activity, (255, 255, 255))
            
            # Background
            cv2.rectangle(display, (w - 320, y), (w - 20, y + bar_height),
                         (50, 50, 50), -1)
            # Filled
            filled = int(bar_width * prob)
            cv2.rectangle(display, (w - 320, y), (w - 320 + filled, y + bar_height),
                         color, -1)
            # Text
            cv2.putText(display, f"{activity}: {prob*100:.0f}%", (w - 315, y + 13),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display
    
    def run(self, camera_id=0):
        """Run dual-model detection."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n" + "=" * 70)
        print("Dual-Model Real-Time Detection")
        print("=" * 70)
        print("\nBoth models running on the same frame:")
        print("  - Face boxes with emotion labels")
        print("  - Activity detection in top-left")
        print("  - Probability bars at bottom")
        print("\nControls:")
        print("  'q' or 'ESC' - Quit")
        print("  's' - Save screenshot")
        print("=" * 70 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces for emotion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # Get emotion prediction
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                face_roi = frame[y:y+fh, x:x+fw]
                emotion_data = self.detect_emotion(face_roi)
            else:
                emotion_data = ("No Face", 0.0, np.zeros(len(self.emotions)))
            
            # Get activity prediction
            activity_data = self.detect_activity(frame)
            
            # Draw overlay with both predictions
            display = self.draw_overlay(frame, emotion_data, activity_data, faces)
            
            # Show
            cv2.imshow('Dual-Model Detection', display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                filename = f"dual_detection_{frame_count}.jpg"
                cv2.imwrite(filename, display)
                print(f"✓ Screenshot saved: {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Detection stopped")


def main():
    """Main function."""
    emotion_config = EmotionConfig()
    activity_config = ActivityConfig()
    
    emotion_model_path = emotion_config.CHECKPOINT_DIR / 'best_model.pth'
    activity_model_path = activity_config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not emotion_model_path.exists():
        print(f"❌ Emotion model not found: {emotion_model_path}")
        return
    
    if not activity_model_path.exists():
        print(f"❌ Activity model not found: {activity_model_path}")
        return
    
    print("=" * 70)
    print("Dual-Model Detection - Webcam Test")
    print("=" * 70)
    print(f"\nEmotion Model: {emotion_model_path}")
    print(f"Activity Model: {activity_model_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    detector = DualModelDetector(
        emotion_model_path,
        activity_model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    try:
        detector.run(camera_id=0)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
