"""
Real-Time Emotion Detection
============================

Test the trained emotion recognition model on webcam feed.
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
from configs.emotion_config import EmotionConfig


class EmotionDetector:
    """Real-time emotion detector using webcam."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize emotion detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = EmotionModel()
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels
        self.emotions = list(EmotionModel.EMOTION_LABELS.values())
        
        # Face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Colors for each emotion (BGR format)
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Dark Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 255, 255), # Yellow
            'Neutral': (128, 128, 128) # Gray
        }
    
    def load_model(self, model_path):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from checkpoint")
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"  Validation accuracy: {metrics.get('val_acc', 'N/A'):.2f}%")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✓ Model loaded")
    
    def detect_emotion(self, face_img):
        """
        Detect emotion from face image.
        
        Args:
            face_img: Face image (numpy array)
        
        Returns:
            tuple: (emotion_label, confidence, probabilities)
        """
        # Convert to RGB
        if len(face_img.shape) == 2:  # Grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # RGBA
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
        
        # Transform image
        img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion_idx = predicted.item()
            emotion_label = self.emotions[emotion_idx]
            confidence_val = confidence.item()
            probs = probabilities.cpu().numpy()[0]
        
        return emotion_label, confidence_val, probs
    
    def draw_emotion_bar(self, frame, probabilities, x, y, w, h):
        """Draw emotion probability bars."""
        bar_height = 20
        bar_width = 200
        start_y = y + h + 10
        
        for i, (emotion, prob) in enumerate(zip(self.emotions, probabilities)):
            bar_y = start_y + i * (bar_height + 5)
            
            # Draw background
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Draw probability bar
            filled_width = int(bar_width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, bar_y), (x + filled_width, bar_y + bar_height),
                         color, -1)
            
            # Draw text
            text = f"{emotion}: {prob*100:.1f}%"
            cv2.putText(frame, text, (x + 5, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self, camera_id=0, show_probabilities=True):
        """
        Run real-time emotion detection.
        
        Args:
            camera_id: Camera device ID (default: 0)
            show_probabilities: Show probability bars for all emotions
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n" + "=" * 70)
        print("Real-Time Emotion Detection")
        print("=" * 70)
        print("\nControls:")
        print("  'q' or 'ESC' - Quit")
        print("  'p' - Toggle probability bars")
        print("  's' - Save screenshot")
        print("\nPress any key to start...")
        print("=" * 70 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence, probs = self.detect_emotion(face_roi)
                
                # Draw rectangle around face
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw emotion label
                label = f"{emotion} ({confidence*100:.1f}%)"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Background for text
                cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y),
                             color, -1)
                
                # Emotion text
                cv2.putText(frame, label, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw probability bars
                if show_probabilities:
                    self.draw_emotion_bar(frame, probs, x, y, w, h)
            
            # Draw info
            info_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))} | Faces: {len(faces)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('p'):  # Toggle probabilities
                show_probabilities = not show_probabilities
                print(f"Probability bars: {'ON' if show_probabilities else 'OFF'}")
            elif key == ord('s'):  # Save screenshot
                filename = f"emotion_detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Emotion detection stopped")


def main():
    """Main function."""
    config = EmotionConfig()
    
    # Path to best model
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not model_path.exists():
        print("=" * 70)
        print("❌ Error: Model checkpoint not found!")
        print("=" * 70)
        print(f"\nExpected location: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_emotion.py")
        print("\nOr specify a different checkpoint path.")
        return
    
    print("=" * 70)
    print("Emotion Detection - Webcam Test")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create detector
    detector = EmotionDetector(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run detection
    try:
        detector.run(camera_id=0, show_probabilities=True)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
