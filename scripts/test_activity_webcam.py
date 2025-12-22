"""
Real-Time Human Activity Detection
===================================

Test the trained activity recognition model on webcam feed.
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
from models.activity_model import ActivityModel
from configs.activity_config import ActivityConfig


class ActivityDetector:
    """Real-time activity detector using webcam."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize activity detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = ActivityModel()
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Activity labels
        self.activities = list(ActivityModel.ACTIVITY_LABELS.values())
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Colors for each activity (BGR format)
        self.activity_colors = {
            'Walking': (0, 255, 0),      # Green
            'Running': (0, 165, 255),    # Orange
            'Sitting': (255, 0, 0),      # Blue
            'Standing': (128, 128, 128), # Gray
            'Jumping': (0, 255, 255)     # Yellow
        }
        
        # Frame buffer for temporal smoothing
        self.prediction_buffer = []
        self.buffer_size = 5
    
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
    
    def detect_activity(self, frame):
        """
        Detect activity from frame.
        
        Args:
            frame: Video frame (numpy array)
        
        Returns:
            tuple: (activity_label, confidence, probabilities)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform frame
        img_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            activity_idx = predicted.item()
            activity_label = self.activities[activity_idx]
            confidence_val = confidence.item()
            probs = probabilities.cpu().numpy()[0]
        
        return activity_label, confidence_val, probs
    
    def smooth_prediction(self, activity):
        """
        Smooth predictions using temporal buffer.
        
        Args:
            activity: Current predicted activity
        
        Returns:
            str: Smoothed activity prediction
        """
        self.prediction_buffer.append(activity)
        
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        # Return most common activity in buffer
        from collections import Counter
        most_common = Counter(self.prediction_buffer).most_common(1)
        return most_common[0][0] if most_common else activity
    
    def draw_activity_info(self, frame, activity, confidence, probabilities):
        """Draw activity information on frame."""
        h, w = frame.shape[:2]
        
        # Get color for activity
        color = self.activity_colors.get(activity, (255, 255, 255))
        
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw activity label
        label = f"Activity: {activity}"
        cv2.putText(frame, label, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw confidence
        conf_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(frame, conf_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw probability bars
        bar_y = 140
        bar_height = 30
        bar_max_width = 300
        
        for i, (act, prob) in enumerate(zip(self.activities, probabilities)):
            y = bar_y + i * (bar_height + 10)
            
            # Background bar
            cv2.rectangle(frame, (20, y), (20 + bar_max_width, y + bar_height),
                         (50, 50, 50), -1)
            
            # Filled bar
            filled_width = int(bar_max_width * prob)
            act_color = self.activity_colors.get(act, (255, 255, 255))
            cv2.rectangle(frame, (20, y), (20 + filled_width, y + bar_height),
                         act_color, -1)
            
            # Text
            text = f"{act}: {prob*100:.1f}%"
            cv2.putText(frame, text, (25, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0, use_smoothing=True):
        """
        Run real-time activity detection.
        
        Args:
            camera_id: Camera device ID (default: 0)
            use_smoothing: Use temporal smoothing for predictions
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n" + "=" * 70)
        print("Real-Time Activity Detection")
        print("=" * 70)
        print("\nControls:")
        print("  'q' or 'ESC' - Quit")
        print("  's' - Save screenshot")
        print("  't' - Toggle temporal smoothing")
        print("\nDetecting activities:")
        for activity in self.activities:
            print(f"  - {activity}")
        print("=" * 70 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect activity
            activity, confidence, probs = self.detect_activity(frame)
            
            # Apply temporal smoothing
            if use_smoothing:
                activity = self.smooth_prediction(activity)
            
            # Draw information
            frame = self.draw_activity_info(frame, activity, confidence, probs)
            
            # Draw FPS
            fps_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Activity Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save screenshot
                filename = f"activity_detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")
            elif key == ord('t'):  # Toggle smoothing
                use_smoothing = not use_smoothing
                print(f"Temporal smoothing: {'ON' if use_smoothing else 'OFF'}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Activity detection stopped")


def main():
    """Main function."""
    config = ActivityConfig()
    
    # Path to best model
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not model_path.exists():
        print("=" * 70)
        print("❌ Error: Model checkpoint not found!")
        print("=" * 70)
        print(f"\nExpected location: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_activity.py")
        print("\nOr create a sample dataset and train:")
        print("  python scripts/create_activity_samples.py")
        print("  python scripts/train_activity.py")
        return
    
    print("=" * 70)
    print("Activity Detection - Webcam Test")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create detector
    detector = ActivityDetector(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run detection
    try:
        detector.run(camera_id=0, use_smoothing=True)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
