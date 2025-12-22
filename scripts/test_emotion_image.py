"""
Test Emotion Detection on Single Image
======================================

Test the trained model on a single image file.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.emotion_model import EmotionModel
from configs.emotion_config import EmotionConfig


def test_single_image(image_path, model_path, device='cuda'):
    """
    Test emotion detection on a single image.
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model checkpoint
        device: Device to run on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = EmotionModel()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'metrics' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['metrics'].get('val_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        emotion_idx = predicted.item()
        emotion_label = model.get_emotion_name(emotion_idx)
        confidence_val = confidence.item()
        probs = probabilities.cpu().numpy()[0]
    
    # Display results
    print("=" * 70)
    print("Prediction Results")
    print("=" * 70)
    print(f"\nPredicted Emotion: {emotion_label}")
    print(f"Confidence: {confidence_val*100:.2f}%\n")
    
    print("All Probabilities:")
    print("-" * 40)
    emotions = list(EmotionModel.EMOTION_LABELS.values())
    for emotion, prob in zip(emotions, probs):
        bar = '█' * int(prob * 50)
        print(f"{emotion:10s} {prob*100:5.2f}% {bar}")
    print("=" * 70)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.set_title(f'Predicted: {emotion_label} ({confidence_val*100:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Show probabilities
    colors = ['red', 'green', 'purple', 'gold', 'blue', 'orange', 'gray']
    ax2.barh(emotions, probs, color=colors)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    for i, (emotion, prob) in enumerate(zip(emotions, probs)):
        ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('emotion_prediction.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: emotion_prediction.png")
    plt.show()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test emotion detection on an image')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to model checkpoint (default: best_model.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Get model path
    if args.model is None:
        config = EmotionConfig()
        model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    else:
        model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("\nTrain the model first:")
        print("  python scripts/train_emotion.py")
        return
    
    # Check image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    # Test
    test_single_image(image_path, model_path, args.device)


if __name__ == "__main__":
    main()
