"""
Pretrained Model Comparison
============================

Compare custom CNN with pretrained models (ResNet18, VGG16, etc.)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import time

from models.emotion_model import EmotionModel
from configs.emotion_config import EmotionConfig
from data.emotion_dataset import FER2013Dataset
from data.transforms import get_train_transforms, get_val_transforms
from utils.evaluation import evaluate_model


class PretrainedEmotionModel(nn.Module):
    """Wrapper for pretrained models."""
    
    def __init__(self, model_name='resnet18', num_classes=7, pretrained=True):
        super(PretrainedEmotionModel, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_pretrained_model(model_name, config, test_loader):
    """
    Evaluate a pretrained model.
    
    Args:
        model_name: Name of pretrained model
        config: Configuration
        test_loader: Test data loader
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'=' * 70}")
    
    # Create model
    model = PretrainedEmotionModel(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    
    model.to(config.DEVICE)
    model.eval()
    
    # Count parameters
    params = count_parameters(model)
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Evaluate (just on pretrained features, no training)
    print("Evaluating pretrained features (no fine-tuning)...")
    
    start_time = time.time()
    results = evaluate_model(model, test_loader, config.DEVICE, config.CLASS_NAMES)
    eval_time = time.time() - start_time
    
    print(f"✓ Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"✓ Evaluation time: {eval_time:.2f}s")
    
    return {
        'model_name': model_name,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'parameters': params,
        'eval_time': eval_time
    }


def compare_models(config):
    """Compare custom model with pretrained models."""
    
    # Load test dataset
    print("=" * 70)
    print("Loading Test Dataset")
    print("=" * 70)
    
    test_transform = get_val_transforms(config)
    test_dataset = FER2013Dataset(
        root_dir=config.TEST_DIR,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"✓ Test samples: {len(test_dataset)}\n")
    
    # Results storage
    all_results = []
    
    # 1. Evaluate custom model
    print("=" * 70)
    print("Evaluating CUSTOM CNN")
    print("=" * 70)
    
    custom_model = EmotionModel()
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        if 'model_state_dict' in checkpoint:
            custom_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            custom_model.load_state_dict(checkpoint)
        
        custom_model.to(config.DEVICE)
        custom_model.eval()
        
        params = count_parameters(custom_model)
        print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
        
        start_time = time.time()
        results = evaluate_model(custom_model, test_loader, config.DEVICE, config.CLASS_NAMES)
        eval_time = time.time() - start_time
        
        print(f"✓ Test Accuracy: {results['accuracy']*100:.2f}%")
        print(f"✓ Evaluation time: {eval_time:.2f}s")
        
        all_results.append({
            'model_name': 'Custom CNN',
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'parameters': params,
            'eval_time': eval_time
        })
    else:
        print("⚠ Custom model not found, skipping...")
    
    # 2. Evaluate pretrained models
    pretrained_models = ['resnet18', 'resnet34', 'mobilenet_v2']
    
    for model_name in pretrained_models:
        try:
            result = evaluate_pretrained_model(model_name, config, test_loader)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Params':<12} {'Time (s)'}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['model_name']:<20} "
              f"{result['accuracy']*100:>10.2f}% "
              f"{result['f1_score']:>10.4f}  "
              f"{result['parameters']/1e6:>9.2f}M "
              f"{result['eval_time']:>8.2f}")
    
    print("=" * 70)
    
    # Find best model
    best_model = max(all_results, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best Model: {best_model['model_name']}")
    print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
    print(f"   Parameters: {best_model['parameters']/1e6:.2f}M")
    
    # Save results
    results_file = config.RESULTS_DIR / 'model_comparison.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Model Comparison Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Params':<12} {'Time (s)'}\n")
        f.write("-" * 70 + "\n")
        for result in all_results:
            f.write(f"{result['model_name']:<20} "
                   f"{result['accuracy']*100:>10.2f}% "
                   f"{result['f1_score']:>10.4f}  "
                   f"{result['parameters']/1e6:>9.2f}M "
                   f"{result['eval_time']:>8.2f}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Best Model: {best_model['model_name']}\n")
        f.write(f"Accuracy: {best_model['accuracy']*100:.2f}%\n")
    
    print(f"\n✓ Results saved: {results_file}")
    
    return all_results


def main():
    """Main comparison function."""
    config = EmotionConfig()
    config.create_dirs()
    
    print("=" * 70)
    print("PRETRAINED MODEL COMPARISON")
    print("=" * 70)
    print("\nThis will compare your custom CNN with:")
    print("  - ResNet18 (pretrained on ImageNet)")
    print("  - ResNet34 (pretrained on ImageNet)")
    print("  - MobileNetV2 (pretrained on ImageNet)")
    print("\nNote: Pretrained models are evaluated WITHOUT fine-tuning")
    print("      (just using pretrained features)")
    print("=" * 70)
    
    results = compare_models(config)
    
    print("\n" + "=" * 70)
    print("✅ Comparison Complete!")
    print("=" * 70)
    print("\nTo fine-tune a pretrained model, use:")
    print("  python scripts/train_pretrained.py --model resnet18")


if __name__ == "__main__":
    main()
