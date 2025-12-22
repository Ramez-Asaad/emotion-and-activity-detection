"""
Model Performance Analysis
==========================

Comprehensive analysis of the trained emotion detection model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.emotion_model import EmotionModel
from configs.emotion_config import EmotionConfig
from data.emotion_dataset import FER2013Dataset
from data.transforms import get_test_transforms
from utils.evaluation import evaluate_model, calculate_per_class_accuracy


def analyze_model(model_path, test_dataset, config):
    """
    Comprehensive model analysis.
    
    Args:
        model_path: Path to model checkpoint
        test_dataset: Test dataset
        config: Configuration object
    """
    device = config.DEVICE
    
    # Load model
    print("=" * 70)
    print("Loading Model")
    print("=" * 70)
    model = EmotionModel()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from checkpoint")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"\nTraining Metrics:")
            print(f"  Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
            print(f"  Train Acc:  {metrics.get('train_acc', 'N/A'):.2f}%")
            print(f"  Val Loss:   {metrics.get('val_loss', 'N/A'):.4f}")
            print(f"  Val Acc:    {metrics.get('val_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    results = evaluate_model(
        model, test_loader, device, 
        class_names=config.CLASS_NAMES
    )
    
    # Print results
    print(f"\n✓ Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"✓ Precision: {results['precision']:.4f}")
    print(f"✓ Recall: {results['recall']:.4f}")
    print(f"✓ F1 Score: {results['f1_score']:.4f}")
    
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(results['classification_report'])
    
    # Per-class accuracy
    per_class_acc = calculate_per_class_accuracy(
        results['predictions'], 
        results['labels'], 
        config.NUM_CLASSES
    )
    
    print("\n" + "=" * 70)
    print("Per-Class Accuracy")
    print("=" * 70)
    for class_idx, acc in per_class_acc.items():
        class_name = config.CLASS_NAMES[class_idx]
        print(f"{class_name:10s}: {acc*100:5.2f}%")
    
    return results, per_class_acc


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: {save_path}")
    plt.close()


def plot_per_class_metrics(results, class_names, save_path):
    """Plot per-class precision, recall, F1."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        results['labels'], 
        results['predictions'],
        average=None
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='orange')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='green')
    
    ax.set_xlabel('Emotion Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics saved: {save_path}")
    plt.close()


def plot_error_analysis(results, class_names, save_path):
    """Analyze and visualize common misclassifications."""
    cm = results['confusion_matrix']
    
    # Find most common errors
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j],
                    'rate': cm[i, j] / cm[i].sum()
                })
    
    # Sort by count
    errors = sorted(errors, key=lambda x: x['count'], reverse=True)[:10]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"{e['true']} → {e['pred']}" for e in errors]
    counts = [e['count'] for e in errors]
    rates = [e['rate'] * 100 for e in errors]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color='coral')
    
    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Misclassification (True → Predicted)', fontsize=12)
    ax.set_ylabel('Number of Errors', fontsize=12)
    ax.set_title('Top 10 Misclassifications', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Error analysis saved: {save_path}")
    plt.close()


def main():
    """Main analysis function."""
    config = EmotionConfig()
    
    # Model path
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not model_path.exists():
        print("=" * 70)
        print("❌ Error: Model checkpoint not found!")
        print("=" * 70)
        print(f"\nExpected: {model_path}")
        print("\nTrain the model first:")
        print("  python scripts/train_emotion.py")
        return
    
    # Load test dataset
    print("=" * 70)
    print("Loading Test Dataset")
    print("=" * 70)
    
    test_transform = get_test_transforms(config)
    test_dataset = FER2013Dataset(
        root_dir=config.TEST_DIR,
        transform=test_transform
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Analyze
    results, per_class_acc = analyze_model(model_path, test_dataset, config)
    
    # Create results directory
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    plot_confusion_matrix(
        results['confusion_matrix'],
        config.CLASS_NAMES,
        results_dir / 'confusion_matrix.png'
    )
    
    plot_per_class_metrics(
        results,
        config.CLASS_NAMES,
        results_dir / 'per_class_metrics.png'
    )
    
    plot_error_analysis(
        results,
        config.CLASS_NAMES,
        results_dir / 'error_analysis.png'
    )
    
    # Save results to file
    results_file = results_dir / 'analysis_results.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Model Performance Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Accuracy: {results['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(results['classification_report'])
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("Per-Class Accuracy\n")
        f.write("=" * 70 + "\n")
        for class_idx, acc in per_class_acc.items():
            class_name = config.CLASS_NAMES[class_idx]
            f.write(f"{class_name:10s}: {acc*100:5.2f}%\n")
    
    print(f"✓ Results saved: {results_file}")
    
    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved in: {results_dir}")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - per_class_metrics.png")
    print("  - error_analysis.png")
    print("  - analysis_results.txt")


if __name__ == "__main__":
    main()
