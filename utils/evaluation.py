"""
Evaluation Utilities
===================

Metrics calculation and evaluation functions.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    if class_names is not None:
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            digits=4
        )
    else:
        report = classification_report(all_labels, all_predictions, digits=4)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary from evaluate_model()
    """
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(results['classification_report'])
    print("=" * 70)


def calculate_per_class_accuracy(predictions, labels, num_classes):
    """
    Calculate per-class accuracy.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        dict: Per-class accuracy
    """
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.sum() > 0:
            class_predictions = predictions[class_mask]
            class_labels = labels[class_mask]
            acc = (class_predictions == class_labels).sum() / len(class_labels)
            per_class_acc[class_idx] = acc
        else:
            per_class_acc[class_idx] = 0.0
    
    return per_class_acc
