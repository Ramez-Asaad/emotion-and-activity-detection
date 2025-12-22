"""
Activity Recognition Model
=========================

UCF101 specific model wrapper for human activity recognition.
"""

import torch
import torch.nn as nn
from .base_cnn import CustomCNN


class ActivityModel(CustomCNN):
    """
    Model for Human Activity Recognition (UCF101 dataset subset).
    
    Inherits from CustomCNN with 5 activity classes.
    Note: Update ACTIVITY_LABELS based on your specific UCF101 subset.
    """
    
    # Daily human activity labels
    ACTIVITY_LABELS = {
        0: 'Walking',
        1: 'Running',
        2: 'Sitting',
        3: 'Standing',
        4: 'Jumping'
    }
    
    def __init__(self):
        """Initialize activity model with 5 classes."""
        super(ActivityModel, self).__init__(num_classes=5)
    
    def predict_activity(self, x):
        """
        Predict activity from input image/frame.
        
        Args:
            x (torch.Tensor): Input image tensor (batch_size, 3, 224, 224)
        
        Returns:
            tuple: (predicted_class, activity_label, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            # Get activity labels
            if predicted_class.dim() == 0:  # Single prediction
                activity_label = self.ACTIVITY_LABELS[predicted_class.item()]
            else:  # Batch prediction
                activity_label = [self.ACTIVITY_LABELS[c.item()] for c in predicted_class]
            
            return predicted_class, activity_label, probabilities
    
    @staticmethod
    def get_activity_name(class_idx):
        """Get activity name from class index."""
        return ActivityModel.ACTIVITY_LABELS.get(class_idx, 'Unknown')
    
    @classmethod
    def update_activity_labels(cls, labels_dict):
        """
        Update activity labels with actual UCF101 subset classes.
        
        Args:
            labels_dict (dict): Dictionary mapping class indices to activity names
        """
        cls.ACTIVITY_LABELS = labels_dict
