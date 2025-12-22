"""
Emotion Recognition Model
========================

FER-2013 specific model wrapper for facial emotion recognition.
"""

import torch
import torch.nn as nn
from .base_cnn import CustomCNN


class EmotionModel(CustomCNN):
    """
    Model for Facial Emotion Recognition (FER-2013 dataset).
    
    Inherits from CustomCNN with 7 emotion classes:
    0: Angry
    1: Disgust
    2: Fear
    3: Happy
    4: Sad
    5: Surprise
    6: Neutral
    """
    
    # Emotion labels for FER-2013
    EMOTION_LABELS = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    def __init__(self):
        """Initialize emotion model with 7 classes."""
        super(EmotionModel, self).__init__(num_classes=7)
    
    def predict_emotion(self, x):
        """
        Predict emotion from input image.
        
        Args:
            x (torch.Tensor): Input image tensor (batch_size, 3, 224, 224)
        
        Returns:
            tuple: (predicted_class, emotion_label, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            # Get emotion labels
            if predicted_class.dim() == 0:  # Single prediction
                emotion_label = self.EMOTION_LABELS[predicted_class.item()]
            else:  # Batch prediction
                emotion_label = [self.EMOTION_LABELS[c.item()] for c in predicted_class]
            
            return predicted_class, emotion_label, probabilities
    
    @staticmethod
    def get_emotion_name(class_idx):
        """Get emotion name from class index."""
        return EmotionModel.EMOTION_LABELS.get(class_idx, 'Unknown')
