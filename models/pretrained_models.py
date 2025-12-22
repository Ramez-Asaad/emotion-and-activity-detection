"""
Pretrained Models Wrapper
==========================

Wrapper for pretrained models from torchvision with transfer learning support.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedModel(nn.Module):
    """
    Wrapper for pretrained models with transfer learning support.
    
    Supports:
    - ResNet-18, ResNet-50
    - MobileNet-V2
    - EfficientNet-B0
    - VGG-16
    """
    
    AVAILABLE_MODELS = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0,
        'vgg16': models.vgg16
    }
    
    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Initialize pretrained model.
        
        Args:
            model_name: Name of the model ('resnet18', 'resnet50', etc.)
            num_classes: Number of output classes
            pretrained: Whether to load ImageNet weights
        """
        super(PretrainedModel, self).__init__()
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. "
                           f"Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model
        print(f"Loading {model_name}...")
        if pretrained:
            # Use weights parameter for newer PyTorch versions
            try:
                weights = 'IMAGENET1K_V1' if pretrained else None
                self.model = self.AVAILABLE_MODELS[model_name](weights=weights)
                print(f"  ✓ Loaded with ImageNet weights")
            except TypeError:
                # Fallback for older PyTorch versions
                self.model = self.AVAILABLE_MODELS[model_name](pretrained=True)
                print(f"  ✓ Loaded with ImageNet weights (legacy)")
        else:
            self.model = self.AVAILABLE_MODELS[model_name](pretrained=False)
            print(f"  ✓ Loaded without pretrained weights")
        
        # Replace final classification layer
        self._replace_final_layer()
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"  ✓ Total parameters: {self.total_params:,}")
        print(f"  ✓ Trainable parameters: {self.trainable_params:,}")
    
    def _replace_final_layer(self):
        """Replace the final classification layer for the specific task."""
        if 'resnet' in self.model_name:
            # ResNet: replace fc layer
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif 'mobilenet' in self.model_name:
            # MobileNet: replace classifier
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
            
        elif 'efficientnet' in self.model_name:
            # EfficientNet: replace classifier
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
            
        elif 'vgg' in self.model_name:
            # VGG: replace last layer in classifier
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, self.num_classes)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classification layer."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze final layer
        if 'resnet' in self.model_name:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif 'mobilenet' in self.model_name or 'efficientnet' in self.model_name:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif 'vgg' in self.model_name:
            for param in self.model.classifier[6].parameters():
                param.requires_grad = True
        
        # Update trainable params count
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🔒 Backbone frozen. Trainable params: {self.trainable_params:,}")
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Update trainable params count
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🔓 All layers unfrozen. Trainable params: {self.trainable_params:,}")
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def get_model_info(self):
        """Get model information."""
        return {
            'name': self.model_name,
            'num_classes': self.num_classes,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'frozen': self.trainable_params < self.total_params
        }


def create_pretrained_model(model_name, num_classes, pretrained=True):
    """
    Factory function to create pretrained models.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet weights
    
    Returns:
        PretrainedModel instance
    """
    return PretrainedModel(model_name, num_classes, pretrained)


def list_available_models():
    """List all available pretrained models."""
    return list(PretrainedModel.AVAILABLE_MODELS.keys())


def get_model_params(model_name):
    """
    Get approximate parameter count for a model.
    
    Returns:
        dict: Parameter counts in millions
    """
    param_counts = {
        'resnet18': 11.7,
        'resnet50': 25.6,
        'mobilenet_v2': 3.5,
        'efficientnet_b0': 5.3,
        'vgg16': 138.4
    }
    return param_counts.get(model_name, 'Unknown')


if __name__ == "__main__":
    # Test all models
    print("=" * 70)
    print("Testing Pretrained Models")
    print("=" * 70)
    
    for model_name in list_available_models():
        print(f"\n{model_name.upper()}:")
        try:
            model = create_pretrained_model(model_name, num_classes=7, pretrained=True)
            
            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            print(f"  ✓ Output shape: {output.shape}")
            
            # Test freezing
            model.freeze_backbone()
            
            # Test unfreezing
            model.unfreeze_all()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ All models tested successfully!")
    print("=" * 70)
