"""
FER-2013 Dataset Loader
=======================

Custom dataset class for loading FER-2013 emotion images.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    """
    FER-2013 dataset loader.
    
    Expected structure:
    root_dir/
        Angry/
            image1.jpg
            image2.jpg
            ...
        Disgust/
            ...
        Fear/
            ...
        Happy/
            ...
        Sad/
            ...
        Surprise/
            ...
        Neutral/
            ...
    """
    
    # Emotion class mapping
    EMOTION_LABELS = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6
    }
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize FER-2013 dataset.
        
        Args:
            root_dir (str): Root directory with emotion class folders
            transform: Torchvision transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get classes from directory structure
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load all samples
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Load all image files
            image_files = (list(class_dir.glob('*.jpg')) + 
                          list(class_dir.glob('*.png')) + 
                          list(class_dir.glob('*.jpeg')))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Returns:
            tuple: (image, label) where image is a tensor
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of samples per class."""
        distribution = {cls: 0 for cls in self.classes}
        
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        
        return distribution
