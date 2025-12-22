"""
Create Sample Human Activity Dataset
=====================================

Generate a small sample dataset for testing the activity recognition model.
Uses simple patterns to simulate different activities.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_activity_sample(activity_name, size=(224, 224)):
    """
    Create a sample image for an activity with SIGNIFICANT variation.
    
    Args:
        activity_name: Name of the activity
        size: Image size (height, width)
    
    Returns:
        numpy array: Generated image
    """
    # Start with random background
    base_color = np.random.randint(20, 80, 3)
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * base_color
    
    # Add significant random variation for each activity
    if activity_name == 'Walking':
        # Random vertical stripes with varying positions and widths
        num_stripes = np.random.randint(8, 15)
        for _ in range(num_stripes):
            x = np.random.randint(0, size[1])
            width = np.random.randint(5, 20)
            color = (np.random.randint(80, 180), 
                    np.random.randint(120, 200), 
                    np.random.randint(150, 230))
            cv2.rectangle(img, (x, 0), (x+width, size[0]), color, -1)
        
        # Random motion blur
        blur_size = np.random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
    elif activity_name == 'Running':
        # Random diagonal lines with varying angles
        num_lines = np.random.randint(15, 30)
        for _ in range(num_lines):
            angle = np.random.uniform(-45, 45)
            x1 = np.random.randint(0, size[1])
            y1 = np.random.randint(0, size[0])
            length = np.random.randint(50, 150)
            x2 = int(x1 + length * np.cos(np.radians(angle)))
            y2 = int(y1 + length * np.sin(np.radians(angle)))
            color = (np.random.randint(120, 200), 
                    np.random.randint(80, 150), 
                    np.random.randint(150, 230))
            thickness = np.random.randint(2, 6)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        # Stronger motion blur
        blur_size = np.random.choice([5, 7, 9])
        img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
    elif activity_name == 'Sitting':
        # Random horizontal bands with varying positions
        num_bands = np.random.randint(5, 12)
        for _ in range(num_bands):
            y = np.random.randint(size[0]//3, size[0])
            height = np.random.randint(10, 30)
            color = (np.random.randint(60, 140), 
                    np.random.randint(100, 180), 
                    np.random.randint(130, 200))
            cv2.rectangle(img, (0, y), (size[1], y+height), color, -1)
        
    elif activity_name == 'Standing':
        # Random vertical gradient with varying colors
        gradient_type = np.random.choice(['linear', 'radial'])
        
        if gradient_type == 'linear':
            for i in range(size[0]):
                ratio = i / size[0]
                color1 = np.random.randint(40, 100)
                color2 = np.random.randint(150, 220)
                color = int(color1 + ratio * (color2 - color1))
                cv2.line(img, (0, i), (size[1], i), 
                        (color, color, color+np.random.randint(-20, 20)), 1)
        else:
            # Radial gradient
            center = (size[1]//2 + np.random.randint(-50, 50), 
                     size[0]//2 + np.random.randint(-50, 50))
            max_radius = int(np.sqrt(size[0]**2 + size[1]**2) / 2)
            for r in range(0, max_radius, 5):
                color = int(50 + (r / max_radius) * 150)
                cv2.circle(img, center, r, (color, color, color+20), 2)
        
    elif activity_name == 'Jumping':
        # Random scattered shapes with varying sizes and positions
        num_shapes = np.random.randint(80, 150)
        for _ in range(num_shapes):
            x = np.random.randint(0, size[1])
            y = np.random.randint(0, size[0])
            shape_type = np.random.choice(['circle', 'rectangle'])
            color = (np.random.randint(150, 255), 
                    np.random.randint(100, 200), 
                    np.random.randint(80, 180))
            
            if shape_type == 'circle':
                radius = np.random.randint(2, 12)
                cv2.circle(img, (x, y), radius, color, -1)
            else:
                width = np.random.randint(5, 20)
                height = np.random.randint(5, 20)
                cv2.rectangle(img, (x, y), (x+width, y+height), color, -1)
    
    # Add SIGNIFICANT random noise
    noise_level = np.random.randint(30, 80)
    noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        center = (size[1]//2, size[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (size[1], size[0]))
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    return img


def create_sample_dataset(output_dir, samples_per_class=100):
    """
    Create a sample dataset for human activity recognition.
    
    Args:
        output_dir: Output directory for the dataset
        samples_per_class: Number of samples per activity class
    """
    output_dir = Path(output_dir)
    
    activities = ['Walking', 'Running', 'Sitting', 'Standing', 'Jumping']
    splits = {
        'train': int(samples_per_class * 0.7),
        'val': int(samples_per_class * 0.15),
        'test': int(samples_per_class * 0.15)
    }
    
    print("=" * 70)
    print("Creating Sample Human Activity Dataset")
    print("=" * 70)
    print(f"\nActivities: {activities}")
    print(f"Total samples per activity: {samples_per_class}")
    print(f"Splits: Train={splits['train']}, Val={splits['val']}, Test={splits['test']}")
    
    for split_name, n_samples in splits.items():
        print(f"\n📂 Creating {split_name} set...")
        
        for activity in activities:
            activity_dir = output_dir / split_name / activity
            activity_dir.mkdir(parents=True, exist_ok=True)
            
            for i in tqdm(range(n_samples), desc=f"  {activity}"):
                # Create sample image
                img = create_activity_sample(activity)
                
                # Save image
                img_path = activity_dir / f'{activity.lower()}_{i:04d}.jpg'
                cv2.imwrite(str(img_path), img)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    for split_name in ['train', 'val', 'test']:
        split_dir = output_dir / split_name
        print(f"\n{split_name.upper()}:")
        total = 0
        for activity in activities:
            activity_dir = split_dir / activity
            count = len(list(activity_dir.glob('*.jpg')))
            print(f"  {activity:10s}: {count:3d} images")
            total += count
        print(f"  {'─' * 20}")
        print(f"  {'TOTAL':10s}: {total:3d} images")
    
    print("\n" + "=" * 70)
    print("✅ Sample dataset created!")
    print("=" * 70)
    print(f"\nLocation: {output_dir}")
    print("\n⚠ Note: This is synthetic data for testing purposes.")
    print("For real activity recognition, you'll need actual video frames.")
    print("\nNext steps:")
    print("  1. Train the model:")
    print("     python scripts/train_activity.py")
    print("  2. Test on webcam:")
    print("     python scripts/test_activity_webcam.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample activity dataset')
    parser.add_argument('--output', type=str, 
                       default='datasets/UCF101',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per activity class')
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output, args.samples)
