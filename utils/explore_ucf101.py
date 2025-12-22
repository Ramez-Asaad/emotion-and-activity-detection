"""
Explore UCF101 Dataset Structure
=================================

Explore the actual structure of the downloaded UCF101 dataset.
"""

from pathlib import Path
import os

dataset_path = Path(r"C:\Users\Ramoz\.cache\kagglehub\datasets\matthewjansen\ucf101-action-recognition\versions\4")

print("=" * 70)
print("UCF101 Dataset Structure")
print("=" * 70)
print(f"\nDataset path: {dataset_path}\n")

# List all items in the dataset
for item in dataset_path.iterdir():
    if item.is_dir():
        print(f"📁 Directory: {item.name}")
        # List contents of each directory
        contents = list(item.iterdir())[:10]  # Show first 10 items
        for content in contents:
            if content.is_file():
                size_mb = content.stat().st_size / (1024 * 1024)
                print(f"   📄 {content.name} ({size_mb:.2f} MB)")
            elif content.is_dir():
                num_files = len(list(content.iterdir()))
                print(f"   📁 {content.name}/ ({num_files} items)")
        if len(list(item.iterdir())) > 10:
            print(f"   ... and {len(list(item.iterdir())) - 10} more items")
    else:
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"📄 File: {item.name} ({size_mb:.2f} MB)")

# Check train folder specifically
train_dir = dataset_path / "train"
if train_dir.exists():
    print("\n" + "=" * 70)
    print("Exploring TRAIN folder")
    print("=" * 70)
    
    train_items = list(train_dir.iterdir())
    print(f"\nFound {len(train_items)} items in train/")
    
    # Show first 20 items
    for item in sorted(train_items)[:20]:
        if item.is_dir():
            num_files = len(list(item.glob('*')))
            print(f"  📁 {item.name}: {num_files} files")
        else:
            print(f"  📄 {item.name}")
    
    if len(train_items) > 20:
        print(f"  ... and {len(train_items) - 20} more items")
