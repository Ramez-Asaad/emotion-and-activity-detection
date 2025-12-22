"""
Explore FER2013 Dataset Structure
"""

import os
from pathlib import Path

source = Path(r"C:\Users\Ramoz\.cache\kagglehub\datasets\msambare\fer2013\versions\1")

print("=" * 70)
print("FER2013 Dataset Structure")
print("=" * 70)
print(f"\nSource: {source}\n")

# List all items
for root, dirs, files in os.walk(source):
    level = root.replace(str(source), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    
    # Show first few files in each directory
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')
    
    if level > 2:  # Don't go too deep
        break
