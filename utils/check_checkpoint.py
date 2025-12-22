"""
Check Activity Model Checkpoint
================================

Check the validation accuracy of the latest activity model checkpoint.
"""

import torch
from pathlib import Path

checkpoint_dir = Path("checkpoints/activity")

# Check best_model.pth
best_model = checkpoint_dir / "best_model.pth"
if best_model.exists():
    checkpoint = torch.load(best_model, map_location='cpu')
    print("=" * 70)
    print("BEST_MODEL.PTH")
    print("=" * 70)
    print(f"File: {best_model}")
    print(f"Size: {best_model.stat().st_size / (1024*1024):.2f} MB")
    print(f"Modified: {best_model.stat().st_mtime}")
    
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo metrics found in checkpoint")

# Check latest epoch checkpoint
latest_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)

if latest_checkpoints:
    print("\n" + "=" * 70)
    print("LATEST EPOCH CHECKPOINTS")
    print("=" * 70)
    
    for ckpt_path in latest_checkpoints[:3]:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"\nFile: {ckpt_path.name}")
        print(f"Size: {ckpt_path.stat().st_size / (1024*1024):.2f} MB")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
