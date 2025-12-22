"""
Experiment Logger
=================

Utilities for logging and tracking model experiments.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class ExperimentLogger:
    """Logger for tracking model experiments."""
    
    def __init__(self, results_file='experiments/results.csv'):
        """Initialize experiment logger."""
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV if it doesn't exist
        if not self.results_file.exists():
            self._create_csv()
    
    def _create_csv(self):
        """Create CSV file with headers."""
        headers = [
            'timestamp', 'model_name', 'task', 'architecture',
            'pretrained', 'freeze_epochs', 'total_epochs',
            'train_acc', 'val_acc', 'test_acc',
            'train_loss', 'val_loss', 'test_loss',
            'total_params', 'trainable_params',
            'train_time_seconds', 'avg_epoch_time',
            'best_epoch', 'learning_rate', 'batch_size',
            'notes', 'checkpoint_path'
        ]
        
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_experiment(self, experiment_data):
        """
        Log experiment results.
        
        Args:
            experiment_data: Dictionary containing experiment results
        """
        # Add timestamp
        experiment_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure all required fields exist
        required_fields = [
            'model_name', 'task', 'architecture', 'pretrained',
            'train_acc', 'val_acc', 'total_params'
        ]
        
        for field in required_fields:
            if field not in experiment_data:
                experiment_data[field] = 'N/A'
        
        # Read existing data
        df = self.load_results()
        
        # Append new row
        new_row = pd.DataFrame([experiment_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save
        df.to_csv(self.results_file, index=False)
        
        print(f"\n✓ Experiment logged to {self.results_file}")
    
    def load_results(self):
        """Load all experiment results."""
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame()
    
    def get_best_model(self, task, metric='val_acc'):
        """
        Get best performing model for a task.
        
        Args:
            task: Task name ('emotion' or 'activity')
            metric: Metric to compare ('val_acc', 'test_acc', etc.)
        
        Returns:
            dict: Best model information
        """
        df = self.load_results()
        
        if df.empty:
            return None
        
        # Filter by task
        task_df = df[df['task'] == task]
        
        if task_df.empty:
            return None
        
        # Get best
        best_idx = task_df[metric].idxmax()
        return task_df.loc[best_idx].to_dict()
    
    def compare_models(self, task):
        """
        Compare all models for a task.
        
        Args:
            task: Task name
        
        Returns:
            DataFrame: Comparison table
        """
        df = self.load_results()
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter by task
        task_df = df[df['task'] == task]
        
        # Select relevant columns
        columns = [
            'model_name', 'architecture', 'pretrained',
            'train_acc', 'val_acc', 'test_acc',
            'total_params', 'train_time_seconds'
        ]
        
        comparison = task_df[columns].copy()
        
        # Sort by val_acc
        comparison = comparison.sort_values('val_acc', ascending=False)
        
        return comparison
    
    def export_summary(self, output_file='experiments/summary.txt'):
        """Export summary of all experiments."""
        df = self.load_results()
        
        if df.empty:
            print("No experiments logged yet.")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary by task
            for task in df['task'].unique():
                f.write(f"\n{task.upper()} RECOGNITION\n")
                f.write("-" * 70 + "\n")
                
                task_df = df[df['task'] == task]
                
                for _, row in task_df.iterrows():
                    f.write(f"\nModel: {row['model_name']}\n")
                    f.write(f"  Architecture: {row['architecture']}\n")
                    f.write(f"  Pretrained: {row['pretrained']}\n")
                    f.write(f"  Train Acc: {row['train_acc']:.2f}%\n")
                    f.write(f"  Val Acc: {row['val_acc']:.2f}%\n")
                    f.write(f"  Parameters: {row['total_params']:,}\n")
                    if pd.notna(row.get('train_time_seconds')):
                        f.write(f"  Training Time: {row['train_time_seconds']:.1f}s\n")
                
                # Best model
                best = self.get_best_model(task)
                if best:
                    f.write(f"\n🏆 Best Model: {best['model_name']} "
                           f"(Val Acc: {best['val_acc']:.2f}%)\n")
        
        print(f"\n✓ Summary exported to {output_path}")


def log_custom_model_results(task='emotion'):
    """
    Log results from custom CNN models (already trained).
    
    Args:
        task: 'emotion' or 'activity'
    """
    logger = ExperimentLogger()
    
    if task == 'emotion':
        experiment_data = {
            'model_name': 'CustomCNN',
            'task': 'emotion',
            'architecture': 'custom',
            'pretrained': False,
            'freeze_epochs': 0,
            'total_epochs': 50,
            'train_acc': 98.29,
            'val_acc': 64.22,
            'test_acc': 0.0,  # Not tested yet
            'train_loss': 0.0795,
            'val_loss': 0.0035,
            'test_loss': 0.0,
            'total_params': 981767,
            'trainable_params': 981767,
            'train_time_seconds': 0,  # Not tracked
            'avg_epoch_time': 0,
            'best_epoch': 50,
            'learning_rate': 0.001,
            'batch_size': 64,
            'notes': 'Custom CNN trained from scratch on FER-2013',
            'checkpoint_path': 'checkpoints/emotion/best_model.pth'
        }
    else:  # activity
        experiment_data = {
            'model_name': 'CustomCNN',
            'task': 'activity',
            'architecture': 'custom',
            'pretrained': False,
            'freeze_epochs': 0,
            'total_epochs': 50,
            'train_acc': 88.07,
            'val_acc': 84.62,
            'test_acc': 0.0,
            'train_loss': 0.3328,
            'val_loss': 0.4649,
            'test_loss': 0.0,
            'total_params': 981767,
            'trainable_params': 981767,
            'train_time_seconds': 0,
            'avg_epoch_time': 0,
            'best_epoch': 49,
            'learning_rate': 0.001,
            'batch_size': 32,
            'notes': 'Custom CNN trained from scratch on UCF101',
            'checkpoint_path': 'checkpoints/activity/best_model.pth'
        }
    
    logger.log_experiment(experiment_data)
    print(f"✓ Logged {task} custom model results")


if __name__ == "__main__":
    # Test experiment logger
    print("=" * 70)
    print("Testing Experiment Logger")
    print("=" * 70)
    
    # Log custom model results
    log_custom_model_results('emotion')
    log_custom_model_results('activity')
    
    # Load and display
    logger = ExperimentLogger()
    results = logger.load_results()
    
    print("\nLogged Experiments:")
    print(results[['model_name', 'task', 'val_acc', 'total_params']])
    
    # Export summary
    logger.export_summary()
    
    print("\n✅ Experiment logger tested successfully!")
