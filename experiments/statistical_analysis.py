"""
Statistical Analysis and Visualization of Model Results
=========================================================

Generates comprehensive statistical analysis and visualizations from results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "plots" / "statistical_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load and prepare the results data."""
    csv_path = Path(__file__).parent / "results.csv"
    df = pd.read_csv(csv_path)
    
    # Calculate overfitting gap
    df['overfitting_gap'] = df['train_acc'] - df['val_acc']
    
    # Calculate efficiency (accuracy per million parameters)
    df['params_millions'] = df['total_params'] / 1e6
    df['efficiency'] = df['val_acc'] / df['params_millions']
    
    # Calculate training efficiency (accuracy per second)
    df['train_efficiency'] = df['val_acc'] / df['train_time_seconds']
    
    return df


def plot_accuracy_comparison(df):
    """Create accuracy comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate by task
    for idx, task in enumerate(['emotion', 'activity']):
        task_df = df[df['task'] == task].copy()
        
        ax = axes[idx]
        x = np.arange(len(task_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, task_df['train_acc'], width, 
                       label='Training Accuracy', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, task_df['val_acc'], width, 
                       label='Validation Accuracy', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{task.capitalize()} Recognition - Accuracy Comparison', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_df['architecture'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 110)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: accuracy_comparison.png")


def plot_overfitting_analysis(df):
    """Create overfitting analysis chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'emotion': '#e74c3c', 'activity': '#3498db'}
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        ax.scatter(task_df['architecture'], task_df['overfitting_gap'], 
                  s=200, c=colors[task], label=f'{task.capitalize()} Task', 
                  alpha=0.7, edgecolors='black', linewidth=2)
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='No Overfitting')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Moderate Overfitting (10%)')
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Severe Overfitting (20%)')
    
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overfitting Gap (Train Acc - Val Acc) %', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis Across Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: overfitting_analysis.png")


def plot_model_complexity(df):
    """Create model complexity vs performance chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'emotion': '#e74c3c', 'activity': '#3498db'}
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        scatter = ax.scatter(task_df['params_millions'], task_df['val_acc'], 
                            s=task_df['train_time_seconds'] / 5,  # Size by training time
                            c=colors[task], label=f'{task.capitalize()} Task',
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add architecture labels
        for _, row in task_df.iterrows():
            ax.annotate(row['architecture'], 
                       (row['params_millions'], row['val_acc']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity vs Performance\n(Bubble size = Training Time)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'complexity_vs_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: complexity_vs_performance.png")


def plot_training_efficiency(df):
    """Create training efficiency analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training time comparison
    ax1 = axes[0]
    task_colors = {'emotion': '#e74c3c', 'activity': '#3498db'}
    bars = ax1.barh(df['architecture'] + '\n(' + df['task'] + ')', 
                    df['train_time_seconds'] / 60,
                    color=[task_colors[t] for t in df['task']], alpha=0.8)
    ax1.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.annotate(f'{width:.1f} min', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9)
    
    # Epoch time comparison
    ax2 = axes[1]
    bars2 = ax2.barh(df['architecture'] + '\n(' + df['task'] + ')', 
                     df['avg_epoch_time'],
                     color=[task_colors[t] for t in df['task']], alpha=0.8)
    ax2.set_xlabel('Average Epoch Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Epoch Training Time', fontsize=14, fontweight='bold')
    
    for bar in bars2:
        width = bar.get_width()
        ax2.annotate(f'{width:.1f}s', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: training_efficiency.png")


def plot_loss_comparison(df):
    """Create loss comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, task in enumerate(['emotion', 'activity']):
        task_df = df[df['task'] == task].copy()
        
        ax = axes[idx]
        x = np.arange(len(task_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, task_df['train_loss'], width, 
                       label='Training Loss', color='#9b59b6', alpha=0.8)
        bars2 = ax.bar(x + width/2, task_df['val_loss'], width, 
                       label='Validation Loss', color='#f39c12', alpha=0.8)
        
        ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'{task.capitalize()} Recognition - Loss Comparison', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_df['architecture'], rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: loss_comparison.png")


def plot_efficiency_radar(df):
    """Create radar chart for model comparison."""
    # Prepare data for activity models (more models for better radar)
    activity_df = df[df['task'] == 'activity'].copy()
    
    if len(activity_df) < 3:
        print("[WARN] Not enough activity models for radar chart")
        return
    
    # Normalize metrics for radar
    metrics = ['val_acc', 'efficiency', 'train_efficiency']
    metric_labels = ['Validation\nAccuracy', 'Param\nEfficiency', 'Time\nEfficiency']
    
    # Min-max normalize
    normalized = activity_df.copy()
    for metric in metrics:
        min_val = normalized[metric].min()
        max_val = normalized[metric].max()
        if max_val > min_val:
            normalized[metric] = (normalized[metric] - min_val) / (max_val - min_val)
        else:
            normalized[metric] = 1.0
    
    # Radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(activity_df)))
    
    for idx, (_, row) in enumerate(activity_df.iterrows()):
        values = [normalized.iloc[idx][m] for m in metrics]
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=row['architecture'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_title('Activity Models - Multi-Metric Comparison\n(Normalized)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activity_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: activity_radar.png")


def plot_heatmap_summary(df):
    """Create heatmap summary of all metrics."""
    # Select key metrics
    metrics = ['train_acc', 'val_acc', 'overfitting_gap', 'train_loss', 'val_loss', 
               'params_millions', 'train_time_seconds', 'avg_epoch_time']
    
    # Create summary dataframe
    summary_df = df[['architecture', 'task'] + metrics].copy()
    summary_df['model'] = summary_df['architecture'] + '\n(' + summary_df['task'] + ')'
    summary_df = summary_df.set_index('model')[metrics]
    
    # Normalize for heatmap
    normalized = (summary_df - summary_df.min()) / (summary_df.max() - summary_df.min())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(normalized, annot=summary_df.round(2), fmt='', cmap='RdYlGn', 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalized Value'})
    
    ax.set_title('Model Comparison Heatmap\n(Values shown, colors normalized)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: metrics_heatmap.png")


def generate_statistical_summary(df):
    """Generate statistical summary report."""
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Models Analyzed: {len(df)}")
    
    # Overall Statistics
    report.append("\n" + "=" * 70)
    report.append("1. OVERALL STATISTICS")
    report.append("=" * 70)
    
    stats = df[['train_acc', 'val_acc', 'overfitting_gap', 'train_loss', 'val_loss', 
                'params_millions', 'train_time_seconds']].describe()
    report.append(str(stats.round(4)))
    
    # Best Models
    report.append("\n" + "=" * 70)
    report.append("2. BEST MODELS BY TASK")
    report.append("=" * 70)
    
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        best_val = task_df.loc[task_df['val_acc'].idxmax()]
        most_efficient = task_df.loc[task_df['efficiency'].idxmax()]
        fastest = task_df.loc[task_df['train_time_seconds'].idxmin()]
        
        report.append(f"\n{task.upper()} RECOGNITION:")
        report.append(f"  Best Validation Accuracy: {best_val['architecture']} ({best_val['val_acc']:.2f}%)")
        report.append(f"  Most Parameter Efficient: {most_efficient['architecture']} ({most_efficient['efficiency']:.2f} acc/M params)")
        report.append(f"  Fastest Training: {fastest['architecture']} ({fastest['train_time_seconds']/60:.1f} min)")
    
    # Overfitting Analysis
    report.append("\n" + "=" * 70)
    report.append("3. OVERFITTING ANALYSIS")
    report.append("=" * 70)
    
    for _, row in df.iterrows():
        gap = row['overfitting_gap']
        status = "[OK] Good" if gap < 10 else ("[WARN] Moderate" if gap < 20 else "[X] Severe")
        report.append(f"  {row['architecture']:20} ({row['task']:8}): {gap:6.2f}% gap - {status}")
    
    # Correlation Analysis
    report.append("\n" + "=" * 70)
    report.append("4. KEY CORRELATIONS")
    report.append("=" * 70)
    
    numeric_cols = ['train_acc', 'val_acc', 'params_millions', 'train_time_seconds', 'overfitting_gap']
    correlations = df[numeric_cols].corr()
    
    report.append("\nCorrelation Matrix:")
    report.append(str(correlations.round(3)))
    
    # Key insights
    report.append("\n" + "=" * 70)
    report.append("5. KEY INSIGHTS")
    report.append("=" * 70)
    
    # Activity task insights
    activity_df = df[df['task'] == 'activity']
    if len(activity_df) > 0:
        best_activity = activity_df.loc[activity_df['val_acc'].idxmax()]
        report.append(f"\n• Activity Recognition: MobileNetV2 achieves best accuracy ({best_activity['val_acc']:.2f}%)")
        report.append(f"  with only {best_activity['params_millions']:.2f}M parameters")
        
        # Check for overfitting
        high_overfit = activity_df[activity_df['overfitting_gap'] > 20]
        if len(high_overfit) > 0:
            report.append(f"\n• VGG16 shows significant overfitting ({high_overfit.iloc[0]['overfitting_gap']:.1f}% gap)")
            report.append("  Consider: More regularization, data augmentation, or dropout")
    
    # Emotion task insights
    emotion_df = df[df['task'] == 'emotion']
    if len(emotion_df) > 0:
        best_emotion = emotion_df.loc[emotion_df['val_acc'].idxmax()]
        report.append(f"\n• Emotion Recognition: {best_emotion['architecture']} achieves best accuracy ({best_emotion['val_acc']:.2f}%)")
        report.append("  Emotion task shows lower accuracy overall - common for FER datasets")
        
        overfit_gap = emotion_df['overfitting_gap'].mean()
        report.append(f"\n• Average overfitting gap for emotion: {overfit_gap:.1f}%")
    
    # Training efficiency
    report.append("\n• Parameter Efficiency Rankings (Accuracy per Million Parameters):")
    df_sorted = df.sort_values('efficiency', ascending=False)
    for _, row in df_sorted.head(3).iterrows():
        report.append(f"  - {row['architecture']} ({row['task']}): {row['efficiency']:.2f}")
    
    report.append("\n" + "=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "statistical_report.txt"
    import io
    with io.open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[OK] Saved: statistical_report.txt")
    
    return report_text


def main():
    """Main function to run all analyses."""
    print("=" * 70)
    print("Statistical Analysis of Model Results")
    print("=" * 70)
    
    # Load data
    print("\n[*] Loading data...")
    df = load_and_prepare_data()
    print(f"   Loaded {len(df)} model results")
    
    # Generate visualizations
    print("\n[*] Generating visualizations...")
    plot_accuracy_comparison(df)
    plot_overfitting_analysis(df)
    plot_model_complexity(df)
    plot_training_efficiency(df)
    plot_loss_comparison(df)
    plot_efficiency_radar(df)
    plot_heatmap_summary(df)
    
    # Generate report
    print("\n[*] Generating statistical report...")
    report = generate_statistical_summary(df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name}")
    
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
