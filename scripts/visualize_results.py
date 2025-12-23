"""
Professional Model Comparison Visualizations
============================================

Generate publication-quality plots for model performance comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Load data
df = pd.read_csv('experiments/results.csv')

# Clean model names for display
df['display_name'] = df['architecture'].str.replace('_', ' ').str.title()
df['params_millions'] = df['total_params'] / 1e6

# Create output directory
output_dir = Path('experiments/plots')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Generating Professional Model Comparison Visualizations")
print("=" * 70)
print(f"\nTotal models: {len(df)}")
print(f"Tasks: {df['task'].unique()}")
print(f"Architectures: {df['architecture'].unique()}")

# ============================================================================
# FIGURE 1: Validation Accuracy Comparison by Task
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, task in enumerate(['emotion', 'activity']):
    ax = axes[idx]
    task_df = df[df['task'] == task].sort_values('val_acc', ascending=False)
    
    if not task_df.empty:
        colors = sns.color_palette("husl", len(task_df))
        bars = ax.barh(task_df['display_name'], task_df['val_acc'], color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, task_df['val_acc'])):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}%', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Validation Accuracy (%)', fontweight='bold')
        ax.set_title(f'{task.capitalize()} Recognition', fontweight='bold', fontsize=14)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / '1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 1_accuracy_comparison.png")
plt.close()

# ============================================================================
# FIGURE 2: Training vs Validation Accuracy
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(df))
width = 0.35

bars1 = ax.bar(x - width/2, df['train_acc'], width, label='Training Accuracy', 
              alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, df['val_acc'], width, label='Validation Accuracy', 
              alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Training vs Validation Accuracy Across All Models', fontweight='bold', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([f"{row['display_name']}\n({row['task']})" for _, row in df.iterrows()], 
                    rotation=45, ha='right')
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 110)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / '2_train_vs_val_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 2_train_vs_val_accuracy.png")
plt.close()

# ============================================================================
# FIGURE 3: Model Efficiency (Parameters vs Accuracy)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, task in enumerate(['emotion', 'activity']):
    ax = axes[idx]
    task_df = df[df['task'] == task]
    
    if not task_df.empty:
        scatter = ax.scatter(task_df['params_millions'], task_df['val_acc'], 
                           s=300, alpha=0.6, c=range(len(task_df)), 
                           cmap='viridis', edgecolors='black', linewidth=2)
        
        # Add model labels
        for _, row in task_df.iterrows():
            ax.annotate(row['display_name'], 
                       (row['params_millions'], row['val_acc']),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
        ax.set_title(f'{task.capitalize()} Recognition - Efficiency Analysis', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / '3_efficiency_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3_efficiency_analysis.png")
plt.close()

# ============================================================================
# FIGURE 4: Training Time Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Sort by training time
df_sorted = df.sort_values('train_time_seconds')

colors = sns.color_palette("coolwarm", len(df_sorted))
bars = ax.barh(df_sorted['display_name'] + '\n(' + df_sorted['task'] + ')', 
              df_sorted['train_time_seconds'] / 60, color=colors, 
              edgecolor='black', linewidth=1.2)

# Add time labels
for bar, time in zip(bars, df_sorted['train_time_seconds']):
    ax.text(time/60 + 0.5, bar.get_y() + bar.get_height()/2,
           f'{time/60:.1f} min', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Training Time (minutes)', fontweight='bold')
ax.set_title('Total Training Time Comparison', fontweight='bold', fontsize=16)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / '4_training_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 4_training_time.png")
plt.close()

# ============================================================================
# FIGURE 5: Comprehensive Performance Matrix
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top-left: Accuracy by architecture
ax1 = fig.add_subplot(gs[0, 0])
arch_acc = df.groupby('architecture')['val_acc'].mean().sort_values(ascending=False)
colors = sns.color_palette("Set2", len(arch_acc))
bars = ax1.bar(range(len(arch_acc)), arch_acc.values, color=colors, 
              edgecolor='black', linewidth=1.2)
ax1.set_xticks(range(len(arch_acc)))
ax1.set_xticklabels([a.replace('_', ' ').title() for a in arch_acc.index], rotation=45, ha='right')
ax1.set_ylabel('Average Validation Accuracy (%)', fontweight='bold')
ax1.set_title('Average Performance by Architecture', fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars, arch_acc.values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1, 
            f'{val:.1f}%', ha='center', fontweight='bold')

# Top-right: Parameters distribution
ax2 = fig.add_subplot(gs[0, 1])
sizes = df.groupby('architecture')['params_millions'].first().sort_values()
colors = sns.color_palette("Spectral", len(sizes))
wedges, texts, autotexts = ax2.pie(sizes.values, labels=[s.replace('_', ' ').title() for s in sizes.index],
                                    autopct='%1.1f%%', colors=colors, startangle=90,
                                    textprops={'fontweight': 'bold'})
ax2.set_title('Model Size Distribution (by Parameters)', fontweight='bold')

# Bottom-left: Task performance comparison
ax3 = fig.add_subplot(gs[1, 0])
task_perf = df.groupby('task')['val_acc'].agg(['mean', 'std'])
x_pos = np.arange(len(task_perf))
bars = ax3.bar(x_pos, task_perf['mean'], yerr=task_perf['std'], 
              capsize=10, alpha=0.7, edgecolor='black', linewidth=1.2)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([t.capitalize() for t in task_perf.index])
ax3.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax3.set_title('Average Performance by Task', fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars, task_perf['mean']):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 2, 
            f'{val:.1f}%', ha='center', fontweight='bold')

# Bottom-right: Overfitting analysis
ax4 = fig.add_subplot(gs[1, 1])
df['overfit_gap'] = df['train_acc'] - df['val_acc']
overfit_sorted = df.sort_values('overfit_gap', ascending=False)
colors = ['red' if gap > 15 else 'orange' if gap > 10 else 'green' 
         for gap in overfit_sorted['overfit_gap']]
bars = ax4.barh(overfit_sorted['display_name'] + '\n(' + overfit_sorted['task'] + ')',
               overfit_sorted['overfit_gap'], color=colors, 
               edgecolor='black', linewidth=1.2, alpha=0.7)
ax4.set_xlabel('Overfitting Gap (Train - Val Accuracy)', fontweight='bold')
ax4.set_title('Overfitting Analysis', fontweight='bold')
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Moderate (10%)')
ax4.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='High (15%)')
ax4.legend()

plt.suptitle('Comprehensive Model Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.savefig(output_dir / '5_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 5_comprehensive_analysis.png")
plt.close()

# ============================================================================
# FIGURE 6: Best Model Summary Table (as image)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_data = []
for task in df['task'].unique():
    task_df = df[df['task'] == task].sort_values('val_acc', ascending=False)
    if not task_df.empty:
        best = task_df.iloc[0]
        summary_data.append([
            task.capitalize(),
            best['display_name'],
            f"{best['val_acc']:.2f}%",
            f"{best['train_acc']:.2f}%",
            f"{best['params_millions']:.1f}M",
            f"{best['train_time_seconds']/60:.1f} min"
        ])

table = ax.table(cellText=summary_data,
                colLabels=['Task', 'Best Model', 'Val Acc', 'Train Acc', 'Parameters', 'Train Time'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.2, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style cells
for i in range(1, len(summary_data) + 1):
    for j in range(6):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

plt.title('Best Performing Models Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / '6_best_models_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 6_best_models_summary.png")
plt.close()

# ============================================================================
# Generate Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for task in df['task'].unique():
    task_df = df[df['task'] == task]
    print(f"\n{task.upper()} RECOGNITION:")
    print(f"  Best Model: {task_df.loc[task_df['val_acc'].idxmax(), 'display_name']}")
    print(f"  Best Val Acc: {task_df['val_acc'].max():.2f}%")
    print(f"  Average Val Acc: {task_df['val_acc'].mean():.2f}%")
    print(f"  Fastest Training: {task_df.loc[task_df['train_time_seconds'].idxmin(), 'display_name']} "
          f"({task_df['train_time_seconds'].min()/60:.1f} min)")

print("\n" + "=" * 70)
print("✅ All visualizations generated successfully!")
print("=" * 70)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  1. 1_accuracy_comparison.png - Side-by-side accuracy comparison")
print("  2. 2_train_vs_val_accuracy.png - Training vs validation accuracy")
print("  3. 3_efficiency_analysis.png - Parameters vs accuracy scatter")
print("  4. 4_training_time.png - Training time comparison")
print("  5. 5_comprehensive_analysis.png - 4-panel comprehensive analysis")
print("  6. 6_best_models_summary.png - Summary table")
