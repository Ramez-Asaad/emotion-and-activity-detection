# Pretrained Model Experiments - Concise Summary

• **Best Emotion Model:** VGG-16 achieved 66.51% validation accuracy with minimal overfitting (9.46% gap), outperforming MobileNet-V2 which showed 25.31% overfitting despite being lightweight.

• **Best Activity Model:** MobileNet-V2 dominated with 82.05% accuracy, smallest size (2.2M params), and fastest training (31s), making it ideal for deployment.

• **Task Difficulty:** Emotion recognition proved significantly more challenging (66% max accuracy) compared to activity recognition (82% max accuracy).

• **Overfitting Patterns:** All models achieved 100% training accuracy on activity tasks, indicating strong ImageNet feature transfer but requiring better regularization.

• **Efficiency Winner:** MobileNet-V2 offers the best accuracy-to-size ratio across both tasks, while VGG-16 (134M params) underperformed on activity despite being the largest model.

• **Transfer Learning Impact:** ImageNet pretraining enabled rapid convergence (10-15 epochs) and consistently outperformed custom CNN baselines in both accuracy and training speed.

• **Production Recommendation:** Deploy MobileNet-V2 for both tasks due to optimal balance of performance (82% activity, 66% emotion), efficiency (2.2M params), and fast inference.

• **Future Improvements:** Ensemble methods, advanced regularization (dropout, mixup), and task-specific augmentation could boost accuracy by 2-5% while reducing overfitting gaps.
