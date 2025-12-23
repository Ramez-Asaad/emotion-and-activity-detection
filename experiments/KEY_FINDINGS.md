# Key Findings - Pretrained Model Experiments

## Emotion Recognition (FER-2013)

• VGG-16 achieved the best validation accuracy at 66.51%, demonstrating superior performance for emotion classification tasks.

• MobileNet-V2 showed significant overfitting with a 25.31% gap between training and validation accuracy, despite being the most lightweight model.

• The emotion recognition task proved more challenging than activity recognition, with maximum accuracy reaching only 66% compared to 82% for activity detection.

• VGG-16 exhibited better generalization with only a 9.46% overfitting gap, making it more reliable for real-world deployment despite its larger size.

• Transfer learning from ImageNet weights significantly accelerated convergence, with models reaching competitive accuracy within 10-15 epochs.

## Activity Recognition (UCF101)

• MobileNet-V2 emerged as the clear winner with 82.05% validation accuracy while maintaining the smallest model size (2.2M parameters) and fastest training time (31 seconds).

• All pretrained models achieved 100% training accuracy, indicating strong feature extraction capabilities from ImageNet pretraining.

• ResNet-50 provided the best accuracy-to-complexity trade-off for applications requiring higher precision, achieving 78.21% validation accuracy.

• VGG-16 underperformed despite being the largest model (134.3M parameters), achieving only 71.79% validation accuracy with the worst overfitting gap of 28.21%.

• Lightweight architectures (MobileNet-V2, EfficientNet-B0) demonstrated superior efficiency, making them ideal for resource-constrained deployment scenarios.

## Overall Conclusions

• Pretrained CNNs consistently outperformed custom CNN architectures in terms of accuracy and convergence speed across both tasks.

• Lightweight architectures such as MobileNet-V2 and EfficientNet-B0 achieved a strong balance between accuracy and computational efficiency.

• The custom CNN baseline required longer training time and showed lower generalization capability compared to pretrained models.

• Transfer learning proved highly effective for both emotion and activity recognition tasks, even with limited training data.

• Pretrained deep CNN models offer significant advantages in performance and efficiency, making them well-suited for real-world visual recognition applications.

• MobileNet-V2 is recommended for production deployment due to its optimal balance of accuracy (82% activity, 66% emotion), model size (2.2M params), and inference speed.

• Further improvements could be achieved through ensemble methods, advanced regularization techniques, and task-specific data augmentation strategies.
