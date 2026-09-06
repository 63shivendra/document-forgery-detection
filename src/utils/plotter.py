import os
import matplotlib.pyplot as plt

def save_training_charts(history, output_dir='./charts'):
    """
    Auto-generates clean, professional training & validation charts (Loss, IoU, F1-Score).
    Saves plots into the `charts/` folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss Chart
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Training & Validation Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Focal-Dice Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_chart_path = os.path.join(output_dir, 'loss_chart.png')
    plt.savefig(loss_chart_path, dpi=300)
    plt.close()

    # 2. Metrics Chart (IoU and F1-Score)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [val * 100 for val in history['val_iou']], label='Val IoU (%)', color='#2ca02c', linewidth=2)
    plt.plot(epochs, [val * 100 for val in history['val_f1']], label='Val F1-Score (%)', color='#d62728', linewidth=2, linestyle='--')
    plt.title('Validation Accuracy & Localization Metrics (IoU & F1)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    metrics_chart_path = os.path.join(output_dir, 'metrics_chart.png')
    plt.savefig(metrics_chart_path, dpi=300)
    plt.close()

    print(f" Saved training charts to:\n  - {loss_chart_path}\n  - {metrics_chart_path}")
