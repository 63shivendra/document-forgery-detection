import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_training_charts(history, output_dir='./charts', run_name='run'):
    """
    Saves a 4-panel professional training report chart per experiment run.
    Output: reports/<run_name>/charts/training_report.png
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Training Report — {run_name}', fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Panel 1: Loss Curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   label='Val Loss',   color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.fill_between(epochs,
                     history['train_loss'], history['val_loss'],
                     alpha=0.1, color='red', label='Overfit Gap')
    ax1.set_title('Loss Curve', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Focal-Dice Loss')
    ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)

    # ── Panel 2: IoU & F1 ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [v*100 for v in history['val_iou']], label='Val IoU (%)',      color='#2ca02c', linewidth=2)
    ax2.plot(epochs, [v*100 for v in history['val_f1']],  label='Val F1-Score (%)', color='#d62728', linewidth=2, linestyle='--')
    ax2.axhline(y=35, color='gray', linestyle=':', linewidth=1, label='Target 35%')
    ax2.set_title('IoU & F1-Score', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Percentage (%)')
    ax2.legend(); ax2.grid(True, linestyle=':', alpha=0.6)

    # ── Panel 3: Precision & Recall ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if 'val_precision' in history and history['val_precision']:
        ax3.plot(epochs, [v*100 for v in history['val_precision']], label='Precision (%)', color='#9467bd', linewidth=2)
        ax3.plot(epochs, [v*100 for v in history['val_recall']],    label='Recall (%)',    color='#8c564b', linewidth=2, linestyle='--')
    ax3.set_title('Precision & Recall', fontweight='bold')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Percentage (%)')
    ax3.legend(); ax3.grid(True, linestyle=':', alpha=0.6)

    # ── Panel 4: Learning Rate Schedule ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if 'lr_history' in history and history['lr_history']:
        ax4.plot(epochs, history['lr_history'], color='#e377c2', linewidth=2)
        ax4.set_yscale('log')
    ax4.set_title('Learning Rate Schedule', fontweight='bold')
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('LR (log scale)')
    ax4.grid(True, linestyle=':', alpha=0.6)

    chart_path = os.path.join(output_dir, 'training_report.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Training report chart saved: {chart_path}", flush=True)
