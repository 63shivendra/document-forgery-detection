import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.models.dual_branch_unet import DualBranchUNet
from src.dataset.base_dataset import ForgeryDataset
from src.utils.metrics import calculate_pixel_metrics

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluation & Benchmarking for Document Forgery Framework")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, default='doctamper')
    parser.add_argument('--model-path', type=str, required=False, help='Path to .pth checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.model_path or os.path.join(config['paths']['saved_models_dir'], f"best_model_{args.dataset.lower()}.pth")

    print(f"==================================================")
    print(f"  Benchmarking Model on [{args.dataset.upper()}] Test Set")
    print(f"  Checkpoint Path : {model_path}")
    print(f"==================================================")

    if not os.path.exists(model_path):
        print(f"[Error] Checkpoint file not found at: {model_path}")
        return

    # Load Model
    model = DualBranchUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_filters=config['model']['base_filters'],
        num_groups=config['model']['num_groups'],
        dropout_prob=config['model']['spatial_dropout']
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load Test Dataset
    data_root = os.path.join(config['paths']['data_dir'], args.dataset)
    test_dataset = ForgeryDataset(data_root=data_root, split='test', img_size=tuple(config['training']['image_size']), is_train=False)
    
    if len(test_dataset) == 0:
        print(f"[Notice] No test images found in '{data_root}/test/images'.")
        return

    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    total_iou, total_f1, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0
    num_samples = len(test_dataset)

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            metrics = calculate_pixel_metrics(probs, masks)
            batch_sz = images.size(0)

            total_iou += metrics['iou'] * batch_sz
            total_f1 += metrics['f1'] * batch_sz
            total_prec += metrics['precision'] * batch_sz
            total_rec += metrics['recall'] * batch_sz

    print("\n" + "="*45)
    print("📊 BENCHMARK PERFORMANCE METRICS")
    print("="*45)
    print(f"Pixel IoU       : {total_iou / num_samples * 100:.2f}%")
    print(f"Pixel F1-Score  : {total_f1 / num_samples * 100:.2f}%")
    print(f"Pixel Precision : {total_prec / num_samples * 100:.2f}%")
    print(f"Pixel Recall    : {total_rec / num_samples * 100:.2f}%")
    print("="*45)

if __name__ == '__main__':
    evaluate()
