import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.models.forgery_net import ForgeryNet
from src.dataset.base_dataset import get_dataset
from src.utils.trainer import Trainer


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True: cuDNN auto-tunes fastest conv kernel for fixed input size (big speedup)
    # deterministic=False: allows cuDNN to use non-deterministic (faster) algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main():
    parser = argparse.ArgumentParser(description="Full Dataset Training for Document Forgery Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='splicing_combined',
                        choices=['doctamper', 'cocoglide', 'splicing', 'casia', 'combined', 'splicing_combined'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, help='Override training epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"==================================================")
    print(f"  Document Forgery Framework - ForgeryNet v2.0")
    print(f"  Encoder        : {config['model']['encoder']} (ImageNet pretrained)")
    print(f"  Target Dataset : {args.dataset.upper()}")
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"  Image Size     : {config['training']['image_size']}")
    print(f"  Loss pos_weight: {config['loss']['pos_weight']}x")
    print(f"==================================================")

    img_size = tuple(config['training']['image_size'])

    # Instantiate Datasets
    train_dataset = get_dataset(args.dataset, data_root=config['paths']['data_dir'], split='train', img_size=img_size)
    val_dataset   = get_dataset(args.dataset, data_root=config['paths']['data_dir'], split='val',   img_size=img_size)

    print(f" Loaded [{len(train_dataset)}] training samples for {args.dataset.upper()}.")
    print(f" Loaded [{len(val_dataset)}] validation samples for {args.dataset.upper()}.")

    # DataLoader with all CUDA parallelization flags
    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,  # pre-load 4 batches per worker
        drop_last=True   # skip partial last batch — avoids GPU underutilization on final batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )

    # Instantiate ForgeryNet (EfficientNet-B2 + SRM + U-Net decoder)
    model = ForgeryNet(
        out_channels=config['model']['out_channels'],
        encoder_name=config['model']['encoder'],
        dropout_prob=config['model']['spatial_dropout']
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f" Model params: {total_params:.1f}M total | {trainable_params:.1f}M trainable")

    # Train Engine
    from datetime import datetime
    run_name = f"{args.dataset}_{config['model']['encoder']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    trainer.fit(dataset_name=args.dataset, run_name=run_name)


if __name__ == '__main__':
    main()
