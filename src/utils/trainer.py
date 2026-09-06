import os
import time
import json
import torch
from torch.amp import autocast, GradScaler
from .metrics import FocalDiceLoss, calculate_pixel_metrics
from .plotter import save_training_charts

class Trainer:
    """
    Production-grade PyTorch Trainer Engine featuring:
    - Automatic Mixed Precision (AMP fp16) for fast VRAM training on RTX 3050.
    - AdamW + CosineAnnealingWarmRestarts for anti-overfitting optimization.
    - Early Stopping to prevent overtraining.
    - Automatic Export to `savedmodels/` (.pth & .pt) and `edgeai/` (.onnx).
    """
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = FocalDiceLoss(
            focal_weight=config['loss']['focal_weight'],
            dice_weight=config['loss']['dice_weight'],
            gamma=config['loss']['focal_gamma'],
            pos_weight=config['loss'].get('pos_weight', 10.0)
        )

        # Differential learning rates:
        # Pretrained encoder -> fine-tune slowly (1/10th LR) to preserve ImageNet features
        # Decoder + SRM + projection head -> train at full LR (random init)
        base_lr = config['training']['learning_rate']
        enc_lr_mult = config['training'].get('encoder_lr_multiplier', 0.1)
        encoder_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if 'unet.encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': base_lr * enc_lr_mult},
            {'params': decoder_params, 'lr': base_lr}
        ], weight_decay=config['training']['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-7
        )

        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)

        self.save_dir = config['paths']['saved_models_dir']
        self.edgeai_dir = "./edgeai"
        self.charts_dir = config['paths']['charts_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.edgeai_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

        # torch.compile: fuses CUDA kernels for 10-30% GPU throughput boost (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print(" torch.compile() enabled — GPU kernel fusion active.", flush=True)
            except Exception as e:
                print(f" torch.compile() skipped: {e}", flush=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)  # set_to_none frees memory immediately

        for i, (images, masks, _) in enumerate(self.train_loader, 1):
            # non_blocking=True: async CPU→GPU transfer overlaps with GPU compute
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                # Scale loss by accumulation steps so gradients average correctly
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if i % self.grad_accum_steps == 0 or i == total_batches:
                # Unscale before clipping so clipping works on true gradient magnitudes
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * self.grad_accum_steps * images.size(0)

            if i % 15 == 0 or i == total_batches:
                print(f"   Batch [{i:03d}/{total_batches:03d}] Loss: {loss.item() * self.grad_accum_steps:.4f}", flush=True)

        return running_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        metrics_acc = {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        for images, masks, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            probs = torch.sigmoid(logits)
            batch_metrics = calculate_pixel_metrics(probs, masks)

            running_loss += loss.item() * images.size(0)
            for k in metrics_acc:
                metrics_acc[k] += batch_metrics[k] * images.size(0)

        val_loss = running_loss / len(self.val_loader.dataset)
        for k in metrics_acc:
            metrics_acc[k] /= len(self.val_loader.dataset)

        return val_loss, metrics_acc

    def fit(self, dataset_name="DocTamper"):
        print(f"\nStarting Full-Dataset Training on [{dataset_name}]...", flush=True)
        history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}
        best_val_iou = 0.0
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']

        start_time = time.time()

        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate()
            self.scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_metrics['iou'])
            history['val_f1'].append(val_metrics['f1'])

            print(f"Epoch [{epoch:02d}/{self.config['training']['epochs']:02d}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_metrics['iou']*100:.2f}% | Val F1: {val_metrics['f1']*100:.2f}%", flush=True)

            # Checkpoint & Production Save
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                patience_counter = 0

                # 1. Save PyTorch State Dict Checkpoint in savedmodels/
                best_ckpt_path = os.path.join(self.save_dir, f"best_model_{dataset_name.lower()}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_iou': val_metrics['iou'],
                    'val_f1': val_metrics['f1'],
                    'config': self.config
                }, best_ckpt_path)

                # 2. Save Production TorchScript Model (.pt) in savedmodels/
                prod_script_path = os.path.join(self.save_dir, f"production_model_{dataset_name.lower()}.pt")
                dummy_input = torch.randn(1, 3, *self.config['training']['image_size']).to(self.device)
                traced_model = torch.jit.trace(self.model, dummy_input)
                traced_model.save(prod_script_path)

                # 3. Export ONNX Model (.onnx) into edgeai/ folder
                onnx_path = os.path.join(self.edgeai_dir, f"model_{dataset_name.lower()}.onnx")
                try:
                    torch.onnx.export(
                        self.model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=14,
                        do_constant_folding=True,
                        input_names=['input_document'],
                        output_names=['forgery_mask_logits'],
                        dynamic_axes={'input_document': {0: 'batch_size'}, 'forgery_mask_logits': {0: 'batch_size'}}
                    )
                    print(f"  --> Saved Checkpoint: savedmodels/best_model_{dataset_name.lower()}.pth", flush=True)
                    print(f"  --> Exported ONNX Model: edgeai/model_{dataset_name.lower()}.onnx", flush=True)
                except Exception as e:
                    print(f"  --> Saved Checkpoint: savedmodels/best_model_{dataset_name.lower()}.pth (ONNX Notice: {e})", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" Early stopping triggered after {epoch} epochs.", flush=True)
                    break

        total_time = (time.time() - start_time) / 60.0
        print(f"\nTraining Completed in {total_time:.2f} minutes. Best Val IoU: {best_val_iou*100:.2f}%", flush=True)

        # Save Plots
        save_training_charts(history, output_dir=self.charts_dir)
        return history
