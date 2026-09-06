import os
import time
import json
import pickle
import torch
from torch.amp import autocast, GradScaler
from .metrics import FocalDiceLoss, calculate_pixel_metrics
from .plotter import save_training_charts


class Trainer:
    """
    Production-grade PyTorch Trainer Engine featuring:
    - AMP fp16 for fast VRAM training on RTX 3050.
    - Differential LR: pretrained encoder at 1/10th LR to prevent feature destruction.
    - Linear Warmup + CosineAnnealing scheduler to prevent overfitting in early epochs.
    - Gradient clipping + accumulation for stable training on large combined datasets.
    - torch.compile kernel fusion for GPU throughput.
    - Per-experiment report saving: reports/<run_name>/
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

        # Linear Warmup (5 epochs) + CosineAnnealing scheduler
        # Warmup: gradually ramps LR from 0 -> base_lr to protect pretrained weights in epoch 1-5
        # After warmup: cosine decay for smooth anti-overfitting convergence
        total_epochs = config['training']['epochs']
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,          # Start at 1% of base_lr
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-7
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_epochs]
        )

        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)

        self.save_dir = config['paths']['saved_models_dir']
        self.edgeai_dir = "./edgeai"
        self.charts_dir = config['paths']['charts_dir']
        self.reports_dir = config['paths'].get('reports_dir', './reports')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.edgeai_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

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
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if i % self.grad_accum_steps == 0 or i == total_batches:
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

    def fit(self, dataset_name="splicing_combined", run_name=None):
        if run_name is None:
            run_name = dataset_name

        # Per-experiment output folders
        run_report_dir = os.path.join(self.reports_dir, run_name)
        run_charts_dir = os.path.join(run_report_dir, 'charts')
        os.makedirs(run_report_dir, exist_ok=True)
        os.makedirs(run_charts_dir, exist_ok=True)

        print(f"\nStarting Training: [{run_name}]", flush=True)
        print(f" Forgery Types Covered: Splicing | Copy-Move | AI-Inpainting | Document-Tampering", flush=True)
        print(f" Reports will be saved to: {run_report_dir}", flush=True)

        history = {
            'run_name': run_name,
            'config': self.config,
            'train_loss': [], 'val_loss': [],
            'val_iou': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [],
            'lr_history': []
        }
        best_val_iou = 0.0
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        start_time = time.time()

        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate()

            current_lr = self.optimizer.param_groups[1]['lr']  # decoder LR
            self.scheduler.step()

            history['train_loss'].append(round(train_loss, 6))
            history['val_loss'].append(round(val_loss, 6))
            history['val_iou'].append(round(val_metrics['iou'], 6))
            history['val_f1'].append(round(val_metrics['f1'], 6))
            history['val_precision'].append(round(val_metrics['precision'], 6))
            history['val_recall'].append(round(val_metrics['recall'], 6))
            history['lr_history'].append(round(current_lr, 8))

            # Overfitting monitor: flag if gap > 0.15
            gap = val_loss - train_loss
            overfit_flag = " ⚠️  OVERFIT" if gap > 0.15 else ""

            print(f"Epoch [{epoch:02d}/{self.config['training']['epochs']:02d}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_metrics['iou']*100:.2f}% | Val F1: {val_metrics['f1']*100:.2f}% | "
                  f"LR: {current_lr:.1e}{overfit_flag}", flush=True)

            # Checkpoint & Production Save
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                patience_counter = 0

                best_ckpt_path = os.path.join(self.save_dir, f"best_model_{run_name}.pth")
                checkpoint = {
                    'epoch': epoch,
                    'run_name': run_name,
                    'forgery_types': ['splicing', 'copy_move', 'ai_inpainting', 'document_tampering'],
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_iou': val_metrics['iou'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'config': self.config
                }
                # Save as .pth (PyTorch standard format)
                torch.save(checkpoint, best_ckpt_path)

                # Save as .pkl (pickle format — compatible with sklearn/general ML pipelines)
                pkl_path = os.path.join(self.save_dir, f"best_model_{run_name}.pkl")
                with open(pkl_path, 'wb') as pkl_file:
                    pickle.dump(checkpoint, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

                prod_script_path = os.path.join(self.save_dir, f"production_model_{run_name}.pt")
                dummy_input = torch.randn(1, 3, *self.config['training']['image_size']).to(self.device)
                try:
                    traced_model = torch.jit.trace(self.model, dummy_input)
                    traced_model.save(prod_script_path)
                except Exception:
                    pass  # compile() wrapping may block trace; checkpoint is still saved

                onnx_path = os.path.join(self.edgeai_dir, f"model_{run_name}.onnx")
                try:
                    torch.onnx.export(
                        self.model, dummy_input, onnx_path,
                        export_params=True, opset_version=14,
                        do_constant_folding=True,
                        input_names=['input_document'],
                        output_names=['forgery_mask_logits'],
                        dynamic_axes={'input_document': {0: 'batch_size'},
                                      'forgery_mask_logits': {0: 'batch_size'}}
                    )
                    print(f"  --> Checkpoint (.pth): {best_ckpt_path}", flush=True)
                    print(f"  --> Pickle    (.pkl) : {pkl_path}", flush=True)
                    print(f"  --> ONNX      (.onnx): {onnx_path}", flush=True)
                except Exception as e:
                    print(f"  --> Checkpoint (.pth): {best_ckpt_path}", flush=True)
                    print(f"  --> Pickle    (.pkl) : {pkl_path}", flush=True)
                    print(f"  --> ONNX skipped: {e}", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" Early stopping after {epoch} epochs (no improvement for {patience} epochs).", flush=True)
                    break

        total_time = (time.time() - start_time) / 60.0

        # Final summary
        history['best_val_iou'] = round(best_val_iou, 6)
        history['total_epochs_run'] = len(history['train_loss'])
        history['training_time_minutes'] = round(total_time, 2)

        print(f"\nTraining Completed in {total_time:.2f} minutes.", flush=True)
        print(f"Best Val IoU : {best_val_iou*100:.2f}%", flush=True)
        print(f"Best Val F1  : {max(history['val_f1'])*100:.2f}%", flush=True)

        # Save full JSON report to reports/<run_name>/report.json
        report_path = os.path.join(run_report_dir, 'report.json')
        with open(report_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f" Full report saved: {report_path}", flush=True)

        # Save charts to reports/<run_name>/charts/
        save_training_charts(history, output_dir=run_charts_dir, run_name=run_name)

        return history
