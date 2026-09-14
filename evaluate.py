import os
import glob
import json
import csv
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from src.models.forgery_net import ForgeryNet
from src.dataset.base_dataset import get_dataset
from src.utils.metrics import calculate_pixel_metrics

def evaluate_dataset(dataset_name, model, config, args, device, model_path):
    print("\n" + "=" * 68)
    print(f"  Evaluating Test Dataset: [{dataset_name.upper()}]")
    print("=" * 68)

    img_size = tuple(config['training']['image_size'])
    full_dataset = get_dataset(dataset_name, data_root=config['paths']['data_dir'], split='test', img_size=img_size)

    total_available = len(full_dataset)
    if args.num_samples and args.num_samples < total_available:
        eval_dataset = Subset(full_dataset, range(args.num_samples))
        num_samples = args.num_samples
        print(f" Evaluating on [{num_samples}/{total_available}] samples from test set.")
    else:
        eval_dataset = full_dataset
        num_samples = total_available
        print(f" Evaluating on all [{num_samples}] samples from test set.")

    if num_samples == 0:
        print(f" [Warning] No test samples found for {dataset_name}.")
        return None

    num_workers = min(4, os.cpu_count() or 2)
    test_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    metric_keys = [
        'iou', 'f1', 'precision', 'recall',
        'pixel_accuracy', 'balanced_accuracy', 'specificity', 'mae',
        'image_accuracy', 'image_tp', 'image_fp', 'image_fn', 'image_tn'
    ]
    accumulated = {k: 0.0 for k in metric_keys}
    total_processed = 0

    use_amp = config['training'].get('use_amp', True) and device == 'cuda'

    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(test_loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images)
                probs = torch.sigmoid(logits)

            metrics = calculate_pixel_metrics(probs, masks, threshold=args.threshold)
            batch_sz = images.size(0)

            for k in metric_keys:
                accumulated[k] += metrics[k] * batch_sz
            total_processed += batch_sz

            if batch_idx % 25 == 0 or total_processed == num_samples:
                print(f"   Batch [{batch_idx:04d}/{len(test_loader):04d}] "
                      f"Processed: {total_processed}/{num_samples} | "
                      f"Pixel Acc: {accumulated['pixel_accuracy'] / total_processed * 100:.2f}% | "
                      f"IoU: {accumulated['iou'] / total_processed * 100:.2f}% | "
                      f"F1: {accumulated['f1'] / total_processed * 100:.2f}%", flush=True)

    # Calculate final averages
    mean_pix_acc = (accumulated['pixel_accuracy'] / num_samples) * 100
    mean_bal_acc = (accumulated['balanced_accuracy'] / num_samples) * 100
    mean_spec = (accumulated['specificity'] / num_samples) * 100
    mean_iou = (accumulated['iou'] / num_samples) * 100
    mean_f1 = (accumulated['f1'] / num_samples) * 100
    mean_prec = (accumulated['precision'] / num_samples) * 100
    mean_rec = (accumulated['recall'] / num_samples) * 100
    mean_mae = (accumulated['mae'] / num_samples)
    mean_img_acc = (accumulated['image_accuracy'] / num_samples) * 100

    img_tp = accumulated['image_tp']
    img_fp = accumulated['image_fp']
    img_fn = accumulated['image_fn']
    img_prec = (img_tp / (img_tp + img_fp + 1e-7)) * 100
    img_rec = (img_tp / (img_tp + img_fn + 1e-7)) * 100
    img_f1 = (2 * img_prec * img_rec / (img_prec + img_rec + 1e-7))

    print("\n" + "-" * 68)
    print(f">> RESULTS FOR [{dataset_name.upper()}]:")
    print(f"  * Samples Evaluated            : {num_samples}")
    print(f"  * Pixel Accuracy               : {mean_pix_acc:.2f}%")
    print(f"  * Pixel Balanced Accuracy      : {mean_bal_acc:.2f}%")
    print(f"  * Pixel Specificity (TNR)      : {mean_spec:.2f}%")
    print(f"  * Pixel Sensitivity / Recall   : {mean_rec:.2f}%")
    print(f"  * Pixel Precision              : {mean_prec:.2f}%")
    print(f"  * Pixel F1-Score (Dice Score)  : {mean_f1:.2f}%")
    print(f"  * Pixel IoU (Jaccard Index)    : {mean_iou:.2f}%")
    print(f"  * Mean Absolute Error (MAE)    : {mean_mae:.5f}")
    print(f"  * Document Classification Acc  : {mean_img_acc:.2f}%")
    print(f"  * Document Detection Precision : {img_prec:.2f}%")
    print(f"  * Document Detection Recall    : {img_rec:.2f}%")
    print(f"  * Document Detection F1-Score  : {img_f1:.2f}%")
    print("-" * 68)

    # Save individual results
    os.makedirs(args.output_dir, exist_ok=True)
    sample_tag = f"_{num_samples}samples" if args.num_samples else ""
    json_path = os.path.join(args.output_dir, f"test_result_{dataset_name}{sample_tag}.json")
    csv_path = os.path.join(args.output_dir, f"test_result_{dataset_name}{sample_tag}.csv")
    txt_path = os.path.join(args.output_dir, f"summary_{dataset_name}{sample_tag}.txt")

    results_data = {
        'test_dataset': dataset_name,
        'checkpoint': os.path.basename(model_path),
        'num_samples_evaluated': num_samples,
        'threshold': args.threshold,
        'pixel_accuracy_percent': round(mean_pix_acc, 4),
        'pixel_balanced_accuracy_percent': round(mean_bal_acc, 4),
        'pixel_specificity_percent': round(mean_spec, 4),
        'pixel_recall_sensitivity_percent': round(mean_rec, 4),
        'pixel_precision_percent': round(mean_prec, 4),
        'pixel_f1_score_dice_percent': round(mean_f1, 4),
        'pixel_iou_jaccard_percent': round(mean_iou, 4),
        'pixel_mae': round(mean_mae, 6),
        'document_level_classification_accuracy_percent': round(mean_img_acc, 4),
        'document_level_detection_precision_percent': round(img_prec, 4),
        'document_level_detection_recall_percent': round(img_rec, 4),
        'document_level_detection_f1_percent': round(img_f1, 4)
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric Category', 'Metric Name', 'Value / Percentage'])
        writer.writerow(['Metadata', 'Test Dataset', dataset_name])
        writer.writerow(['Metadata', 'Checkpoint', os.path.basename(model_path)])
        writer.writerow(['Metadata', 'Samples Evaluated', num_samples])
        writer.writerow(['Metadata', 'Detection Threshold', args.threshold])
        writer.writerow(['Pixel-Level', 'Pixel Accuracy (%)', round(mean_pix_acc, 4)])
        writer.writerow(['Pixel-Level', 'Pixel Balanced Accuracy (%)', round(mean_bal_acc, 4)])
        writer.writerow(['Pixel-Level', 'Pixel Specificity / TNR (%)', round(mean_spec, 4)])
        writer.writerow(['Pixel-Level', 'Pixel Recall / Sensitivity (%)', round(mean_rec, 4)])
        writer.writerow(['Pixel-Level', 'Pixel Precision (%)', round(mean_prec, 4)])
        writer.writerow(['Pixel-Level', 'Pixel F1-Score / Dice (%)', round(mean_f1, 4)])
        writer.writerow(['Pixel-Level', 'Pixel IoU / Jaccard (%)', round(mean_iou, 4)])
        writer.writerow(['Pixel-Level', 'Mean Absolute Error (MAE)', round(mean_mae, 6)])
        writer.writerow(['Document-Level', 'Document Classification Accuracy (%)', round(mean_img_acc, 4)])
        writer.writerow(['Document-Level', 'Document Detection Precision (%)', round(img_prec, 4)])
        writer.writerow(['Document-Level', 'Document Detection Recall (%)', round(img_rec, 4)])
        writer.writerow(['Document-Level', 'Document Detection F1-Score (%)', round(img_f1, 4)])

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 68 + "\n")
        f.write(f"TEST SET BENCHMARK REPORT: {dataset_name.upper()}\n")
        f.write("=" * 68 + "\n")
        f.write(f"Checkpoint Used                 : {os.path.basename(model_path)}\n")
        f.write(f"Test Samples Evaluated          : {num_samples}\n")
        f.write(f"Detection Threshold             : {args.threshold}\n")
        f.write("-" * 68 + "\n")
        f.write("[PIXEL-LEVEL SEGMENTATION METRICS]\n")
        f.write(f"  * Pixel Accuracy              : {mean_pix_acc:.2f}%\n")
        f.write(f"  * Pixel Balanced Accuracy     : {mean_bal_acc:.2f}%\n")
        f.write(f"  * Pixel Specificity (TNR)     : {mean_spec:.2f}%\n")
        f.write(f"  * Pixel Recall (Sensitivity)  : {mean_rec:.2f}%\n")
        f.write(f"  * Pixel Precision             : {mean_prec:.2f}%\n")
        f.write(f"  * Pixel F1-Score (Dice)       : {mean_f1:.2f}%\n")
        f.write(f"  * Pixel IoU (Jaccard)         : {mean_iou:.2f}%\n")
        f.write(f"  * Mean Absolute Error (MAE)   : {mean_mae:.5f}\n")
        f.write("-" * 68 + "\n")
        f.write("[DOCUMENT-LEVEL CLASSIFICATION METRICS]\n")
        f.write(f"  * Document Classification Acc : {mean_img_acc:.2f}%\n")
        f.write(f"  * Document Detection Precision: {img_prec:.2f}%\n")
        f.write(f"  * Document Detection Recall   : {img_rec:.2f}% (Fraud Capture Rate)\n")
        f.write(f"  * Document Detection F1-Score : {img_f1:.2f}%\n")
        f.write("=" * 68 + "\n")

    return results_data


def main():
    parser = argparse.ArgumentParser(description="Test Dataset Evaluation & Benchmarking for Document Forgery Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'doctamper', 'doctamper_scd', 'cocoglide', 'splicing', 'casia', 'casia1', 'recodai'],
                        help="Test dataset to evaluate ('all' benchmarks all available test sets)")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to .pth checkpoint (default: auto-picks best model in saved_models_dir)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Maximum test samples to evaluate per dataset (default: all)')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Classification probability threshold for binary mask (default: 0.35)')
    parser.add_argument('--output-dir', type=str, default='testingreport',
                        help='Output folder where test results will be saved (default: testingreport)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Auto-resolve model path if not specified
    model_path = args.model_path
    if not model_path:
        saved_dir = config['paths']['saved_models_dir']
        candidates = sorted(glob.glob(os.path.join(saved_dir, "best_model_*.pth")), key=os.path.getmtime, reverse=True)
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(saved_dir, "*.pth")), key=os.path.getmtime, reverse=True)
        if candidates:
            model_path = candidates[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in '{saved_dir}' to evaluate.")

    print("=" * 68)
    print("  Document Forgery Framework - Test Set Evaluation Engine")
    print(f"  Architecture    : {config['model']['name']} ({config['model']['encoder']})")
    print(f"  Target Mode     : {args.dataset.upper()}")
    print(f"  Checkpoint Path : {model_path}")
    print(f"  Device          : {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"  Threshold       : {args.threshold}")
    print(f"  Output Folder   : {args.output_dir}")
    print("=" * 68)

    # Load Model (ForgeryNet) once
    model = ForgeryNet(
        out_channels=config['model']['out_channels'],
        encoder_name=config['model']['encoder'],
        dropout_prob=0.0
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (Training Val IoU: {checkpoint.get('val_iou', 0)*100:.2f}%)")
    else:
        model.load_state_dict(checkpoint)
        print(" Loaded raw state dict into ForgeryNet.")

    model.eval()

    if args.dataset == 'all':
        datasets_to_run = ['doctamper', 'doctamper_scd', 'casia', 'cocoglide', 'splicing', 'casia1']
    else:
        datasets_to_run = [args.dataset]

    all_results = []
    for dname in datasets_to_run:
        res = evaluate_dataset(dname, model, config, args, device, model_path)
        if res:
            all_results.append(res)

    # If multiple datasets evaluated, generate Master Summary Report
    if len(all_results) > 1:
        master_txt = os.path.join(args.output_dir, "MASTER_ALL_TEST_DATASETS_SUMMARY.txt")
        master_csv = os.path.join(args.output_dir, "MASTER_ALL_TEST_DATASETS_SUMMARY.csv")
        master_json = os.path.join(args.output_dir, "MASTER_ALL_TEST_DATASETS_SUMMARY.json")

        print("\n" + "=" * 90)
        print("📊 MASTER BENCHMARK SUMMARY ACROSS ALL TEST DATASETS")
        print("=" * 90)
        print(f"{'Dataset':<16} | {'Samples':<8} | {'Pixel Acc':<10} | {'Pixel Recall':<12} | {'Pixel IoU':<10} | {'Doc Recall':<10} | {'Doc F1':<8}")
        print("-" * 90)
        for r in all_results:
            print(f"{r['test_dataset']:<16} | {r['num_samples_evaluated']:<8} | {r['pixel_accuracy_percent']:<9.2f}% | {r['pixel_recall_sensitivity_percent']:<11.2f}% | {r['pixel_iou_jaccard_percent']:<9.2f}% | {r['document_level_detection_recall_percent']:<9.2f}% | {r['document_level_detection_f1_percent']:<7.2f}%")
        print("=" * 90)

        with open(master_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("MASTER BENCHMARK SUMMARY ACROSS ALL TEST DATASETS\n")
            f.write(f"Checkpoint Used: {os.path.basename(model_path)}\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'Dataset':<16} | {'Samples':<8} | {'Pixel Acc':<10} | {'Pixel Recall':<12} | {'Pixel IoU':<10} | {'Doc Recall':<10} | {'Doc F1':<8}\n")
            f.write("-" * 90 + "\n")
            for r in all_results:
                f.write(f"{r['test_dataset']:<16} | {r['num_samples_evaluated']:<8} | {r['pixel_accuracy_percent']:<9.2f}% | {r['pixel_recall_sensitivity_percent']:<11.2f}% | {r['pixel_iou_jaccard_percent']:<9.2f}% | {r['document_level_detection_recall_percent']:<9.2f}% | {r['document_level_detection_f1_percent']:<7.2f}%\n")
            f.write("=" * 90 + "\n")

        with open(master_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Samples Evaluated', 'Pixel Accuracy (%)', 'Pixel Balanced Acc (%)', 'Pixel Specificity (%)', 'Pixel Recall (%)', 'Pixel Precision (%)', 'Pixel F1-Score (%)', 'Pixel IoU (%)', 'MAE', 'Document Acc (%)', 'Document Recall (%)', 'Document Precision (%)', 'Document F1-Score (%)'])
            for r in all_results:
                writer.writerow([
                    r['test_dataset'],
                    r['num_samples_evaluated'],
                    r['pixel_accuracy_percent'],
                    r['pixel_balanced_accuracy_percent'],
                    r['pixel_specificity_percent'],
                    r['pixel_recall_sensitivity_percent'],
                    r['pixel_precision_percent'],
                    r['pixel_f1_score_dice_percent'],
                    r['pixel_iou_jaccard_percent'],
                    r['pixel_mae'],
                    r['document_level_classification_accuracy_percent'],
                    r['document_level_detection_recall_percent'],
                    r['document_level_detection_precision_percent'],
                    r['document_level_detection_f1_percent']
                ])

        master_tex = os.path.join(output_dir, 'MASTER_ALL_TEST_DATASETS_SUMMARY.tex')
        with open(master_tex, 'w', encoding='utf-8') as f:
            f.write("% MASTER BENCHMARK SUMMARY ACROSS ALL TEST DATASETS (LaTeX Table)\n")
            f.write("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n")
            f.write("\\caption{Summary of Unseen Test Evaluation across All Datasets.}\n")
            f.write("\\label{tab:master_all_metrics}\n")
            f.write("\\begin{tabular}{l r rrrrrrrr rrrr}\n\\toprule\n")
            f.write("\\textbf{Dataset} & \\textbf{Samples} & \\textbf{Pix-Acc} & \\textbf{Pix-BAcc} & \\textbf{Pix-Spec} & \\textbf{Pix-Rec} & \\textbf{Pix-Prec} & \\textbf{Pix-F1} & \\textbf{Pix-IoU} & \\textbf{MAE} & \\textbf{Doc-Acc} & \\textbf{Doc-Rec} & \\textbf{Doc-Prec} & \\textbf{Doc-F1} \\\\\n")
            f.write("& & (\\%) & (\\%) & (\\%) & (\\%) & (\\%) & (\\%) & (\\%) & & (\\%) & (\\%) & (\\%) & (\\%) \\\\\n\\midrule\n")
            for r in all_results:
                clean_name = r['test_dataset'].replace('_', '\\_')
                f.write(f"{clean_name:<14} & {r['num_samples_evaluated']:,} & {r['pixel_accuracy_percent']:.2f} & {r['pixel_balanced_accuracy_percent']:.2f} & {r['pixel_specificity_percent']:.2f} & {r['pixel_recall_sensitivity_percent']:.2f} & {r['pixel_precision_percent']:.2f} & {r['pixel_f1_score_dice_percent']:.2f} & {r['pixel_iou_jaccard_percent']:.2f} & {r['pixel_mae']:.4f} & {r['document_level_classification_accuracy_percent']:.2f} & {r['document_level_detection_recall_percent']:.2f} & {r['document_level_detection_precision_percent']:.2f} & {r['document_level_detection_f1_percent']:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

        print(f"\n[OK] Master summary exported to:")
        print(f"  +-- Text Summary : {master_txt}")
        print(f"  +-- CSV Table    : {master_csv}")
        print(f"  +-- JSON Aggreg  : {master_json}")
        print(f"  +-- LaTeX Table  : {master_tex}\n")


if __name__ == '__main__':
    main()
