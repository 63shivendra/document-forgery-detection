import os
import argparse
import yaml
import torch
from src.models.dual_branch_unet import DualBranchUNet

def export_to_onnx(model_path=None, dataset_name="doctamper", output_dir="edgeai", config_path="config.yaml"):
    """
    Exports trained PyTorch model to ONNX format (.onnx) for Edge AI deployment.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cpu'  # Export on CPU for universal ONNX runtime compatibility
    os.makedirs(output_dir, exist_ok=True)

    if model_path is None:
        model_path = os.path.join(config['paths']['saved_models_dir'], f"best_model_{dataset_name.lower()}.pth")

    print(f"==================================================")
    print(f"  ONNX Model Exporter for Edge AI Deployment       ")
    print(f"  Source Checkpoint : {model_path}")
    print(f"  Output Directory  : {output_dir}")
    print(f"==================================================")

    # Instantiate Model
    model = DualBranchUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_filters=config['model']['base_filters'],
        num_groups=config['model']['num_groups'],
        dropout_prob=config['model']['spatial_dropout']
    ).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f" Loaded model weights from: {model_path}")
    else:
        print(f" [Notice] No existing weights found at '{model_path}'. Exporting initialized architecture to ONNX.")

    model.eval()

    # Create dummy input tensor for ONNX tracing [Batch=1, Channels=3, Height=256, Width=256]
    img_h, img_w = config['training']['image_size']
    dummy_input = torch.randn(1, 3, img_h, img_w, device=device)

    onnx_filename = f"model_{dataset_name.lower()}.onnx"
    onnx_output_path = os.path.join(output_dir, onnx_filename)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_document'],
        output_names=['forgery_mask_logits'],
        dynamic_axes={
            'input_document': {0: 'batch_size'},
            'forgery_mask_logits': {0: 'batch_size'}
        }
    )

    print(f"\n SUCCESS! ONNX Model exported to:\n  - [file:///{os.path.abspath(onnx_output_path)}]")

def main():
    parser = argparse.ArgumentParser(description="Export Trained PyTorch Model to ONNX format for Edge AI")
    parser.add_argument('--model-path', type=str, help='Path to .pth checkpoint')
    parser.add_argument('--dataset', type=str, default='doctamper', help='Dataset name identifier')
    parser.add_argument('--output-dir', type=str, default='edgeai', help='Folder to save .onnx file')
    args = parser.parse_args()

    export_to_onnx(model_path=args.model_path, dataset_name=args.dataset, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
