import sys
import os
import torch

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("=" * 60)
    print("DEVICE & ENVIRONMENT DIAGNOSTIC CHECK")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available in PyTorch: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        dev_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"Primary GPU: {dev_name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total VRAM: {vram_gb:.2f} GB ({props.total_memory / (1024 ** 2):.0f} MB)")
        print(f"Multi-Processor Count: {props.multi_processor_count}")
    else:
        print("WARNING: CUDA is NOT available to PyTorch!")
        return

    print("\n" + "=" * 60)
    print("MODEL & CODE COMPATIBILITY TEST ON RTX 5080")
    print("=" * 60)
    try:
        from src.models.forgery_net import ForgeryNet
        print("Successfully imported ForgeryNet.")
        
        device = 'cuda'
        model = ForgeryNet(out_channels=1, encoder_name="efficientnet-b2", dropout_prob=0.2).to(device)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model instantiated successfully: {total_params:.2f}M parameters.")

        # Test forward pass with AMP at batch_size=24, res=384x384
        batch_test = 24
        dummy_input = torch.randn(batch_test, 3, 384, 384, device=device)
        print(f"Testing forward pass with batch size = {batch_test}, shape: {list(dummy_input.shape)}...")
        
        with torch.amp.autocast('cuda', enabled=True):
            out = model(dummy_input)
        
        print(f"Forward pass test: SUCCESS. Output shape: {list(out.shape)}")
        alloc_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"VRAM Allocated for batch size {batch_test}: {alloc_mem:.1f} MB (Reserved: {reserved_mem:.1f} MB)")

        # Test ONNX export capability
        print("\nTesting ONNX export...")
        os.makedirs("bigpower", exist_ok=True)
        onnx_test_path = "bigpower/test_model.onnx"
        dummy_onnx_input = torch.randn(1, 3, 384, 384, device=device)
        model.eval()
        torch.onnx.export(
            model, dummy_onnx_input, onnx_test_path,
            export_params=True, opset_version=14,
            do_constant_folding=True,
            dynamo=False,
            input_names=['input_document'],
            output_names=['forgery_mask_logits'],
            dynamic_axes={'input_document': {0: 'batch_size'},
                          'forgery_mask_logits': {0: 'batch_size'}}
        )
        if os.path.exists(onnx_test_path):
            size_mb = os.path.getsize(onnx_test_path) / (1024 ** 2)
            print(f"ONNX export test: SUCCESS (File size: {size_mb:.2f} MB)")
            os.remove(onnx_test_path)
            
    except Exception as e:
        print(f"Compatibility test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
