"""
Model Quantization Script: Converts FP32 ONNX Model into ~10 MB INT8 Quantized Mobile ONNX Engine.
Zero PyTorch dependency at runtime.
"""

import os
import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Force UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def export_int8_mobile_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bigpower_dir = os.path.join(base_dir, "bigpower")
    edge_models_dir = os.path.join(base_dir, "edgedevice", "models")
    os.makedirs(edge_models_dir, exist_ok=True)

    input_onnx = os.path.join(bigpower_dir, "best_model_splicing_combined_efficientnet-b2_20260908_1523.onnx")
    output_onnx = os.path.join(edge_models_dir, "model_forgery_int8.onnx")

    if not os.path.exists(input_onnx):
        # Fallback to final_model onnx if best_model onnx missing
        input_onnx = os.path.join(bigpower_dir, "final_model_splicing_combined_efficientnet-b2_20260908_1523.onnx")

    if not os.path.exists(input_onnx):
        print(f"❌ Error: ONNX base model not found at {input_onnx}")
        return False

    print(f"⚡ Quantizing FP32 ONNX Model ({os.path.getsize(input_onnx) / (1024*1024):.2f} MB)...")
    print(f"   Input:  {input_onnx}")
    print(f"   Output: {output_onnx}")

    try:
        quantize_dynamic(
            model_input=input_onnx,
            model_output=output_onnx,
            weight_type=QuantType.QUInt8
        )
        
        orig_size = os.path.getsize(input_onnx) / (1024 * 1024)
        quant_size = os.path.getsize(output_onnx) / (1024 * 1024)
        
        print(f"✅ Success! INT8 Quantized Mobile Engine generated:")
        print(f"   Original Size: {orig_size:.2f} MB")
        print(f"   Quantized Size: {quant_size:.2f} MB (Compression: {(1 - quant_size/orig_size)*100:.1f}%)")
        print(f"   Saved at: {output_onnx}")
        return True
    except Exception as e:
        print(f"❌ Quantization Error: {e}")
        return False


if __name__ == "__main__":
    export_int8_mobile_model()
