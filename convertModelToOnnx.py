from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

model_path = "./nigeria-address-ner"
save_directory = "./nigeria-address-ner-onnx"
save_quantized_directory = "./nigeria-address-ner-onnx-quantized"

# Load and export in one step
ort_model = ORTModelForTokenClassification.from_pretrained(
    model_path, 
    export=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    fix_mistral_regex=True  
)

# Save the ONNX model
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"✅ Model exported to {save_directory}")

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 2. Set up the quantizer
quantizer = ORTQuantizer.from_pretrained(save_directory)

# Define quantization configuration (ARM or x86/CPU)
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

# 3. Apply dynamic INT8 quantization
quantizer.quantize(
    save_dir=save_quantized_directory,
    quantization_config=qconfig
)

print(f"✅ Quantized Model exported to {save_directory}")
