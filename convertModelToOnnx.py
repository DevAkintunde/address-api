from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

model_path = "./nigeria-address-ner"
save_directory = "./nigeria-address-ner-onnx"

# Load and export in one step
ort_model = ORTModelForTokenClassification.from_pretrained(
    model_path, 
    export=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Save the ONNX model
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"✅ Model exported to {save_directory}")