from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
import torch
import os

def convert_model(model_path, output_path):
    """
    Convert your trained model to ONNX format
    """
    print(f"Converting model from {model_path} to {output_path}")
    
    # Load the model and export to ONNX
    model = ORTModelForTokenClassification.from_pretrained(
        model_path, 
        export=True,
        provider="CPUExecutionProvider"  # Use CPU for export to avoid GPU issues
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Save the ONNX model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Model converted successfully and saved to {output_path}")
    
    # Verify the export
    print("\nVerifying exported model...")
    test_text = "Plot 7 Admiralty Way, Lekki, Lagos"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Test with ONNX Runtime
    from optimum.onnxruntime import ORTModelForTokenClassification
    ort_model = ORTModelForTokenClassification.from_pretrained(output_path)
    
    outputs = ort_model(**inputs)
    print(f"✅ Model verification successful. Output shape: {outputs.logits.shape}")

if __name__ == "__main__":
    model_path = "./nigeria-address-ner"
    output_path = "./nigeria-address-ner-onnx"
    convert_model(model_path, output_path)