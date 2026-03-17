from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI(title="Nigeria Address Extractor API")

# Load your fine-tuned model from the local directory
model_path = "./nigeria-address-ner"
try:
    nlp_pipeline = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    nlp_pipeline = None

# Define input data structure using Pydantic
class AddressRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract_address(request: AddressRequest):
    if not nlp_pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Process the text and return structured JSON
    results = nlp_pipeline(request.text)
    return {
        "original_text": request.text,
        "entities": [
            {
                "word": res["word"],
                "label": res["entity_group"],
                "confidence": round(float(res["score"]), 4)
            } for res in results
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
