import logging
from transformers import pipeline
from thefuzz import process, fuzz

# Configure logging to terminal
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load Reference Data (Generated during training from OSM)
try:
    with open("ref_streets.txt", "r") as f:
        STREET_REF = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    logger.warning("ref_streets.txt not found. Auto-correction disabled.")
    STREET_REF = []

def get_fuzzy_correction(word, reference_list, min_match_score=85):
    """Snaps a typo to the closest official OSM name."""
    if not reference_list:
        return word, 0
    
    # Finds the closest string match
    match, score = process.extractOne(word, reference_list, scorer=fuzz.token_sort_ratio)
    
    if score >= min_match_score:
        return match, score
    return word, 0

def process_address(text, model_path, conf_threshold=0.90):
    """Full Pipeline: Extraction -> Confidence Filter -> Auto-Correction"""
    
    # 1. Load Model
    extractor = pipeline(
        "ner", 
        model=model_path, 
        tokenizer=model_path, 
        aggregation_strategy="simple"
    )
    
    raw_results = extractor(text)
    final_output = []

    logger.info(f"\n[Raw Input]: {text}")

    for ent in raw_results:
        label = ent['entity_group']
        word = ent['word']
        score = ent['score']

        # 2. Apply Confidence Threshold
        if score < conf_threshold:
            logger.warning(f"  ! Low Confidence ({score:.2f}): '{word}' ignored.")
            continue

        # 3. Apply Auto-Correction (Only for Streets/Landmarks)
        corrected_word = word
        match_score = 0
        
        if label == "STREET":
            corrected_word, match_score = get_fuzzy_correction(word, STREET_REF)
        
        if match_score > 0 and corrected_word != word:
            logger.info(f"  ✓ Fixed: {word} -> {corrected_word} (Match: {match_score}%)")
        
        final_output.append({
            "label": label,
            "original": word,
            "corrected": corrected_word,
            "confidence": round(float(score), 3)
        })

    return final_output

if __name__ == "__main__":
    MODEL_DIR = "./nigeria-address-ner"
    
    # Test with a common Nigerian typo
    test_input = "Deliver to 15 Admirallty Way near Zenit Bank Lekki"
    
    structured_data = process_address(test_input, MODEL_DIR, conf_threshold=0.85)
    
    print("\n--- FINAL STRUCTURED DATA ---")
    for item in structured_data:
        print(f"{item['label']}: {item['corrected']} (Conf: {item['confidence']})")
