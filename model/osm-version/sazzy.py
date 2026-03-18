"""
Nigerian Address NER Training Script
Trains a BERT-based model to extract address components from unstructured text
"""

import os
import sys
import argparse
import random
import logging
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset, Features, Sequence, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import quackosm as qosm
from shapely.geometry import box

# Import the regions module
from nigeria_regions import get_regions, get_southwest_nigeria, get_major_cities

# ===================== DEFAULT CONFIGURATION =====================
DEFAULT_MODEL = "bert-base-multilingual-cased"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_REGION = "lagos"

# Paths
script_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(script_dir, "nigeria-address-ner")
os.makedirs(save_dir, exist_ok=True)

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(script_dir, "training_log.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ===================== LABELS =====================
LABEL_LIST = [
    "O",
    "B-HOUSE",
    "I-HOUSE",
    "B-STREET",
    "I-STREET",
    "B-CITY",
    "I-CITY",
    "B-LANDMARK",
    "I-LANDMARK",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

# ===================== ARGUMENT PARSING =====================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Nigerian Address NER Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sazzy.py --region southwest
  python sazzy.py --region "lagos,abuja,kano"
  python sazzy.py --region all --epochs 5 --batch-size 32
  python sazzy.py --region major_cities --model xlm-roberta-base
  
Region options:
  - "all": Entire Nigeria
  - "southwest": Lagos, Ogun, Oyo, Osun, Ondo, Ekiti
  - "major_cities": All major Nigerian cities
  - "zones": All 6 geopolitical zones
  - Comma-separated list: e.g., "lagos,abuja,kano"
  - State names: "lagos", "ogun", "oyo", etc.
        """
    )
    
    parser.add_argument(
        "--region", "-r",
        type=str,
        default=DEFAULT_REGION,
        help=f"Region to process (default: {DEFAULT_REGION})"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Transformer model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Validation set size (default: {DEFAULT_TEST_SIZE})"
    )
    
    parser.add_argument(
        "--pbf-dir",
        type=str,
        default="latestOsm",
        help="Directory containing OSM PBF file (default: latestOsm)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=save_dir,
        help=f"Output directory for model (default: {save_dir})"
    )
    
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List all available regions and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def list_available_regions():
    """Print all available region options"""
    from nigeria_regions import GEOPOLITICAL_ZONES, STATES, MAJOR_CITIES, REGION_GROUPS
    
    print("\n" + "="*60)
    print("AVAILABLE REGIONS")
    print("="*60)
    
    print("\n📌 Predefined Groups:")
    print("  - all (entire Nigeria)")
    print("  - southwest (Lagos, Ogun, Oyo, Osun, Ondo, Ekiti)")
    print("  - major_cities (all major cities)")
    print("  - zones (all 6 geopolitical zones)")
    print("  - southwest_states")
    print("  - southern_nigeria")
    print("  - northern_nigeria")
    
    print("\n📌 Geopolitical Zones:")
    for key, zone in GEOPOLITICAL_ZONES.items():
        print(f"  - {key}: {zone['name']}")
    
    print("\n📌 States (36 + FCT):")
    # Print in columns for better readability
    state_items = list(STATES.items())
    for i in range(0, len(state_items), 4):
        row = state_items[i:i+4]
        print("  " + "  ".join([f"{key}: {info['name']:15}" for key, info in row]))
    
    print("\n📌 Major Cities:")
    city_items = list(MAJOR_CITIES.items())
    for i in range(0, len(city_items), 3):
        row = city_items[i:i+3]
        print("  " + "  ".join([f"{key}: {info['name']:20}" for key, info in row]))
    
    print("\n" + "="*60)
    print("Usage Examples:")
    print("  python sazzy.py --region lagos")
    print("  python sazzy.py --region lagos,abuja,kano")
    print("  python sazzy.py --region southwest --epochs 5")
    print("="*60)

def parse_region_input(region_str):
    """
    Parse region input from command line
    Handles: "southwest", "lagos,abuja,kano", ["lagos", "abuja"], etc.
    """
    if not region_str:
        return DEFAULT_REGION
    
    # If it's a comma-separated string, split it
    if "," in region_str:
        return [r.strip() for r in region_str.split(",")]
    
    # If it's a single value, return as is
    return region_str

# ===================== METRICS =====================
metric = evaluate.load("seqeval")

def compute_metrics(p):
    """Compute evaluation metrics for NER"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Clean up predictions and labels
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute metrics
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Log category performance
    logger.info("--- CATEGORY PERFORMANCE REPORT ---")
    for category in ["HOUSE", "STREET", "LANDMARK", "CITY"]:
        if category in results:
            f1 = results[category]["f1"]
            precision = results[category]["precision"]
            logger.info(
                f"Category: {category:10} | F1: {f1:.4f} | Precision: {precision:.4f}"
            )
        else:
            logger.warning(f"Category {category} was not found in validation data.")

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ===================== DATA EXTRACTION =====================
def get_latest_pbf_filename(directory="latestOsm"):
    """Read PBF filename from text file in directory"""
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist.")
        return None

    # First look for .txt files with filename
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            txt_path = os.path.join(directory, file)
            try:
                with open(txt_path, "r") as f:
                    filename = f.read().strip()
                    if filename.endswith(".osm.pbf"):
                        full_path = os.path.join(directory, filename)
                        if os.path.exists(full_path):
                            logger.info(f"Found PBF filename '{filename}' in {txt_path}")
                            return full_path
            except Exception as e:
                logger.error(f"Could not read {txt_path}: {e}")
    
    # If no text file, look for PBF files directly
    for file in os.listdir(directory):
        if file.endswith(".osm.pbf"):
            full_path = os.path.join(directory, file)
            logger.info(f"Found PBF file directly: {file}")
            return full_path
    
    return None

def get_nigeria_master_data(pbf_path, region_param="southwest"):
    """
    HYBRID APPROACH: Extract Nigerian address data with one scan per region
    Combines building and POI extraction in a single pass
    """
    try:
        logger.info(f"Extracting data with region parameter: {region_param}")
        
        # Determine which regions to process
        if region_param == "all":
            regions = [{"name": "nigeria", "bbox": box(2.5, 4.0, 14.5, 14.0)}]
        elif region_param == "southwest":
            regions = get_southwest_nigeria()
        elif region_param == "major_cities":
            regions = get_major_cities()
        elif region_param == "zones":
            regions = get_regions("zones")
        elif isinstance(region_param, list):
            regions = get_regions("custom", region_param)
        else:
            # Try as a group name
            try:
                regions = get_regions("group", [region_param])
            except ValueError:
                # If not a group, try as a single custom region
                regions = get_regions("custom", [region_param])
        
        logger.info(f"Processing {len(regions)} regions")
        
        all_buildings = []
        all_pois = []
        total_features = 0
        
        for region_idx, region in enumerate(regions, 1):
            logger.info(f"[{region_idx}/{len(regions)}] Processing region: {region['name']}")
            
            try:
                # SINGLE CALL per region - gets everything in one pass
                combined_gdf = qosm.convert_pbf_to_geodataframe(
                    pbf_path,
                    tags_filter={
                        "building": True,    # All buildings
                        "amenity": True,     # All amenities
                        "shop": True,        # All shops
                        "tourism": True      # All tourism features
                    },
                    geometry_filter=region['bbox']
                )
                
                region_features = len(combined_gdf)
                total_features += region_features
                
                if region_features == 0:
                    logger.info(f"  No features found in {region['name']}")
                    continue
                
                logger.info(f"  Found {region_features} total features")
                
                # Process this region's data in a single loop
                region_buildings = 0
                region_pois = 0
                
                for _, row in combined_gdf.iterrows():
                    tags = row.get('tags', {})
                    
                    # Check if it's a building with address components
                    if 'building' in tags:
                        if tags.get('addr:housenumber') or tags.get('addr:street'):
                            all_buildings.append({
                                'addr:housenumber': tags.get('addr:housenumber', ''),
                                'addr:street': tags.get('addr:street', ''),
                                'addr:city': tags.get('addr:city', '') or region['name'],
                                'name': ''
                            })
                            region_buildings += 1
                    
                    # Check if it's a POI with a name
                    elif any(k in tags for k in ['amenity', 'shop', 'tourism']):
                        if tags.get('name'):
                            all_pois.append({
                                'addr:housenumber': '',
                                'addr:street': tags.get('addr:street', ''),
                                'addr:city': tags.get('addr:city', '') or region['name'],
                                'name': tags.get('name', '')
                            })
                            region_pois += 1
                
                logger.info(f"  Region {region['name']}: {region_buildings} buildings with addresses, {region_pois} named POIs")
                
            except Exception as e:
                logger.error(f"Error processing region {region['name']}: {e}")
                continue
        
        # Create DataFrames
        buildings_df = pd.DataFrame(all_buildings) if all_buildings else pd.DataFrame(columns=['addr:housenumber', 'addr:street', 'addr:city', 'name'])
        pois_df = pd.DataFrame(all_pois) if all_pois else pd.DataFrame(columns=['addr:housenumber', 'addr:street', 'addr:city', 'name'])
        
        # Combine and clean
        master_df = pd.concat([buildings_df, pois_df], ignore_index=True)
        master_df = master_df.dropna(subset=['addr:street', 'name'], how='all').fillna('')
        
        # Final stats
        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total features processed: {total_features}")
        logger.info(f"Buildings with addresses: {len(buildings_df)}")
        logger.info(f"Named POIs: {len(pois_df)}")
        logger.info(f"Final training records: {len(master_df)}")
        logger.info("=" * 60)
        
        return master_df
        
    except Exception as e:
        logger.error(f"Failed to parse OSM data: {e}")
        raise

# ===================== DATA AUGMENTATION =====================
def create_bio_data(row):
    """Convert a row of address data to BIO-tagged tokens"""
    tokens, labels = [], []
    mapping = {
        "name": "LANDMARK",
        "addr:housenumber": "HOUSE",
        "addr:street": "STREET",
        "addr:city": "CITY",
    }

    # Add connectors for landmarks (50% of the time)
    connectors = ["Opposite", "Near", "Beside", "Behind", "By", "Close to"]
    if row["name"] and random.random() > 0.5:
        tokens.append(random.choice(connectors))
        labels.append("O")

    # Process each component
    for col, label in mapping.items():
        val = str(row[col]).strip()
        if not val or val == "nan" or val == "":
            continue

        # Split into words
        words = val.split()
        for i, word in enumerate(words):
            # Randomly lowercase for augmentation
            if random.random() > 0.3:
                word = word.lower()
            # Remove noisy punctuation
            word = word.replace(",", "").replace(".", "").replace(";", "")
            if word:  # Only add non-empty words
                tokens.append(word)
                labels.append(f"B-{label}" if i == 0 else f"I-{label}")

    return {"tokens": tokens, "ner_tags": labels}

# ===================== MAIN TRAINING PIPELINE =====================
def main():
    """Main training pipeline with command-line arguments"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle special commands
    if args.list_regions:
        list_available_regions()
        return
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse region input
    region_param = parse_region_input(args.region)
    
    logger.info("=" * 60)
    logger.info("NIGERIAN ADDRESS NER TRAINING")
    logger.info("=" * 60)
    logger.info(f"Region: {args.region}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"PBF directory: {args.pbf_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)
    
    # 1. Find PBF file
    logger.info("Step 1: Locating PBF file")
    pbf_filename = get_latest_pbf_filename(args.pbf_dir)
    
    if not pbf_filename or not os.path.exists(pbf_filename):
        logger.error(f"Error: Could not find a valid .osm.pbf file in {args.pbf_dir}")
        logger.info("Please place your Nigeria OSM PBF file in the specified directory")
        return
    
    logger.info(f"Using PBF file: {pbf_filename}")
    
    # 2. Extract data
    logger.info(f"Step 2: Extracting address data")
    df = get_nigeria_master_data(pbf_filename, region_param=region_param)
    
    if len(df) == 0:
        logger.error("No data extracted! Check your PBF file and filters.")
        logger.info("Try with a different region parameter or check if your PBF contains address data.")
        return
    
    # 3. Convert to BIO format
    logger.info(f"Step 3: Converting to BIO format ({len(df)} records)")
    raw_data = [create_bio_data(row) for _, row in df.iterrows()]
    
    # Filter out empty examples
    raw_data = [item for item in raw_data if len(item["tokens"]) > 0]
    logger.info(f"  Created {len(raw_data)} valid training examples")
    
    # Convert labels to IDs
    for item in raw_data:
        item["ner_tags"] = [LABEL2ID[tag] for tag in item["ner_tags"]]

    # 4. Create HuggingFace dataset
    logger.info("Step 4: Creating HuggingFace dataset")
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=LABEL_LIST)),
        }
    )
    dataset = Dataset.from_list(raw_data, features=features).train_test_split(
        test_size=args.test_size, seed=42
    )

    logger.info(f"  Training set: {len(dataset['train'])} examples")
    logger.info(f"  Validation set: {len(dataset['test'])} examples")

    # 5. Tokenize
    logger.info("Step 5: Tokenizing with alignment")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            max_length=128
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            labels.append([label[w] if w is not None else -100 for w in word_ids])
        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = dataset.map(tokenize_and_align, batched=True)

    # 6. Load model
    logger.info(f"Step 6: Loading model {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, 
        num_labels=len(LABEL_LIST),
        ignore_mismatched_sizes=True
    )

    # 7. Training arguments
    logger.info("Step 7: Configuring training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard if not needed
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 9. Train
    logger.info("Step 8: Starting training")
    trainer.train()

    # 10. Save
    logger.info("Step 9: Saving model and artifacts")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    dataset["train"].to_json(os.path.join(args.output_dir, "train_dataset.jsonl"))
    dataset["test"].to_json(os.path.join(args.output_dir, "test_dataset.jsonl"))
    
    # Save configuration
    with open(os.path.join(args.output_dir, "training_config.txt"), "w") as f:
        f.write(f"Region: {args.region}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Training examples: {len(dataset['train'])}\n")
        f.write(f"Validation examples: {len(dataset['test'])}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
    
    logger.info(f"✅ Training complete! Model saved to {args.output_dir}")
    
    # Final evaluation
    logger.info("Final evaluation on validation set:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()