import os
import random
import logging
import quackosm as qosm
from shapely.geometry import box
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
# Import the regions module
from nigeria_regions import get_regions, get_southwest_nigeria, get_major_cities

# from test_inference import run_live_test
# from generate_refs import generate_ref_files

# --Pre training data save directory
script_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(script_dir, "nigeria-address-ner")

# --- Pre. CONFIGURE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_log.log"),  # Saves to file
        logging.StreamHandler(),  # Prints to console
    ],
)
logger = logging.getLogger(__name__)

# ---Pre. Load the seqeval metric
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Define label names for mapping IDs back to strings
    label_list = [
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

    # Clean up predictions and labels (removing the -100 ignore index)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute overall and per-category metrics
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # 2. Log specific performance for Nigeria-critical categories
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


# 1. DATA EXTRACTION & MERGING
def get_nigeria_master_data(pbf_path):
    """
    Process Nigeria in regions to avoid memory issues
    """
    # Define Nigerian regions (you can adjust these bounds)
    regions = [
        {"name": "lagos", "bbox": box(2.5, 6.0, 4.5, 7.0)},  # Lagos metro
        {"name": "ibadan", "bbox": box(3.5, 7.0, 4.5, 8.0)},  # Ibadan area
        {"name": "abuja", "bbox": box(7.0, 8.5, 8.0, 9.5)},  # Abuja area
        {"name": "kano", "bbox": box(8.0, 11.5, 9.5, 13.0)},  # Kano region
        {"name": "portharcourt", "bbox": box(6.5, 4.5, 7.5, 5.5)},  # Port Harcourt
    ]

    all_data = []

    for region in regions:
        try:
            logger.info(f"Processing region: {region['name']}")

            # Process one region at a time with geometry filter
            gdf = qosm.convert_pbf_to_geodataframe(
                pbf_path,
                tags_filter={
                    "building": True,
                },
                geometry_filter=region["bbox"],  # This limits the area processed
            )

            if len(gdf) > 0:
                # Extract address data
                rows = []
                for _, row in gdf.iterrows():
                    tags = row.get("tags", {})

                    # Check if it has address components
                    if tags.get("addr:street") or tags.get("name"):
                        rows.append(
                            {
                                "addr:housenumber": tags.get("addr:housenumber", ""),
                                "addr:street": tags.get("addr:street", ""),
                                "addr:city": tags.get("addr:city", "")
                                or region["name"].title(),
                                "name": tags.get("name", ""),
                            }
                        )

                region_df = pd.DataFrame(rows)
                all_data.append(region_df)
                logger.info(f"Region {region['name']}: {len(region_df)} records")

        except Exception as e:
            logger.error(f"Failed on region {region['name']}: {e}")
            continue

    # Combine all regions
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df = master_df.dropna(subset=["addr:street", "name"], how="all").fillna(
            ""
        )
        logger.info(f"Total records: {len(master_df)}")
        return master_df
    else:
        raise Exception("No data could be extracted from any region")


# def get_nigeria_master_data(pbf_path):
#     """
#     Extract all relevant features in one go
#     """
#     # Combined tags filter
#     tags_filter = {
#         "building": True,
#         "amenity": ["bank", "fuel", "hospital", "place_of_worship", "school"],
#         "shop": True,  # Optional: include shops as potential landmarks
#     }

#     # Extract everything at once
#     gdf = qosm.convert_pbf_to_geodataframe(
#         pbf_path,
#         tags_filter=tags_filter,
#         # Optional: limit to specific region
#         geometry_filter=box(2.5, 5.5, 7.5, 7.5), # box(2.5, 6.5, 3.5, 7.5)
#         # rows_per_batch=500000  # Reduce batch size
#     )

#     logger.info(f"Extracted {len(gdf)} total features")

#     # Process into your training format
#     rows = []
#     for _, row in gdf.iterrows():
#         tags = row.get('tags', {})

#         # Determine if it's a building or POI
#         if 'building' in tags:
#             # It's a building
#             rows.append({
#                 'addr:housenumber': tags.get('addr:housenumber', ''),
#                 'addr:street': tags.get('addr:street', ''),
#                 'addr:city': tags.get('addr:city', ''),
#                 'name': ''
#             })
#         elif any(k in tags for k in ['amenity', 'shop']):
#             # It's a POI/landmark
#             rows.append({
#                 'name': tags.get('name', ''),
#                 'addr:street': tags.get('addr:street', ''),
#                 'addr:city': tags.get('addr:city', ''),
#                 'addr:housenumber': ''
#             })

#     master_df = pd.DataFrame(rows)
#     master_df = master_df.dropna(subset=['addr:street', 'name'], how='all').fillna('')

#     return master_df


# 2. BIO TAGGING & AUGMENTATION
def create_bio_data(row, label_list):
    tokens, labels = [], []
    mapping = {
        "name": "LANDMARK",
        "addr:housenumber": "HOUSE",
        "addr:street": "STREET",
        "addr:city": "CITY",
    }

    # Optional connector for landmarks
    connectors = ["Opposite", "Near", "Beside", "Behind", "By"]
    if row["name"] and random.random() > 0.4:
        tokens.append(random.choice(connectors))
        labels.append("O")

    for col, label in mapping.items():
        val = str(row[col]).strip()
        if not val or val == "nan":
            continue

        # Augmentation: Random lowercasing
        words = val.split()
        for i, word in enumerate(words):
            if random.random() > 0.5:
                word = word.lower()
            tokens.append(word.replace(",", ""))  # Remove noisy commas
            labels.append(f"B-{label}" if i == 0 else f"I-{label}")

    return {"tokens": tokens, "ner_tags": labels}


# --- Pre-3: DYNAMIC FILENAME LOOKUP ---
def get_latest_pbf_filename(directory="latestOsm"):
    """Reads the pbf filename from any .txt file in the specified directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist.")
        return None

    # Search for any .txt file in the directory
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            txt_path = os.path.join(directory, file)
            try:
                with open(txt_path, "r") as f:
                    # Read first line, strip whitespace/newlines
                    filename = f.read().strip()
                    if filename.endswith(".osm.pbf"):
                        logger.info(f"Found PBF filename '{filename}' in {txt_path}")
                        return os.path.join(directory, filename)
            except Exception as e:
                logger.error(f"Could not read {txt_path}: {e}")
    return None


# 3. TRAINING PIPELINE
def main():
    # Look for filename in latestOsm/pbf_name.txt (or any .txt in that folder)
    pbf_filename = get_latest_pbf_filename("latestOsm")
    print(f"pbf_filename: {pbf_filename}")
    if not pbf_filename or not os.path.exists(pbf_filename):
        logger.error(
            f"Error: Could not find a valid .osm.pbf filename in latestOsm/ or file {pbf_filename} is missing."
        )
    else:
        PBF_FILE = pbf_filename  # Now uses the name read from your .txt file
        MODEL_NAME = "bert-base-multilingual-cased"
        OUTPUT_DIR = save_dir

    # Step A: Prepare Dataset
    logger.info(f"Loading PBF file '{pbf_filename}'.")
    df = get_nigeria_master_data(PBF_FILE)
    label_list = [
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
    label2id = {l: i for i, l in enumerate(label_list)}

    raw_data = [create_bio_data(row, label_list) for _, row in df.iterrows()]
    for item in raw_data:
        item["ner_tags"] = [label2id[tag] for tag in item["ner_tags"]]

    # Build HF Dataset
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_list)),
        }
    )
    dataset = Dataset.from_list(raw_data, features=features).train_test_split(
        test_size=0.2
    )

    # Step B: Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            labels.append([label[w] if w is not None else -100 for w in word_ids])
        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = dataset.map(tokenize_and_align, batched=True)

    # Step C: Model Training
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list)
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=save_dir,
            evaluation_strategy="epoch",
            logging_steps=10,
        ),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Step D: Save and Export
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    dataset["train"].to_json("nigeria_train_set.jsonl")
    logger.info(f"Success! Model and Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    logger.info("Training complete. Invoking external inference test...")
    # run_live_test(save_dir)
