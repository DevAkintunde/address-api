"""
Nigerian Address NER Training Script
Trains a BERT-based model using Microsoft Building Footprints
Default: Uses Lagos region with data from ./data/Nigeria.geojsonl

All training data stays LOCAL - only downloads pre-trained model weights from Hugging Face
"""

import os
import sys
import argparse
import random
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import evaluate
import json
import torch
from torch.utils.data import DataLoader
from shapely.geometry import shape, box
from datasets import Dataset, Features, Sequence, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from tqdm import tqdm

# Import the regions module
from nigeria_regions import get_region_from_coordinates, get_regions, get_southwest_nigeria, get_major_cities

# ===================== CONFIGURATION =====================
DEFAULT_MODEL = "bert-base-multilingual-cased"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SAMPLES = 100000  # Number of buildings to sample
DEFAULT_REGION = "lagos"  # Default to Lagos
DEFAULT_DATA_PATH = os.path.join("data", "Nigeria.geojsonl")  # Default data path
LOCAL_MODEL_PATH = os.path.join("local_bert_model")  # For offline mode

# Paths
script_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(script_dir, "nigeria-address-ner")
os.makedirs(save_dir, exist_ok=True)

# Create data directory if it doesn't exist
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# ===================== LOGGING =====================
# Fix for Windows Unicode encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(script_dir, "training_log.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ===================== CHECK IF SYSTEM SUPPORT GPU =====================
if torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    logger.info("⚠️  GPU not found, using CPU (will be slower)")
    
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

# ===================== NIGERIAN STREET NAMES =====================
NIGERIAN_STREETS = [
    "Admiralty Way", "Marina", "Ozumba Mbadiwe Road", "Awolowo Road",
    "Allen Avenue", "Isaac John Street", "Opebi Road", "Toyin Street",
    "Kofo Abayomi Street", "Broad Street", "Falomo Road", "Bourdillon Road",
    "Ahmadu Bello Way", "Obafemi Awolowo Way", "Tafawa Balewa Road",
    "Muritala Mohammed Way", "Independence Road", "Liberty Road",
    "Airport Road", "New Market Road", "Bank Road", "Station Road",
    "Hospital Road", "Post Office Road",
]

# Lagos-specific cities/areas
LAGOS_AREAS = [
    "Lagos", "Ikeja", "Victoria Island", "Lekki", "Ajah", "Surulere",
    "Yaba", "Mainland", "Ikoyi", "Apapa", "Festac", "Maryland",
    "Ikorodu", "Badagry", "Epe", "Oshodi", "Mushin", "Agege"
]

# Nigerian landmarks
NIGERIAN_LANDMARKS = [
    "Eko Hotel", "TBS", "Silverbird Galleria", "National Stadium",
    "National Theatre", "Freedom Park", "University of Lagos",
    "Lekki Conservation Centre", "Marina", "Victoria Island",
    "Ikoyi Golf Club", "Lagos City Mall", "Palms Shopping Mall",
    "Computer Village", "Murtala Mohammed Airport", "Third Mainland Bridge"
]

# ===================== CUSTOM DATA COLLATOR =====================
class CustomDataCollatorForTokenClassification:
    """
    Custom data collator that ensures proper padding and tensor conversion
    """
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, features):
        # Extract features
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Pad sequences to max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(features)):
            # Pad input_ids
            padding_length = max_length - len(input_ids[i])
            padded_input_ids.append(
                input_ids[i] + [self.tokenizer.pad_token_id] * padding_length
            )
            
            # Pad attention_mask
            padded_attention_mask.append(
                attention_mask[i] + [0] * padding_length
            )
            
            # Pad labels
            padded_labels.append(
                labels[i] + [self.label_pad_token_id] * padding_length
            )
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

# ===================== ARGUMENT PARSING =====================
def parse_arguments():
    """Parse command line arguments with Lagos as default"""
    parser = argparse.ArgumentParser(
        description="Train Nigerian Address NER Model using Microsoft Building Footprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python sazzy.py                                    # Train on 100k Lagos buildings (default)
  python sazzy.py --all-buildings                    # Train on ALL Lagos buildings (~1.5M)
  python sazzy.py --samples 500000                   # Train on 500k Lagos buildings
  python sazzy.py --region all --all-buildings       # Train on ALL Nigeria (35.7M buildings)
  python sazzy.py --region lagos,abuja,kano          # Train on specific states
  python sazzy.py --offline                           # Run in offline mode
  python sazzy.py --download-model                    # Download model for offline use
  python sazzy.py --list-checkpoints                  # List available building checkpoints
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to Microsoft Building Footprints GeoJSONL file (default: {DEFAULT_DATA_PATH})"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of buildings to sample (default: {DEFAULT_SAMPLES})"
    )
    
    parser.add_argument(
        "--all-buildings",
        action="store_true",
        help="Use ALL buildings in the region (ignore samples limit)"
    )
    
    parser.add_argument(
        "--region", "-r",
        type=str,
        default=DEFAULT_REGION,
        help=f"Region to focus on: 'lagos', 'southwest', 'all', or comma-separated states (default: {DEFAULT_REGION})"
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
        "--output", "-o",
        type=str,
        default=save_dir,
        help=f"Output directory for model (default: {save_dir})"
    )
    
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download Microsoft Nigeria building footprints before training"
    )
    
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download pre-trained model for offline use"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (use locally downloaded model)"
    )
    
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available building checkpoints"
    )
    
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload buildings from GeoJSONL even if checkpoint exists"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

# ===================== CHECKPOINT MANAGEMENT =====================
def list_checkpoints(output_dir):
    """List all available building checkpoints"""
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith("buildings_") and f.endswith(".pkl")]
    
    if checkpoint_files:
        logger.info("📋 Available building checkpoints:")
        for cf in sorted(checkpoint_files):
            size_mb = os.path.getsize(os.path.join(output_dir, cf)) / (1024 * 1024)
            parts = cf.replace("buildings_", "").replace(".pkl", "").split("_")
            if len(parts) >= 2:
                region = parts[0]
                if parts[1] == "ALL":
                    samples = "ALL BUILDINGS"
                else:
                    samples = f"{parts[1]} samples"
                logger.info(f"  - {cf} ({size_mb:.1f} MB) [Region: {region}, {samples}]")
            else:
                logger.info(f"  - {cf} ({size_mb:.1f} MB)")
    else:
        logger.info("No building checkpoints found in output directory")

def get_checkpoint_path(output_dir, region, samples, all_buildings=False):
    """Generate checkpoint filename based on parameters"""
    if all_buildings:
        return os.path.join(output_dir, f"buildings_{region}_ALL.pkl")
    else:
        return os.path.join(output_dir, f"buildings_{region}_{samples}.pkl")

# ===================== DATA DOWNLOAD =====================
def download_microsoft_data(target_path):
    """Download Microsoft Building Footprints for Nigeria"""
    import urllib.request
    import zipfile
    
    url = "https://microsoftbuildingfootprints.blob.core.windows.net/geojson/Nigeria.geojsonl.zip"
    zip_path = target_path + ".zip"
    
    logger.info(f"Downloading Microsoft Nigeria building footprints from {url}")
    logger.info(f"This is a 2.3GB file and may take a while...")
    
    def download_progress(count, block_size, total_size):
        if count == 0:
            return
        downloaded = count * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        sys.stdout.write(f"\rDownloading: {percent}% ({downloaded/1e6:.1f}MB / {total_size/1e6:.1f}MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
    print()
    
    logger.info("Download complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(target_path))
    
    os.remove(zip_path)
    logger.info(f"Data extracted to: {target_path}")
    return os.path.exists(target_path)

def download_model_for_offline(model_name, save_path):
    """Download model for offline use"""
    logger.info(f"Downloading model {model_name} for offline use...")
    logger.info(f"This will save the model to: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    
    logger.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    
    logger.info("Downloading model weights (this may take a while)...")
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.save_pretrained(save_path)
    
    logger.info(f"✅ Model downloaded and saved to {save_path}")
    return True

# ===================== GEOJSONL READER =====================
def read_geojsonl_manual(file_path, filter_geom=None, max_features=None):
    """Manually read GeoJSONL file line by line"""
    import json
    from shapely.geometry import shape
    
    logger.info(f"Reading GeoJSONL file line by line: {file_path}")
    
    features = []
    total_lines = 0
    kept_features = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    logger.info(f"Total lines in file: {total_lines:,}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Reading buildings"):
            line = line.strip()
            if not line:
                continue
            
            try:
                feature = json.loads(line)
                geom = shape(feature['geometry'])
                
                if filter_geom and not geom.intersects(filter_geom):
                    continue
                
                features.append({
                    'geometry': geom,
                    'tags': {}
                })
                kept_features += 1
                
                if max_features and len(features) >= max_features:
                    break
                    
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    logger.info(f"Kept {kept_features:,} buildings in region")
    
    if not features:
        return None
    
    gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
    return gdf

def load_microsoft_buildings(file_path, sample_size=100000, region_filter="lagos", all_buildings=False):
    """
    Load Microsoft building footprints with option to get all buildings
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    logger.info(f"Loading Microsoft building footprints from: {file_path}")
    
    filter_geom = None
    if region_filter and region_filter != "all":
        logger.info(f"Filtering for region: {region_filter}")
        
        if region_filter == "lagos":
            filter_geom = box(2.5, 6.0, 4.5, 7.0)
        elif region_filter == "southwest":
            filter_geom = box(2.5, 6.0, 6.0, 9.0)
        elif "," in region_filter:
            regions = region_filter.split(",")
            region_list = get_regions("custom", [r.strip() for r in regions])
            if region_list:
                bounds = region_list[0]['bbox'].bounds
                minx, miny, maxx, maxy = bounds
                for region in region_list[1:]:
                    b = region['bbox'].bounds
                    minx = min(minx, b[0])
                    miny = min(miny, b[1])
                    maxx = max(maxx, b[2])
                    maxy = max(maxy, b[3])
                filter_geom = box(minx, miny, maxx, maxy)
        else:
            region_list = get_regions("custom", [region_filter])
            if region_list:
                filter_geom = region_list[0]['bbox']
    
    # If all_buildings is True, set max_features to None (read all)
    max_features = None if all_buildings else sample_size * 2
    
    buildings = read_geojsonl_manual(
        file_path, 
        filter_geom=filter_geom,
        max_features=max_features
    )
    
    if buildings is None or len(buildings) == 0:
        logger.error("No buildings found in the specified region")
        return None
    
    logger.info(f"Total buildings in region: {len(buildings):,}")
    
    # Only sample if not getting all buildings
    if not all_buildings and len(buildings) > sample_size:
        buildings = buildings.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled to {len(buildings):,} buildings")
    else:
        logger.info(f"Using ALL {len(buildings):,} buildings")
    
    return buildings

# ===================== METRICS =====================
metric = evaluate.load("seqeval")

def compute_metrics(p):
    """Compute evaluation metrics for NER"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    logger.info("--- CATEGORY PERFORMANCE REPORT ---")
    for category in ["HOUSE", "STREET", "LANDMARK", "CITY"]:
        if category in results:
            f1 = results[category]["f1"]
            precision = results[category]["precision"]
            logger.info(f"Category: {category:10} | F1: {f1:.4f} | Precision: {precision:.4f}")
        else:
            logger.warning(f"Category {category} was not found in validation data.")

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ===================== DATA GENERATION =====================
class NigerianAddressGenerator:
    """Generate realistic Nigerian addresses from building footprints"""
    
    def __init__(self, region="lagos"):
        self.region = region
        
        if region == "lagos" or region == "southwest":
            self.cities = LAGOS_AREAS
            self.streets = NIGERIAN_STREETS
            self.landmarks = NIGERIAN_LANDMARKS
        else:
            from nigeria_regions import STATES
            self.cities = [state["name"] for state in STATES.values()]
            self.streets = NIGERIAN_STREETS + [
                "Main Street", "Market Road", "Station Road", "Hospital Road"
            ]
            self.landmarks = NIGERIAN_LANDMARKS + [
                "Central Market", "General Hospital", "Local Government Secretariat",
                "Community Center", "Main Mosque", "Central Church"
            ]
    
    def generate_house_number(self):
        patterns = [
            f"{random.randint(1, 500)}",
            f"{random.randint(1, 50)}{random.choice(['A', 'B', 'C'])}",
            f"Plot {random.randint(1, 500)}",
            f"Block {random.randint(1, 20)}, Flat {random.randint(1, 10)}",
            f"Suite {random.randint(1, 50)}",
            f"Shop {random.randint(1, 50)}",
        ]
        return random.choice(patterns)
    
    def generate_address_from_building(self, building, region_context=True):
        centroid = building.geometry.centroid
        lat, lon = centroid.y, centroid.x
        
        if self.region == "lagos":
            city = random.choice(LAGOS_AREAS)
        elif region_context:
            try:
                city = get_region_from_coordinates(lat, lon)
            except:
                city = random.choice(self.cities)
        else:
            city = random.choice(self.cities)
        
        format_type = random.random()
        
        if format_type < 0.4:
            return {
                'addr:housenumber': self.generate_house_number(),
                'addr:street': random.choice(self.streets),
                'addr:city': city,
                'name': '',
                'latitude': lat,
                'longitude': lon,
                'region': city
            }
        elif format_type < 0.7:
            return {
                'addr:housenumber': '',
                'addr:street': random.choice(self.streets),
                'addr:city': city,
                'name': '',
                'latitude': lat,
                'longitude': lon,
                'region': city
            }
        else:
            connector = random.choice(['Near', 'Opposite', 'Beside', 'Behind', 'Close to'])
            landmark = random.choice(self.landmarks)
            return {
                'addr:housenumber': '',
                'addr:street': random.choice(self.streets) if random.random() > 0.3 else '',
                'addr:city': city,
                'name': f"{connector} {landmark}",
                'latitude': lat,
                'longitude': lon,
                'region': city
            }

# ===================== DATA CONVERSION TO BIO =====================
def create_bio_data(address_components):
    """Convert address components to BIO-tagged tokens"""
    tokens = []
    labels = []
    
    fields = [
        ('addr:housenumber', 'HOUSE'),
        ('addr:street', 'STREET'),
        ('name', 'LANDMARK'),
        ('addr:city', 'CITY')
    ]
    
    for field, entity_type in fields:
        value = address_components.get(field, '')
        if not value or pd.isna(value) or value == '':
            continue
        
        words = str(value).split()
        for i, word in enumerate(words):
            word = word.strip('.,;:()"\'')
            if word:
                tokens.append(word)
                labels.append(f"B-{entity_type}" if i == 0 else f"I-{entity_type}")
    
    return {
        'tokens': tokens,
        'ner_tags': labels
    }

# ===================== TOKENIZATION =====================
def tokenize_and_align(examples, tokenizer, max_length=128):
    """
    Tokenize and align labels with explicit padding to max_length
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
        return_tensors=None
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                if word_idx < len(label):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

# ===================== MAIN TRAINING PIPELINE =====================
def main():
    """Main training pipeline with Lagos as default"""
    
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("NIGERIAN ADDRESS NER TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data file: {args.data}")
    logger.info(f"Samples: {args.samples if not args.all_buildings else 'ALL BUILDINGS'}")
    logger.info(f"Region: {args.region}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output}")
    if args.offline:
        logger.info("Mode: OFFLINE (using local model)")
    if args.force_reload:
        logger.info("Mode: FORCE RELOAD (ignoring checkpoints)")
    logger.info("=" * 60)
    
    if args.list_checkpoints:
        list_checkpoints(args.output)
        return
    
    if args.download_model:
        download_model_for_offline(args.model, LOCAL_MODEL_PATH)
        logger.info("Model downloaded. Run again without --download-model to train.")
        return
    
    if args.download_data:
        success = download_microsoft_data(args.data)
        if not success:
            logger.error("Failed to download data")
            return
    
    # Step 1: Load Microsoft building footprints
    logger.info("Step 1: Loading Microsoft building footprints")
    
    checkpoint_file = get_checkpoint_path(args.output, args.region, args.samples, args.all_buildings)
    
    if os.path.exists(checkpoint_file) and not args.force_reload:
        logger.info(f"✅ Found existing checkpoint: {checkpoint_file}")
        file_size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)
        logger.info(f"Checkpoint size: {file_size_mb:.1f} MB")
        logger.info("Loading buildings from checkpoint...")
        
        buildings = pd.read_pickle(checkpoint_file)
        logger.info(f"✅ Loaded {len(buildings):,} buildings from checkpoint")
    else:
        if args.force_reload:
            logger.info("Force reload requested. Ignoring checkpoints.")
        else:
            logger.info("No checkpoint found. Loading from original GeoJSONL file...")
        
        if args.all_buildings:
            logger.info("📊 Getting ALL buildings in region - this may take 30-60 minutes...")
        else:
            logger.info(f"This will take 10-15 minutes for {args.samples:,} buildings...")
        
        buildings = load_microsoft_buildings(
            args.data, 
            sample_size=args.samples,
            region_filter=args.region,
            all_buildings=args.all_buildings
        )
        
        if buildings is None or len(buildings) == 0:
            logger.error("No buildings loaded! Check your data file path.")
            logger.info("\nTips:")
            logger.info("  1. Use --download-data to download the Microsoft dataset")
            logger.info("  2. Or specify the correct path with --data")
            logger.info("  3. Or use --region lagos (default) for Lagos-only training")
            return
        
        logger.info(f"Saving buildings checkpoint to {checkpoint_file}...")
        os.makedirs(args.output, exist_ok=True)
        buildings.to_pickle(checkpoint_file)
        logger.info("✅ Checkpoint saved successfully")
    
    # Step 2: Generate addresses
    logger.info("Step 2: Generating realistic Nigerian addresses")
    generator = NigerianAddressGenerator(region=args.region)
    
    training_examples = []
    total_buildings = len(buildings)
    for idx, (_, building) in enumerate(buildings.iterrows()):
        address = generator.generate_address_from_building(building, region_context=True)
        training_examples.append(address)
        
        if (idx + 1) % 10000 == 0:
            percent = (idx + 1) / total_buildings * 100
            logger.info(f"  Generated {idx + 1:,} addresses... ({percent:.1f}%)")
    
    df = pd.DataFrame(training_examples)
    logger.info(f"Generated {len(df):,} address examples")
    
    logger.info("\nSample generated addresses:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        addr_parts = []
        if row['addr:housenumber']:
            addr_parts.append(row['addr:housenumber'])
        if row['addr:street']:
            addr_parts.append(row['addr:street'])
        if row['name']:
            addr_parts.append(row['name'])
        if row['addr:city']:
            addr_parts.append(row['addr:city'])
        logger.info(f"  {i+1}. {' '.join(addr_parts)}")
    
    # Step 3: Convert to BIO
    logger.info("\nStep 3: Converting to BIO format")
    raw_data = [create_bio_data(row) for _, row in df.iterrows()]
    raw_data = [item for item in raw_data if len(item["tokens"]) > 0]
    logger.info(f"  Created {len(raw_data):,} valid training examples")
    
    for item in raw_data:
        item["ner_tags"] = [LABEL2ID[tag] for tag in item["ner_tags"]]

    # Step 4: Create dataset
    logger.info("Step 4: Creating HuggingFace dataset")
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABEL_LIST)),
    })
    
    dataset = Dataset.from_list(raw_data, features=features)
    train_test_split = dataset.train_test_split(test_size=args.test_size, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    logger.info(f"  Training set: {len(train_dataset):,} examples")
    logger.info(f"  Validation set: {len(eval_dataset):,} examples")

    # Step 5: Tokenize
    logger.info("Step 5: Tokenizing with alignment")
    
    if args.offline and os.path.exists(LOCAL_MODEL_PATH):
        model_path = LOCAL_MODEL_PATH
        logger.info(f"  Using local model from: {model_path}")
    else:
        model_path = args.model
        logger.info(f"  Using model from Hugging Face: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    
    logger.info(f"  Tokenizer pad_token: {tokenizer.pad_token}")
    
    from functools import partial
    tokenize_fn = partial(tokenize_and_align, tokenizer=tokenizer, max_length=128)
    
    logger.info("  Tokenizing training set...")
    tokenized_train = train_dataset.map(
        tokenize_fn, 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    logger.info("  Tokenizing validation set...")
    tokenized_eval = eval_dataset.map(
        tokenize_fn, 
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    logger.info("  Verifying tokenized data...")
    sample = tokenized_train[0]
    logger.info(f"  Sample input_ids length: {len(sample['input_ids'])}")
    logger.info(f"  Sample labels length: {len(sample['labels'])}")
    logger.info(f"  All sequences length 128: {len(sample['input_ids']) == 128}")

    # Step 6: Load model
    logger.info(f"Step 6: Loading model")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            num_labels=len(LABEL_LIST),
            ignore_mismatched_sizes=True
        )
        logger.info(f"  Model loaded successfully with {len(LABEL_LIST)} labels")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Step 7: Training arguments
    logger.info("Step 7: Configuring training")
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size= 8 if torch.cuda.is_available() else args.batch_size, # Reduced to max 8, critical for 4g GPU!
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(args.output, "logs"),
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2 if sys.platform != 'win32' else 0,
        gradient_accumulation_steps= 2 if torch.cuda.is_available() else 1,   # Compensates for smaller batch on 4g GPU
    )

    # Step 8: Create trainer with custom collator
    logger.info("Step 8: Creating trainer with custom data collator")
    
    data_collator = CustomDataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 9: Train
    logger.info("Step 9: Starting training")
    logger.info(f"  Training on {len(tokenized_train):,} examples")
    logger.info(f"  Validating on {len(tokenized_eval):,} examples")
    logger.info(f"  Number of epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    # Calculate and display estimated training time
    total_batches = len(tokenized_train) // args.batch_size
    logger.info(f"  Total batches per epoch: {total_batches:,}")
    
    if args.all_buildings:
        logger.info("  ⚠️  Training on ALL buildings will take significantly longer!")
        if args.region == "lagos":
            logger.info("     Estimated time: 24-48 hours for 3 epochs")
        elif args.region == "all":
            logger.info("     Estimated time: 2-3 weeks for 3 epochs")
    
    logger.info("  Testing a batch with custom collator...")
    try:
        sample_batch = next(iter(DataLoader(
            tokenized_train.select(range(args.batch_size)),
            batch_size=args.batch_size,
            collate_fn=data_collator
        )))
        logger.info(f"  ✓ Batch test passed. Shapes: { {k: v.shape for k, v in sample_batch.items()} }")
    except Exception as e:
        logger.error(f"  ✗ Batch test failed: {e}")
        logger.info("  Using fallback collator...")
        
        def manual_collate_fn(batch):
            input_ids = torch.tensor([f["input_ids"] for f in batch], dtype=torch.long)
            attention_mask = torch.tensor([f["attention_mask"] for f in batch], dtype=torch.long)
            labels = torch.tensor([f["labels"] for f in batch], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=manual_collate_fn,
            compute_metrics=compute_metrics,
        )
        logger.info("  ✓ Using manual collate function")
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Attempting to save partial model...")
        trainer.save_model(os.path.join(args.output, "partial_model"))
        return

    # Step 10: Save
    logger.info("Step 10: Saving model and artifacts")
    
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    logger.info("  Saving datasets...")
    train_dataset.to_json(os.path.join(args.output, "train_dataset.jsonl"))
    eval_dataset.to_json(os.path.join(args.output, "eval_dataset.jsonl"))
    
    logger.info("  Saving configuration...")
    with open(os.path.join(args.output, "training_config.txt"), "w") as f:
        f.write(f"Data file: {args.data}\n")
        f.write(f"Samples: {args.samples if not args.all_buildings else 'ALL BUILDINGS'}\n")
        f.write(f"Region: {args.region}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Training examples: {len(train_dataset):,}\n")
        f.write(f"Validation examples: {len(eval_dataset):,}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Test size: {args.test_size}\n")
    
    logger.info(f"✅ Training complete! Model saved to {args.output}")
    
    logger.info("Final evaluation on validation set:")
    try:
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()