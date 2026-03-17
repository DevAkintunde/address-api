import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def generate_ref_files(df, output_dir="."):
    """
    Extracts unique street and landmark names from the dataframe
    and saves them to text files for fuzzy matching.
    """
    # 1. Process Streets
    if 'addr:street' in df.columns:
        # Get unique, non-null values and sort them
        streets = df['addr:street'].dropna().unique().tolist()
        # Clean: remove empty strings and 'nan' text
        streets = [str(s).strip() for s in streets if str(s).strip() and str(s).lower() != 'nan']
        
        with open(os.path.join(output_dir, "ref_streets.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(streets)))
        logger.info(f"Generated ref_streets.txt with {len(streets)} unique names.")

    # 2. Process Landmarks
    if 'name' in df.columns:
        landmarks = df['name'].dropna().unique().tolist()
        landmarks = [str(l).strip() for l in landmarks if str(l).strip() and str(l).lower() != 'nan']
        
        with open(os.path.join(output_dir, "ref_landmarks.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(landmarks)))
        logger.info(f"Generated ref_landmarks.txt with {len(landmarks)} unique names.")

# To integrate into your training script:
# generate_ref_files(master_df)
