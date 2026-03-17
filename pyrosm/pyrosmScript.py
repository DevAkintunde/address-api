from pyrosm import OSM, get_data
import pandas as pd

# 1. Download or locate your Nigeria PBF file
# fp = get_data("Nigeria") # Downloads to temp directory if not present
fp = "./latestOsm/nigeria-260316.osm.pbf" 

# 2. Initialize OSM parser
osm = OSM(fp)

# 3. Extract POIs or Buildings with address tags
# We use a custom filter to ensure we get entities with address information
custom_filter = {"addr:street": True}
nodes_with_addresses = osm.get_pois(custom_filter=custom_filter)

# 4. Clean and select first 1,000 valid addresses
address_columns = ['addr:housenumber', 'addr:street', 'addr:city', 'addr:state']
df = nodes_with_addresses[address_columns].dropna(subset=['addr:street']).head(1000).fillna("")

def create_bio_tags(row):
    tokens = []
    labels = []
    
    mapping = {
        'addr:housenumber': 'HOUSE',
        'addr:street': 'STREET',
        'addr:city': 'CITY',
        'addr:state': 'STATE'
    }
    
    for col, label in mapping.items():
        val = str(row[col]).strip()
        if not val:
            continue
            
        words = val.split()
        for i, word in enumerate(words):
            tokens.append(word)
            if i == 0:
                labels.append(f"B-{label}") # Beginning of entity
            else:
                labels.append(f"I-{label}") # Inside entity
                
    return tokens, labels

# Apply to your data
df['ner_data'] = df.apply(create_bio_tags, axis=1)

# Convert to a format compatible with Hugging Face datasets
formatted_data = []
for _, row in df.iterrows():
    tokens, labels = row['ner_data']
    formatted_data.append({
        "tokens": tokens,
        "ner_tags": labels
    })

# Example output for the first address:
# {'tokens': ['12', 'Admiralty', 'Way', 'Lekki'], 
#  'ner_tags': ['B-HOUSE', 'B-STREET', 'I-STREET', 'B-CITY']}

