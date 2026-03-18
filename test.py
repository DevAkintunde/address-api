"""
Quick diagnostic test to see what tags exist in your OSM data
"""

import quackosm as qosm
from shapely.geometry import box
from collections import Counter
import os

def analyze_osm_tags(pbf_path):
    """Analyze what tags are actually in the data"""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING OSM TAGS IN: {pbf_path}")
    print(f"{'='*60}")
    
    # Small area in Lagos (Victoria Island/Ajah area)
    test_bbox = box(3.5, 6.4, 3.6, 6.5)
    
    try:
        print(f"\n📊 Querying small test area: {test_bbox.bounds}")
        print("This will get all features in this small area...")
        
        # No result_limit - get everything in this small area
        gdf = qosm.convert_pbf_to_geodataframe(
            pbf_path,
            tags_filter={},  # No filter - get everything
            geometry_filter=test_bbox,
        )
        
        feature_count = len(gdf)
        print(f"\n✅ Found {feature_count} features in test area")
        
        if feature_count == 0:
            print("❌ No features found! Try a different bounding box.")
            return
        
        # Analyze tags
        tag_counter = Counter()
        addr_counter = Counter()
        building_counter = 0
        amenity_counter = 0
        name_counter = 0
        
        print("\n🔍 Analyzing first 1000 features (or all if less)...")
        
        # Limit analysis to first 1000 to keep it fast
        max_analyze = min(1000, feature_count)
        
        for idx, (_, row) in enumerate(gdf.head(max_analyze).iterrows()):
            tags = row.get('tags', {})
            
            # Count all tags
            for key in tags:
                tag_counter[key] += 1
                if key.startswith('addr:'):
                    addr_counter[key] += 1
            
            # Count specific feature types
            if 'building' in tags:
                building_counter += 1
            if 'amenity' in tags:
                amenity_counter += 1
            if 'name' in tags:
                name_counter += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} features...")
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        print(f"\n📊 FEATURE TYPES (in sample of {max_analyze} features):")
        print(f"  Buildings: {building_counter}")
        print(f"  Amenities: {amenity_counter}")
        print(f"  Named features: {name_counter}")
        
        print("\n🏷️ TOP 20 MOST COMMON TAGS:")
        for tag, count in tag_counter.most_common(20):
            print(f"  {tag:25}: {count:5} occurrences")
        
        if addr_counter:
            print("\n📍 ADDRESS-RELATED TAGS FOUND:")
            for tag, count in addr_counter.most_common():
                print(f"  {tag:25}: {count:5} occurrences")
            
            # Show sample of address data
            print("\n📝 SAMPLE ADDRESS DATA:")
            samples_shown = 0
            for _, row in gdf.head(20).iterrows():
                tags = row.get('tags', {})
                addr_tags = {k: v for k, v in tags.items() if k.startswith('addr:')}
                if addr_tags:
                    print(f"  {addr_tags}")
                    samples_shown += 1
                    if samples_shown >= 5:
                        break
        else:
            print("\n❌ NO ADDRESS TAGS FOUND in sample!")
            print("\n💡 This explains why you're getting 0 records.")
            print("The OSM data for this area doesn't have structured address tags.")
            
            # Show what IS available
            print("\n📋 Here are some actual tags from the data:")
            for _, row in gdf.head(5).iterrows():
                tags = row.get('tags', {})
                # Show first 5 tags
                sample = dict(list(tags.items())[:5])
                print(f"  {sample}")
        
        # Check for alternative address patterns
        print("\n🔎 CHECKING FOR ALTERNATIVE ADDRESS PATTERNS:")
        
        # Look for 'name' + 'place' combinations that might indicate addresses
        place_features = 0
        for _, row in gdf.head(max_analyze).iterrows():
            tags = row.get('tags', {})
            if 'name' in tags and any(p in tags for p in ['place', 'locality', 'village']):
                place_features += 1
        
        print(f"  Named places: {place_features}")
        
        if place_features > 0:
            print("\n  These could be used as landmarks or locality names")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def check_specific_tags(pbf_path):
    """Check for specific tag combinations"""
    
    test_bbox = box(3.5, 6.4, 3.6, 6.5)
    
    print("\n" + "="*60)
    print("CHECKING SPECIFIC TAG COMBINATIONS")
    print("="*60)
    
    tag_combinations = [
        ("Buildings with names", {"building": True, "name": True}),
        ("Named amenities", {"amenity": True, "name": True}),
        ("Any address tag", {"addr:*": True}),
        ("Street names", {"addr:street": True}),
        ("House numbers", {"addr:housenumber": True}),
        ("Full addresses", {"addr:full": True}),
        ("Named places", {"place": True, "name": True}),
    ]
    
    for description, tags in tag_combinations:
        try:
            gdf = qosm.convert_pbf_to_geodataframe(
                pbf_path,
                tags_filter=tags,
                geometry_filter=test_bbox,
            )
            count = len(gdf)
            print(f"  {description:25}: {count:6} features")
            
            if count > 0 and description == "Any address tag":
                print("\n  Sample address tags:")
                for _, row in gdf.head(3).iterrows():
                    tags = row.get('tags', {})
                    addr = {k: v for k, v in tags.items() if k.startswith('addr:')}
                    print(f"    {addr}")
        except Exception as e:
            print(f"  {description:25}: ERROR - {str(e)[:50]}")

if __name__ == "__main__":
    # Required for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    pbf_path = "./latestOsm/nigeria-260316.osm.pbf"
    
    if not os.path.exists(pbf_path):
        print(f"❌ File not found: {pbf_path}")
    else:
        analyze_osm_tags(pbf_path)
        check_specific_tags(pbf_path)