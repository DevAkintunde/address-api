"""
Nigeria administrative regions and major city bounding boxes
Based on authoritative geographic boundaries
"""

from shapely.geometry import box
from typing import Dict, List, Optional, Union

# Nigeria's overall boundaries
NIGERIA_BOUNDS = {
    "west": 2.5,
    "east": 14.5,
    "south": 4.0, 
    "north": 14.0
}

# Geopolitical zones
GEOPOLITICAL_ZONES = {
    "north_central": {
        "name": "North Central",
        "states": ["FCT", "Niger", "Kwara", "Kogi", "Benue", "Plateau"],
        "bbox": box(4.0, 7.0, 10.0, 11.0)
    },
    "north_east": {
        "name": "North East",
        "states": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
        "bbox": box(9.0, 9.0, 14.5, 14.0)
    },
    "north_west": {
        "name": "North West",
        "states": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
        "bbox": box(3.0, 11.0, 9.0, 14.0)
    },
    "south_east": {
        "name": "South East",
        "states": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
        "bbox": box(6.5, 5.0, 8.5, 7.0)
    },
    "south_south": {
        "name": "South South",
        "states": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
        "bbox": box(4.0, 4.0, 9.0, 7.0)
    },
    "south_west": {
        "name": "South West",
        "states": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
        "bbox": box(2.5, 6.0, 6.0, 9.0)
    }
}

# Individual states
STATES = {
    "abia": {"name": "Abia", "bbox": box(7.0, 5.0, 8.0, 6.0)},
    "abuja": {"name": "Abuja (FCT)", "bbox": box(6.5, 8.5, 7.8, 9.5)},
    "adamawa": {"name": "Adamawa", "bbox": box(11.5, 8.5, 13.5, 10.5)},
    "akwa_ibom": {"name": "Akwa Ibom", "bbox": box(7.5, 4.5, 8.5, 5.5)},
    "anambra": {"name": "Anambra", "bbox": box(6.5, 6.0, 7.5, 7.0)},
    "bauchi": {"name": "Bauchi", "bbox": box(8.5, 10.0, 11.0, 12.0)},
    "bayelsa": {"name": "Bayelsa", "bbox": box(5.5, 4.0, 6.5, 5.5)},
    "benue": {"name": "Benue", "bbox": box(7.5, 6.5, 10.0, 8.5)},
    "borno": {"name": "Borno", "bbox": box(11.5, 11.0, 14.5, 14.0)},
    "cross_river": {"name": "Cross River", "bbox": box(7.5, 5.0, 9.5, 7.0)},
    "delta": {"name": "Delta", "bbox": box(5.0, 5.0, 6.5, 6.5)},
    "ebonyi": {"name": "Ebonyi", "bbox": box(7.5, 5.5, 8.5, 7.0)},
    "edo": {"name": "Edo", "bbox": box(5.0, 6.0, 6.5, 7.5)},
    "ekiti": {"name": "Ekiti", "bbox": box(4.5, 7.0, 6.0, 8.5)},
    "enugu": {"name": "Enugu", "bbox": box(7.0, 6.0, 8.0, 7.5)},
    "gombe": {"name": "Gombe", "bbox": box(10.5, 10.0, 12.0, 11.5)},
    "imo": {"name": "Imo", "bbox": box(6.5, 5.0, 7.5, 6.0)},
    "jigawa": {"name": "Jigawa", "bbox": box(8.0, 11.5, 10.5, 13.0)},
    "kaduna": {"name": "Kaduna", "bbox": box(6.5, 9.5, 9.0, 11.5)},
    "kano": {"name": "Kano", "bbox": box(7.5, 11.5, 9.5, 13.0)},
    "katsina": {"name": "Katsina", "bbox": box(6.5, 12.0, 9.0, 13.5)},
    "kebbi": {"name": "Kebbi", "bbox": box(3.0, 11.0, 6.0, 13.0)},
    "kogi": {"name": "Kogi", "bbox": box(5.5, 7.0, 7.5, 8.5)},
    "kwara": {"name": "Kwara", "bbox": box(2.5, 8.0, 6.0, 10.0)},
    "lagos": {"name": "Lagos", "bbox": box(2.5, 6.0, 4.5, 7.0)},
    "nasarawa": {"name": "Nasarawa", "bbox": box(7.0, 8.0, 9.5, 9.5)},
    "niger": {"name": "Niger", "bbox": box(4.0, 8.5, 7.5, 11.0)},
    "ogun": {"name": "Ogun", "bbox": box(2.5, 6.5, 5.0, 8.0)},
    "ondo": {"name": "Ondo", "bbox": box(4.0, 5.5, 6.0, 8.0)},
    "osun": {"name": "Osun", "bbox": box(4.0, 7.0, 5.0, 8.5)},
    "oyo": {"name": "Oyo", "bbox": box(2.5, 7.0, 5.0, 9.0)},
    "plateau": {"name": "Plateau", "bbox": box(8.0, 8.5, 10.0, 10.5)},
    "rivers": {"name": "Rivers", "bbox": box(6.5, 4.5, 7.5, 5.5)},
    "sokoto": {"name": "Sokoto", "bbox": box(4.0, 12.0, 7.0, 13.5)},
    "taraba": {"name": "Taraba", "bbox": box(9.5, 7.0, 11.5, 9.0)},
    "yobe": {"name": "Yobe", "bbox": box(10.5, 11.0, 12.5, 13.5)},
    "zamfara": {"name": "Zamfara", "bbox": box(5.0, 11.0, 7.5, 13.0)},
}

# Major cities (more granular than states)
MAJOR_CITIES = {
    "lagos_metro": {"name": "Lagos Metro", "bbox": box(2.5, 6.0, 4.0, 7.0)},
    "ibadan": {"name": "Ibadan", "bbox": box(3.5, 7.0, 4.5, 8.0)},
    "kano_city": {"name": "Kano City", "bbox": box(8.0, 11.5, 9.0, 12.5)},
    "abuja_city": {"name": "Abuja City", "bbox": box(7.0, 8.5, 8.0, 9.5)},
    "port_harcourt": {"name": "Port Harcourt", "bbox": box(6.5, 4.5, 7.5, 5.5)},
    "benin_city": {"name": "Benin City", "bbox": box(5.0, 6.0, 6.0, 7.0)},
    "kaduna_city": {"name": "Kaduna City", "bbox": box(7.0, 10.0, 8.0, 11.0)},
    "jos": {"name": "Jos", "bbox": box(8.5, 9.5, 9.5, 10.5)},
    "ilorin": {"name": "Ilorin", "bbox": box(4.0, 8.0, 5.0, 9.0)},
    "maiduguri": {"name": "Maiduguri", "bbox": box(13.0, 11.5, 14.0, 12.5)},
    "enugu_city": {"name": "Enugu", "bbox": box(7.0, 6.0, 8.0, 7.0)},
    "warri": {"name": "Warri", "bbox": box(5.5, 5.0, 6.0, 6.0)},
    "abeokuta": {"name": "Abeokuta", "bbox": box(3.0, 6.5, 3.5, 7.5)},
}

# Predefined region groups - FIXED: removed the undefined 'i' variable
REGION_GROUPS = {
    "all": [{"name": "nigeria", "bbox": box(2.5, 4.0, 14.5, 14.0)}],  # Fixed: removed {i}
    "southwest_states": [
        STATES["lagos"], STATES["ogun"], STATES["oyo"], 
        STATES["osun"], STATES["ondo"], STATES["ekiti"]
    ],
    "major_cities": list(MAJOR_CITIES.values()),
    "southern_nigeria": [
        GEOPOLITICAL_ZONES["south_west"],
        GEOPOLITICAL_ZONES["south_east"], 
        GEOPOLITICAL_ZONES["south_south"]
    ],
    "northern_nigeria": [
        GEOPOLITICAL_ZONES["north_central"],
        GEOPOLITICAL_ZONES["north_east"],
        GEOPOLITICAL_ZONES["north_west"]
    ],
}

def get_regions(region_type: str, specific_regions: Optional[List[str]] = None) -> List[Dict]:
    """
    Get region definitions based on parameters
    
    Args:
        region_type: One of ["all", "zones", "states", "cities", "group", "custom"]
        specific_regions: List of specific region keys to include
    
    Returns:
        List of region dictionaries with 'name' and 'bbox' keys
    """
    if region_type == "all":
        return [{"name": "nigeria", "bbox": box(2.5, 4.0, 14.5, 14.0)}]
    
    elif region_type == "zones":
        return [{"name": zone["name"], "bbox": zone["bbox"]} 
                for zone in GEOPOLITICAL_ZONES.values()]
    
    elif region_type == "states":
        return [{"name": state["name"], "bbox": state["bbox"]} 
                for state in STATES.values()]
    
    elif region_type == "cities":
        return [{"name": city["name"], "bbox": city["bbox"]} 
                for city in MAJOR_CITIES.values()]
    
    elif region_type == "group":
        if not specific_regions or len(specific_regions) == 0:
            raise ValueError("Must specify at least one group name")
        
        regions = []
        for group_name in specific_regions:
            if group_name in REGION_GROUPS:
                regions.extend(REGION_GROUPS[group_name])
            else:
                raise ValueError(f"Unknown group: {group_name}")
        return regions
    
    elif region_type == "custom":
        if not specific_regions:
            raise ValueError("Must specify custom region keys")
        
        regions = []
        source_maps = [
            ("zone", GEOPOLITICAL_ZONES),
            ("state", STATES),
            ("city", MAJOR_CITIES)
        ]
        
        for key in specific_regions:
            found = False
            for source_name, source_map in source_maps:
                if key in source_map:
                    regions.append({"name": source_map[key]["name"], "bbox": source_map[key]["bbox"]})
                    found = True
                    break
            if not found:
                raise ValueError(f"Unknown region key: {key}")
        return regions
    
    else:
        raise ValueError(f"Unknown region type: {region_type}")

# Convenience functions
def get_southwest_nigeria():
    """Get all southwestern states (Lagos, Ogun, Oyo, Osun, Ondo, Ekiti)"""
    return get_regions("custom", ["lagos", "ogun", "oyo", "osun", "ondo", "ekiti"])

def get_major_cities():
    """Get all major Nigerian cities"""
    return get_regions("cities")

def get_geopolitical_zones():
    """Get all 6 geopolitical zones"""
    return get_regions("zones")

def get_nigeria_bbox():
    """Get the overall bounding box for Nigeria"""
    return box(2.5, 4.0, 14.5, 14.0)

if __name__ == "__main__":
    # Test the module
    print("Testing Nigeria Regions Module")
    print("=" * 50)
    
    print("\nSouthwest Nigeria:")
    for region in get_southwest_nigeria():
        print(f"  - {region['name']}")
    
    print("\nMajor Cities:")
    for region in get_major_cities()[:5]:  # First 5
        print(f"  - {region['name']}")
    
    print("\nGeopolitical Zones:")
    for region in get_geopolitical_zones():
        print(f"  - {region['name']}")