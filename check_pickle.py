
import numpy as np

for i, w in enumerate(walk_obj):
    print(f"Index {i}: type={type(w)}, shape={np.shape(w)}")

'''import osmnx as ox
import geopandas as gpd
import pandas as pd
from map_utils import get_NIAs_boundary
import os

# Define your data path
data_root = "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"

# List of NIA IDs to check
nia_ids_to_check = [
    5, 121
]

for nia_id in nia_ids_to_check:
    print("=" * 50)
    print(f"🔍 Checking NIA {nia_id}")

    # 1️⃣ Check if the boundary exists
    boundary = get_NIAs_boundary(nia_id, "ox", data_root)

    if boundary is None or boundary.empty:
        print(f"❌ ERROR: No boundary found for NIA {nia_id}.")
        continue  # Skip to the next NIA if no boundary exists

    print(f"✅ Found boundary for NIA {nia_id}.")
    print(f"🌍 Coordinates: {boundary.geometry.bounds}")

    # 2️⃣ Try querying OpenStreetMap for department stores (example)
    tags= {
     "shop":"confectionery"
    }  
    
    
    
    ''' 
    tag_parking={"amenity":"parking"}✅ALL
    tag_residential={"building":["apartments","bungalow","cabin","detached","dormitory","farm","ger","hotel","house","houseboat","residential","semidetached_house","static_caravan","terrace"]}✅ALL   
    tag_grocery={"shop":"supermarket"}❌22, 28, 115, 121
    tag_school={"amenity":"school"}✅ALL
    tag_department_store={"shop":["mall","department_store"]}❌21, 22, 3, 5, 6, 28, 43, 44, 61, 72, 110, 111, 115, 121, 91, 125, 135, 136, 138, 139, 142
    tag_cafe={"amenity":["cafe", "internet_cafe"]}❌21, 5, 121
    tag_restaurant={"amenity":"restaurant"}✅ALL
    
    orig
    tag_mall={"shop":"mall"}
    tag_coffee={"shop":"coffee"}❌ ALL
    '''

    try:
        features = ox.features.features_from_polygon(boundary.geometry.iloc[0], tags)

        if features.empty:
            print(f"⚠️ WARNING: No OSM data found for {tags} in NIA {nia_id}.")
        else:
            print(f"✅ OSM data found: {len(features)} records.")

    except ox._errors.InsufficientResponseError:
        print(f"❌ ERROR: OSM server returned no data for NIA {nia_id} with tags {tags}.")
        print(f"🔍 Manually check this area: https://www.openstreetmap.org/")
'''
