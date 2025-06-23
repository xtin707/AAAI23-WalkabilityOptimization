import osmnx as ox
import geopandas as gpd
import pandas as pd
from map_utils import get_NIAs_boundary
import os

# Define your data path
data_root = "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"

# List of NIA IDs to check
nia_ids_to_check = [
2, 3, 6, 21, 22, 24, 25, 154, 155, 27, 28, 43, 44, 55, 61, 72, 85, 91, 110, 
111, 112, 113, 115, 121, 124, 125, 135, 136, 141, 142, 138, 139 ]

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

    # 2️⃣ Try querying OpenStreetMap for: department stores (input tag here)
    tags= {"shop":["mall","department_store", "supermarket"]
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



'''import pandas as pd
import os

# Set the folder paths
base_folder = "OptMultipleDepth_False_0"
sol_folder = os.path.join(base_folder, "sol")
summary_folder = os.path.join(base_folder, "summary")
os.makedirs(summary_folder, exist_ok=True)

# Loop through all NIA files that already have .csv/.sol
for sol_file in os.listdir(sol_folder):
    if sol_file.endswith(".sol"):
        nia_id = sol_file.split("_")[1]  # or whatever naming convention you used
        # Load other necessary files like .txt or .csv to extract needed values
        # Example: if you saved a detailed log CSV:
        detailed_csv = os.path.join(sol_folder, sol_file.replace(".sol", ".csv"))
        if os.path.exists(detailed_csv):
            df = pd.read_csv(detailed_csv)
            # Extract necessary info and build results_D manually
            # This depends on your column structure
            results_D = {
                "nia_id": [nia_id],
                "nia_name": [df["nia_name"].iloc[0]],
                "k_L_grocery": [df["k_grocery"].iloc[0]],
                "k_L_restaurant": [df["k_restaurant"].iloc[0]],
                "k_L_school": [df["k_school"].iloc[0]],
                "obj": [df["obj"].iloc[0]],
                "dist_obj_L_grocery": [df["dist_grocery"].iloc[0]],
                "dist_obj_L_restaurant": [df["dist_restaurant"].iloc[0]],
                "dist_obj_L_school": [df["dist_school"].iloc[0]],
                "solving_time": [df["time"].iloc[0]],
                "num_res": [df["residents"].iloc[0]],
                "num_parking": [df["allocations"].iloc[0]],
                "num_existing_L_grocery": [df["existing_grocery"].iloc[0]],
                "num_existing_L_restaurant": [df["existing_restaurant"].iloc[0]],
                "num_existing_L_school": [df["existing_school"].iloc[0]],
                "model_status": [df["status"].iloc[0]]
            }

            # Save to summary CSV
            summary_df = pd.DataFrame(results_D)
            summary_filename = os.path.join(summary_folder, f"NIA_{nia_id}_summary.csv")
            summary_df.to_csv(summary_filename, index=False)'''
