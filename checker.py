import os
import pandas as pd
import numpy as np

# 🔹 Modify these paths accordingly
results_folder = "results"
filename = "NIA_155_0,0,0.csv"
csv_path = os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", filename)

# 🔹 Check if file exists
if not os.path.exists(csv_path):
    print(f"❌ File not found: {csv_path}")
else:
    print(f"✅ File found: {csv_path}")

    # 🔹 Load the CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove spaces in column names

    # 🔹 Check if "obj" column exists
    if "obj" in df.columns:
        walk_obj = df["obj"].dropna().values  # Convert to NumPy array

        print("\n📌 Checking 'walk_obj' structure:")
        print(f"Type: {type(walk_obj)}")
        print(f"Shape: {walk_obj.shape}")
        print(f"First 5 elements: {walk_obj[:5]}")

        # 🔹 Check for sequences inside the array
        problem_elements = [i for i, item in enumerate(walk_obj) if not np.isscalar(item)]
        
        if problem_elements:
            print(f"\n❌ Issue Found! Non-scalar values detected at indices: {problem_elements}")
            print("Here are the problematic values:")
            print([walk_obj[i] for i in problem_elements])
        else:
            print("\n✅ 'walk_obj' looks fine! No sequences detected.")

    else:
        print("\n❌ Warning: 'obj' column is missing!")
