from graph_utils import *
from map_utils import *
import model_latest
#from CP_models import *
#from MIP_models import *
from model_latest import opt_single, cur_assignment_single, opt_multiple, opt_single_depth, cur_assignment_single_depth, weights_array, dist_to_score, L_a, L_f_a, opt_multiple_depth, weights_array_multi, choice_weights, opt_single_CP, opt_multiple_CP,opt_single_depth_CP, opt_multiple_depth_CP
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import numpy as np
import pickle
from greedy import greedy_multiple_depth, greedy_multiple, greedy_single, greedy_single_depth, get_nearest

data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
D_NIA = ct_nia_mapping(os.path.join(data_root,"Neighbourhood Improvement Areas - 4326/processed_TSNS 2020 NIA Census Tracts.xlsx"))

models_folder = "models"
results_folder = "results"
plot_folder = "results_plot"
data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
processed_folder= "processed_results"
preprocessing_folder = "./preprocessing"

model_name = "OptMultipleDepth_False_0"
sol_folder = os.path.join(results_folder,os.path.join("sol",model_name))

net_save_path = os.path.join(preprocessing_folder, 'saved_nets')
df_save_path = os.path.join(preprocessing_folder, 'saved_dfs')
sp_save_path = os.path.join(preprocessing_folder, 'saved_SPs')

nia= 43
k_name = 2

myhatch = 'o'
myscale = 4


if __name__ == "__main__":


    pednet = load_pednet(data_root)
    nia_id_L = []
    nia_name_L = []
    obj_L = []
    solving_time_L = []
    num_residents_L = []
    num_allocations_L = []
    status_L = []



    pednet_nia = pednet_NIA(pednet, nia, preprocessing_folder)


    # # load net
    prec = 2

    # load dfs
    all_strs = ['residential', 'department_store', 'parking', 'grocery', 'school', 'cafe', 'restaurant']
    colors = ['g', 'lightcoral', 'grey', 'red', 'yellow', 'brown', 'orange']
    df_filenames = ["NIA_%s_%s.pkl" % (nia, str) for str in all_strs]

    all_dfs = []
    for df_filename in df_filenames:
        file_path = os.path.join(df_save_path, df_filename)
        if os.path.exists(file_path):
            try:
                all_dfs.append(pd.read_pickle(file_path))
            except Exception as e:
                print(f"Error loading {df_filename}: {e}")
        else:
            print(f"Skipping missing file: {df_filename}")

    # Ensure all_dfs has the correct number of elements, replacing missing ones with empty DataFrames
    while len(all_dfs) < len(all_strs):
        all_dfs.append(pd.DataFrame())  # Add an empty DataFrame for missing data

    residentials_df, department_store_df, parking_df, grocery_df, school_df, cafe_df, restaurant_df = all_dfs

    # load SP
    SP_filename = "NIA_%s_prec_%s.txt" % (nia, prec)
    D = np.loadtxt(os.path.join(sp_save_path, SP_filename))

    allocated_f_name = os.path.join(sol_folder, "allocation_NIA_%s_%s,%s,%s.pkl" % (nia, k_name, k_name, k_name))

    if os.path.exists(allocated_f_name):
        with open(allocated_f_name, 'rb') as f:
            sol = pickle.load(f)
    else:
        print(f"Skipping missing allocation file: {allocated_f_name}")
        sol = None  # Set sol to None or an empty dictionary if you need to reference it later


    ax = pednet_nia.plot(figsize=(15, 15), color='blue', markersize=1)
    
    if residentials_df is not None and not residentials_df.empty:
        res = residentials_df.plot(ax=ax, color='green', markersize=80, label='Residential location')
    else:
        print("Skipping plot: No residential data available for this NIA.")

    if parking_df is not None and not parking_df.empty:
        parking = parking_df.plot(ax=ax, color='gray', markersize=80, label='Parking lot')
    else:
        print("Skipping plot: No parking data available for this NIA.")
    #,fontsize=20
    
    if grocery_df is not None and not parking_df.empty:
        grocery = grocery_df.plot(ax=ax,color='red', markersize=80, label='Existing grocery')
    else:
        print("Skipping plot: No grocery data available for this NIA.")
    
    if restaurant_df is not None and not parking_df.empty:
        res = restaurant_df.plot(ax=ax, color='orange', markersize=80, label='Existing restaurant')
    else:
        print("Skipping plot: No restaurant data available for this NIA.")
    
    if school_df is not None and not parking_df.empty:
        school = school_df.plot(ax=ax, color='yellow', markersize=80, label='Existing school')
    else:
        print("Skipping plot: No school data available for this NIA.")
    
    all_dfs.append(pd.read_pickle(file_path) if os.path.exists(file_path) else pd.DataFrame())


    if sol is not None:
        allocated_grocery = parking_df.iloc[sol["allocate_row_id_grocery"]]
        #ALTERED move the ff block of codes in if-sttmnt to handle nonetype error
        allocated_grocery2=allocated_grocery.copy()
        allocated_grocery2["geometry"] = allocated_grocery["geometry"].scale(myscale,myscale, origin='center')
        allocated_grocery2.plot(ax=ax, edgecolor='black',facecolor='red', hatch=myhatch,markersize=80, label='Allocated grocery', zorder=100)
    else:
        print("Skipping allocation: No allocation data available for this NIA.")
        allocated_grocery = None  # Set to None or an empty DataFrame if needed

    

    allocated_res = parking_df.iloc[sol["allocate_row_id_restaurant"]]
    allocated_res2 = allocated_res.copy()
    allocated_res2["geometry"] = allocated_res2["geometry"].scale(myscale, myscale, origin='center')
    allocated_res2.plot(ax=ax,  edgecolor='black',facecolor='orange', hatch=myhatch, markersize=80, label='Allocated restaurant', zorder=101)

    allocated_school = parking_df.iloc[sol["allocate_row_id_school"]]
    allocated_school2 = allocated_school.copy()
    allocated_school2["geometry"] = allocated_school2["geometry"].scale(myscale, myscale, origin='center')
    allocated_school2.plot(ax=ax,  edgecolor='black',facecolor='yellow', hatch=myhatch, markersize=80, label='Allocated school', zorder=102)

    green_patch = mpatches.Patch(color='green', label='Residential location')
    gray_patch = mpatches.Patch(color='gray', label='Parking lot')
    red_patch = mpatches.Patch(color='red', label='Existing grocery')
    orange_patch = mpatches.Patch(color='orange', label='Existing restaurant')
    yellow_patch = mpatches.Patch(color='yellow', label='Existing school')

    red_patch2 = mpatches.Patch(facecolor='red',  edgecolor='black',hatch=myhatch, label='Allocated grocery')
    orange_patch2 = mpatches.Patch(facecolor='orange', edgecolor='black', hatch=myhatch, label='Allocated restaurant')
    yellow_patch2 = mpatches.Patch(facecolor='yellow',  edgecolor='black',hatch=myhatch, label='Allocated school')

    #last_patch = mpatches.Patch(color=colors[all_strs.index(args.amenity)], label='Existing amenity')
    plt.rcParams['hatch.linewidth'] = 2.0

    #plt.legend(handles=[gray_patch,green_patch,red_patch,orange_patch,yellow_patch,red_patch2,orange_patch2,yellow_patch2])
    plt.legend(
        handles=[gray_patch, green_patch, red_patch, orange_patch, yellow_patch, red_patch2, orange_patch2, yellow_patch2],
        loc='lower left',  # Change legend position
        fontsize=20,       # Adjust font size
        frameon=True,      # Optional: adds a border box
        ncol=1             # Optional: controls number of columns
    )


    plt.title("neighbourhood: %s" % (D_NIA[nia]['name']))

    fig_name = "nia_%s_%s_allocation.png" % (nia, k_name)


    plt.savefig(os.path.join(plot_folder,fig_name))




