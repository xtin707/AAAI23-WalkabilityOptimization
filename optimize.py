# Edited optimize.py - single-amenity & CP models removed
from graph_utils import *
from map_utils import *
from model_latest import * 
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import numpy as np
import pickle
from greedy import *
import sys


parser = argparse.ArgumentParser(description='Enter model name:grb_PWL,scratch')
parser.add_argument("model", help="model", type=str)
parser.add_argument("nias", help="nias to run", type=str)
parser.add_argument("--k_array", help="upper bound", type=str)
parser.add_argument("--bp", help="whether to set branching priority", default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--focus", help="MIPFocus parameter", default=0,type=int)

args, unknown = parser.parse_known_args()

print("Received arguments:", sys.argv)
if unknown:
    print("Ignoring unknown arguments:", unknown)

data_root = r"C:\Users\annve\Downloads\Walkability For All"
preprocessing_folder = "./preprocessing"
#Thread count (18) is larger than processor count (12) Reduce the value of the Threads parameter to improve performance
threads = 12 


file_path = os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx")
df = pd.read_excel(file_path)
D_NIA = ct_nia_mapping(file_path)

models_folder = "models"
results_folder = "results"
Path(models_folder).mkdir(parents=True,exist_ok=True)
Path(results_folder).mkdir(parents=True,exist_ok=True)

net_save_path = os.path.join(preprocessing_folder, 'saved_nets')
df_save_path = os.path.join(os.getcwd(), "preprocessing", "saved_dfs")
sp_save_path = os.path.join(preprocessing_folder, 'saved_SPs')

model_save_name = args.model + "_" + str(args.bp) + "_" + str(args.focus)

visual_folder = os.path.join(results_folder,os.path.join("visualization",model_save_name))
sol_folder = os.path.join(results_folder,os.path.join("sol",model_save_name))
summary_folder = os.path.join(results_folder,os.path.join("summary",model_save_name))

Path(visual_folder).mkdir(parents=True, exist_ok=True)
Path(sol_folder).mkdir(parents=True, exist_ok=True)
Path(summary_folder).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    nia_list = [int(x) for x in args.nias.split(',') if x.strip().isdigit()]

    pednet = load_pednet(data_root)
    nia_id_L = []
    nia_name_L = []
    obj_L = []
    solving_time_L = []
    num_residents_L = []
    num_allocations_L = []
    status_L = []

    # For the simplified script we only keep multiple-amenity result collectors
    num_existing_L_grocery, num_existing_L_restaurant, num_existing_L_school, num_existing_L_healthcare = [], [], [],  []
    dist_obj_L_grocery, dist_obj_L_restaurant, dist_obj_L_school, dist_obj_L_healthcare = [], [], [], []
    k_L_grocery, k_L_restaurant, k_L_school, k_L_healthcare = [], [], [], []

    for nia_id in nia_list:
        pednet_nia = pednet_NIA(pednet, nia_id, preprocessing_folder)
        print("NIA ",nia_id)

        # load network
        prec = 2
        net_filename = "NIA_%s_prec_%s.hd5" % (nia_id, prec)
        if os.path.exists(os.path.join(net_save_path, net_filename)):
            transit_ped_net = pdna.Network.from_hdf5(os.path.join(net_save_path, net_filename))
        else:
            G = create_graph(pednet_nia, precision=prec)
            transit_ped_net = get_pandana_net(G, os.path.join(net_save_path, net_filename))

        # load dfs
        all_strs = ['residential', 'department_store', 'parking', 'grocery', 'school', 'cafe', 'restaurant','healthcare']
        colors = ['g', 'lightcoral', 'grey', 'red', 'yellow', 'brown', 'orange', 'blue']
        df_filenames = ["NIA_%s_%s.pkl" % (nia_id, s) for s in all_strs]
        all_dfs = [pd.read_pickle(os.path.join(df_save_path, df_filename)) for df_filename in df_filenames]
        residentials_df, department_store_df, parking_df, grocery_df, school_df, cafe_df, restaurant_df, healthcare_df = all_dfs

        # load SP matrix
        SP_filename = "NIA_%s_prec_%s.txt" % (nia_id, prec)
        D = np.loadtxt(os.path.join(sp_save_path, SP_filename))

        # Handles multiple-amenity model
        if args.model in ['OptMultiple','OptMultipleDepth','GreedyMultiple','GreedyMultipleDepth']:
            if args.k_array != '0,0,0' and args.k_array is not None:
                k_array = [int(x) for x in args.k_array.split(',')]
                log_file_name = os.path.join(sol_folder, "log_NIA_%s_%s.txt" % (nia_id, args.k_array))
                if not 'Greedy' in args.model:
                    # call MILP multiple-amenity solver (opt_multiple / opt_multiple_depth)
                    if args.model == 'OptMultiple':
                        score_obj, [dist_grocery, dist_restaurant, dist_school, dist_healthcare], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school, num_cur_healthcare], status\
                            = opt_multiple(residentials_df, parking_df, grocery_df, restaurant_df, school_df, healthcare_df, D, k_array,threads, log_file_name,args.bp, args.focus, EPS = 0.5)
                    else:
                        score_obj, [dist_grocery, dist_restaurant, dist_school, dist_healthcare], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school, num_cur_healthcare], status\
                            = opt_multiple_depth(residentials_df, parking_df, grocery_df, restaurant_df, school_df, healthcare_df, D, k_array,threads, log_file_name,args.bp, args.focus, EPS = 0.5)
                    
                # Greedy variants for multiple
                else:
                    if args.model == 'GreedyMultiple':
                        score_obj, [dist_grocery, dist_restaurant, dist_school, dist_healthcare], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school, num_cur_healthcare], status \
                            = greedy_multiple(residentials_df, parking_df, grocery_df, restaurant_df, school_df, healthcare_df, D, k_array)
                    else:
                        score_obj, [dist_grocery, dist_restaurant, dist_school, dist_healthcare], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school, num_cur_healthcare], status \
                            = greedy_multiple_depth(residentials_df, parking_df, grocery_df, restaurant_df, school_df, healthcare_df, D, k_array)
            else:    
                # k_array == '0,0,0' compute current assignment distances per amenity and compute weighted score
                multiple_dist = []
                # grocery
                score_obj, dist_grocery, solving_time, m, assigned_D, num_residents, num_cur_grocery, status = cur_assignment_single(residentials_df, grocery_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)
                    
                # restaurant (depth case)
                score_obj, dist_restaurant, solving_time, m, assigned_D, num_residents, num_cur_restaurant, status = cur_assignment_single_depth(residentials_df, restaurant_df, D, args.bp, args.focus, EPS=0.5)
                tot_choices = min(num_cur_restaurant, len(choice_weights))
                for c in range(tot_choices):
                    multiple_dist.append(assigned_D[str(c) + "_dist"])
                for choice in range(tot_choices, len(choice_weights)):
                    multiple_dist.append([L_a[-2]] * num_residents)
                    
                # school
                score_obj, dist_school, solving_time, m, assigned_D, num_residents, num_cur_school, status = cur_assignment_single(residentials_df, school_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)
                    
                # healthcare
                score_obj, dist_healthcare, solving_time, m, assigned_D, num_residents, num_cur_healthcare, status = cur_assignment_single(residentials_df, healthcare_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])    
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)

                multiple_dist = np.array(multiple_dist)
                # choose proper weights depending on having depth or not
                try:
                    weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                except Exception:
                    weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)

                solving_time=None
                status=None
                assigned_D = get_nearest(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D)

        else:
            print("choose a model name - allowed: OptMultiple, OptMultipleDepth, GreedyMultiple, GreedyMultipleLazy, GreedyMultipleDepth")

        # saving & plotting 
        if args.model in ['OptMultiple','OptMultipleDepth','GreedyMultipleDepth','GreedyMultiple','GreedyMultipleLazy']:
            if args.k_array != '0,0,0' and args.k_array is not None:
                k_name = args.k_array
                allocated_f_name = os.path.join(sol_folder, "allocation_NIA_%s_%s.pkl" % (nia_id, k_name))
                with open(allocated_f_name, 'wb') as f:
                    pickle.dump(allocated_D, f)
            else:
                k_name = '0,0,0'

            assigned_f_name = os.path.join(sol_folder, "assignment_NIA_%s_%s.csv" % (nia_id, k_name))
            model_f_name = os.path.join(sol_folder, "NIA_%s_%s.sol" % (nia_id, k_name))

        if assigned_D:
            pd.DataFrame.from_dict(assigned_D).to_csv(assigned_f_name)
        if m is not None:
            try:
                m.write(model_f_name)
                print(f"Model saved to {model_f_name}")
            except Exception as e:
                print(f"Failed to save model: {e}")

        # Check if the key exists before accessing it
        if nia_id in D_NIA:
            print("Key exists! Proceeding with access.")
            nia_name_L.append(D_NIA[nia_id]['name'])
        else:
            # Handle missing key case
            print(f"Key {nia_id} not found in D_NIA!")
            nia_name_L.append("Unknown")

        # save summary
        nia_id_L.append(nia_id)

        num_existing_L_grocery.append(num_cur_grocery if 'num_cur_grocery' in locals() else None)
        num_existing_L_restaurant.append(num_cur_restaurant if 'num_cur_restaurant' in locals() else None)
        num_existing_L_school.append(num_cur_school if 'num_cur_school' in locals() else None)
        num_existing_L_healthcare.append(num_cur_healthcare if 'num_cur_healthcare' in locals() else None)
        
        dist_obj_L_grocery.append(dist_grocery if 'dist_grocery' in locals() else None)
        dist_obj_L_restaurant.append(dist_restaurant if 'dist_restaurant' in locals() else None)
        dist_obj_L_school.append(dist_school if 'dist_school' in locals() else None)
        dist_obj_L_healthcare.append(dist_healthcare if 'dist_healthcare' in locals() else None)

        if args.k_array != '0,0,0' and args.k_array is not None:
            k_L_grocery.append(k_array[0])
            k_L_restaurant.append(k_array[1])
            k_L_school.append(k_array[2])
            k_L_healthcare.append(k_array[3])
            num_allocations_L.append(num_allocation if 'num_allocation' in locals() else None)
        else:
            k_L_grocery.append(0)
            k_L_restaurant.append(0)
            k_L_school.append(0)
            k_L_healthcare.append(0)
            num_allocations_L.append(None)

        obj_L.append(score_obj)
        solving_time_L.append(solving_time)
        num_residents_L.append(num_residents)
        status_L.append(status)

        # plotting 
        if args.k_array != '0,0,0' and args.k_array is not None:
            pass  # skip single-amenity plotting

        os.makedirs(summary_folder, exist_ok=True)
        results_D = {
            "nia_id": nia_id_L,
            "nia_name": nia_name_L,
            "k_grocery": k_L_grocery,
            "k_restaurant": k_L_restaurant,
            "k_school": k_L_school,
            "k_healthcare": k_L_healthcare,
            "obj": obj_L,
            "dist_obj_grocery": dist_obj_L_grocery,
            "dist_obj_restaurant": dist_obj_L_restaurant,
            "dist_obj_school": dist_obj_L_school,
            "dist_obj_healthcare": dist_obj_L_healthcare,
            "solving_time": solving_time_L,
            "num_res": num_residents_L,
            "num_parking": num_allocations_L,
            "num_cur_grocery": num_existing_L_grocery,
            "num_cur_restaurant": num_existing_L_restaurant,
            "num_cur_school": num_existing_L_school,
            "num_cur_healthcare": num_existing_L_healthcare,
            "model_status": status_L
        }
        summary_df_filename = os.path.join(summary_folder, "NIA_%s_%s_summary.csv" % (nia_id, model_save_name))
        pd.DataFrame(results_D).to_csv(summary_df_filename, index=False)
