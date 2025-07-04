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
from greedy import greedy_multiple_depth, greedy_multiple, greedy_single, greedy_single_depth, get_nearest, greedy_multiple_lazy
import sys
from docplex.cp.config import context

# Update this line with the exact path to cpoptimizer.exe
context.solver.local.execfile = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cpoptimizer\bin\x64_win64\cpoptimizer.exe"




parser = argparse.ArgumentParser(description='Enter model name:grb_PWL,scratch')
parser.add_argument("model", help="model", type=str)
parser.add_argument("nias", help="nias to run", type=str)
parser.add_argument("--cc", help="run on compute canada?", type=bool)
parser.add_argument("--amenity", help="run on compute canada?", type=str)
parser.add_argument("--k", help="upper bound", type=int)
parser.add_argument("--k_array", help="upper bound", type=str)
parser.add_argument("--bp", help="whether to set branching priority", default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--focus", help="MIPFocus parameter", default=0,type=int)

# NEW alter args = parser.parse_args() with: 
args, unknown = parser.parse_known_args()

#NEW ADDED FEATURE FOR CHECKING: 
print("Received arguments:", sys.argv)

if unknown:
    print("Ignoring unknown arguments:", unknown)

if args.cc:
    data_root = "/home/huangw98/projects/def-khalile2/huangw98/walkability_datapi"
    preprocessing_folder = "./preprocessing"
    threads = 8
    solver_path = "/home/huangw98/modulefiles/mycplex/cpoptimizer/bin/x86-64_linux/cpoptimizer"
else:
    data_root = r"C:\Users\annve\Downloads\AAAI23-WalkabilityOptimization" #NEW ALTERED
    preprocessing_folder = "./preprocessing"
    threads = 12
    # CHANGED 18 to 12
    solver_path = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cpoptimizer\bin\x64_win64\cpoptimizer.exe"
    

# NEW Construct the file path using os.path.join with separate arguments.
file_path = os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx")
#NEW ADDED FEATURE FOR CHECKING: print("Looking for file at:", file_path)

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

#NEW ADDED FEATURE FOR CHECKING Display the first few rows
#print(df.head())  # Print first 5 rows


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
Path(sol_folder).mkdir(parents=True,exist_ok=True)
Path(summary_folder).mkdir(parents=True,exist_ok=True)

if __name__ == "__main__":
    nia_list = [int(x) for x in args.nias.split(',') if x.strip().isdigit()]
    #  NEW ADDED block
    nia_remap = {
        137: [141, 142],
        26: [154, 155]
    }

    expanded_nia_list = []
    for nia in nia_list:
        if nia in nia_remap:
            expanded_nia_list.extend(nia_remap[nia])
        else:
            expanded_nia_list.append(nia)
    nia_list = expanded_nia_list
    # --- end of added block
    
    pednet = load_pednet(data_root)
    nia_id_L = []
    nia_name_L = []
    obj_L = []
    solving_time_L = []
    num_residents_L = []
    num_allocations_L = []
    status_L = []

    if args.model in ['OptSingle', 'OptSingleDepth','OptSingleCP','OptSingleDepthCP','GreedySingle','GreedySingleDepth']:
        num_existing_L = []
        dist_obj_L = []
        k_L = []

    elif args.model in ['OptMultiple', 'OptMultipleDepth','OptMultipleCP','OptMultipleDepthCP','GreedyMultipleDepth','GreedyMultiple','GreedyMultipleLazy']:
        num_existing_L_grocery, num_existing_L_restaurant, num_existing_L_school = [], [], []
        dist_obj_L_grocery, dist_obj_L_restaurant, dist_obj_L_school = [], [], []
        k_L_grocery, k_L_restaurant, k_L_school = [], [], []

    for nia_id in nia_list:

        pednet_nia = pednet_NIA(pednet, nia_id, preprocessing_folder)
        print("NIA ",nia_id)

        #NEW UNCOMMENT load net
        prec = 2
        net_filename = "NIA_%s_prec_%s.hd5" % (nia_id, prec)
        if os.path.exists(os.path.join(net_save_path, net_filename)):
            transit_ped_net = pdna.Network.from_hdf5(os.path.join(net_save_path, net_filename))
        else:
            G = create_graph(pednet_nia, precision=prec)
            transit_ped_net = get_pandana_net(G, os.path.join(net_save_path, net_filename))

        # load dfs
        all_strs = ['residential', 'department_store', 'parking', 'grocery', 'school', 'cafe', 'restaurant']
        colors = ['g', 'lightcoral', 'grey', 'red', 'yellow', 'brown', 'orange']
        df_filenames = ["NIA_%s_%s.pkl" % (nia_id, str) for str in all_strs]
        all_dfs = [pd.read_pickle(os.path.join(df_save_path, df_filename)) for df_filename in df_filenames]
        residentials_df, department_store_df, parking_df, grocery_df, school_df, cafe_df, restaurant_df = all_dfs

        # load SP
        SP_filename = "NIA_%s_prec_%s.txt" % (nia_id, prec)
        D = np.loadtxt(os.path.join(sp_save_path, SP_filename))

        if args.model in ['OptSingle','OptSingleCP','GreedySingle']:
            amenity_type = args.amenity
            amenity_df = all_dfs[all_strs.index(args.amenity)]
            if args.k:
                log_file_name = os.path.join(sol_folder, "log_NIA_%s_%s_%s.txt" % (nia_id, args.k, args.amenity))
                if not 'CP' in args.model:
                    if (not 'Greedy' in args.model):
                        score_obj, dist_obj, solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, num_existing, status = opt_single(
                            residentials_df, parking_df, amenity_df, D, args.k, threads, log_file_name, args.bp, args.focus, EPS = 0.5)
                    else:
                        score_obj, dist_obj, solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, num_existing, status = greedy_single(
                            residentials_df, parking_df, amenity_df, D, args.k)
                else:
                    score_obj, dist_obj, solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, num_existing, status = opt_single_CP(
                        residentials_df, parking_df, amenity_df, D, args.k, threads, log_file_name, solver_path, EPS=0.5)

            else:
                log_file_name = os.path.join(sol_folder, "log_NIA_%s_%s_%s.txt" % (nia_id, 0, args.amenity))
                score_obj, dist_obj, solving_time, m, assigned_D, num_residents, num_existing, status = cur_assignment_single(residentials_df,amenity_df, D,args.bp, args.focus,EPS=0.5)
        elif args.model in ['OptSingleDepth','OptSingleDepthCP','GreedySingleDepth']:
            amenity_type = args.amenity
            amenity_df = all_dfs[all_strs.index(args.amenity)]
            if args.k:
                log_file_name = os.path.join(sol_folder,"log_NIA_%s_%s_%s.txt" % (nia_id, args.k, args.amenity))
                if not 'CP' in args.model:
                    if (not 'Greedy' in args.model):
                        score_obj, dist_obj, solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, num_existing, status = opt_single_depth(
                            residentials_df, parking_df, amenity_df, D, args.k, threads, log_file_name, args.bp, args.focus, EPS=0.5)
                    else:
                        score_obj, dist_obj, solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, num_existing, status = greedy_single_depth(
                            residentials_df, parking_df, amenity_df, D, args.k)
                else:
                    score_obj, dist_obj, solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, num_existing, status = opt_single_depth_CP(
                        residentials_df, parking_df, amenity_df, D, args.k, threads, log_file_name, solver_path,args.bp, EPS=0.5)
            else:
                log_file_name = os.path.join(sol_folder, "log_NIA_%s_%s_%s.txt" % (nia_id, 0, args.amenity))
                score_obj, dist_obj, solving_time, m, assigned_D, num_residents, num_existing, status = cur_assignment_single_depth(residentials_df,amenity_df, D,args.bp, args.focus,EPS=0.5)

        elif args.model in ['OptMultiple','OptMultipleCP', 'GreedyMultiple','GreedyMultipleLazy']:
            if args.k_array != '0,0,0':
                k_array = [int(x) for x in args.k_array.split(',')]
                log_file_name = os.path.join(sol_folder, "log_NIA_%s_%s.txt" % (nia_id, args.k_array))
                if not 'CP' in args.model:
                    if not 'Greedy' in args.model:
                        score_obj, [dist_grocery, dist_restaurant, dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status\
                            = opt_multiple(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D, k_array,threads, log_file_name,args.bp, args.focus, EPS = 0.5)
                    else:
                        if 'Lazy' in args.model:
                            score_obj, [dist_grocery, dist_restaurant,dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status \
                                = greedy_multiple_lazy(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D,
                                              k_array)
                        else:
                            score_obj, [dist_grocery, dist_restaurant, dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status \
                                = greedy_multiple(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D, k_array)
                else:
                    score_obj, [dist_grocery, dist_restaurant,dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status \
                        = opt_multiple_CP(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D, k_array,threads, log_file_name, solver_path, EPS = 0.5)
            else:
                multiple_dist = []
                # grocery
                score_obj, dist_grocery, solving_time, m, assigned_D, num_residents, num_cur_grocery, status = cur_assignment_single(residentials_df, grocery_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)
                # what if assigned_D is None
                # restaurant
                score_obj, dist_restaurant, solving_time, m, assigned_D, num_residents, num_cur_restaurant, status = cur_assignment_single(residentials_df, restaurant_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)
                # school
                score_obj, dist_school, solving_time, m, assigned_D, num_residents, num_cur_school, status = cur_assignment_single(residentials_df, school_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)

                multiple_dist = np.array(multiple_dist)
                weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)

                solving_time=None
                status=None

        elif args.model in ['OptMultipleDepth', 'OptMultipleDepthCP', 'GreedyMultipleDepth']:
            if args.k_array != '0,0,0':
                k_array = [int(x) for x in args.k_array.split(',')]
                log_file_name = os.path.join(sol_folder, "log_NIA_%s_%s.txt" % (nia_id, args.k_array))
                if not 'CP' in args.model:
                    if not 'Greedy' in args.model:
                        score_obj, [dist_grocery, dist_restaurant, dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status\
                            = opt_multiple_depth(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D, k_array,threads, log_file_name,args.bp, args.focus, EPS = 0.5)
                    else:
                        score_obj, [dist_grocery, dist_restaurant, dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status \
                            = greedy_multiple_depth(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D, k_array)
                else:
                    score_obj, [dist_grocery, dist_restaurant, dist_school], solving_time, m, allocated_D, assigned_D, num_residents, num_allocation, [num_cur_grocery, num_cur_restaurant, num_cur_school], status \
                        = opt_multiple_depth_CP(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D, k_array, threads, log_file_name, solver_path, args.bp, EPS=0.5)
            else:
                multiple_dist = []
                # grocery
                score_obj, dist_grocery, solving_time, m, assigned_D, num_residents, num_cur_grocery, status = cur_assignment_single(residentials_df, grocery_df, D, args.bp, args.focus, EPS=0.5)
                if assigned_D:
                    multiple_dist.append(assigned_D["dist"])
                else:
                    multiple_dist.append([L_a[-2]] * num_residents)
                # restaurant
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

                #TODO: finish this calculation: need to re-define weights

                multiple_dist = np.array(multiple_dist)
                weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)

                solving_time=None
                status=None

                assigned_D = get_nearest(residentials_df, parking_df, grocery_df, restaurant_df, school_df, D)


        else:
            print("choose model name")

        # save allocated results for mapping
        if args.model in ['OptSingle','OptSingleDepth','OptSingleCP','OptSingleDepthCP','GreedySingle','GreedySingleDepth']:

            if args.k:
                k_name = args.k
                allocated_f_name = os.path.join(sol_folder,"allocation_NIA_%s_%s_%s.csv" % (nia_id, k_name, args.amenity))
                pd.DataFrame.from_dict(allocated_D).to_csv(allocated_f_name)
            else:
                k_name = 0

            assigned_f_name = os.path.join(sol_folder, "assignment_NIA_%s_%s_%s.csv" % (nia_id, k_name, args.amenity))
            model_f_name = os.path.join(sol_folder, "NIA_%s_%s_%s.sol" % (nia_id, k_name, args.amenity))
        elif args.model in ['OptMultiple','OptMultipleDepth','OptMultipleCP','GreedyMultipleDepth','OptMultipleDepthCP', 'GreedyMultiple','GreedyMultipleLazy']:
            if args.k_array != '0,0,0':
                k_name = args.k_array
                #allocated_f_name = os.path.join(sol_folder, "allocation_NIA_%s_%s.csv" % (nia_id, k_name))
                allocated_f_name = os.path.join(sol_folder, "allocation_NIA_%s_%s.pkl" % (nia_id, k_name))
                #pd.DataFrame.from_dict(allocated_D).to_csv(allocated_f_name)
                with open(allocated_f_name, 'wb') as f:
                    pickle.dump(allocated_D, f)
            else:
                k_name = '0,0,0'

            assigned_f_name = os.path.join(sol_folder, "assignment_NIA_%s_%s.csv" % (nia_id, k_name))
            model_f_name = os.path.join(sol_folder, "NIA_%s_%s.sol" % (nia_id, k_name))

        if assigned_D:
            pd.DataFrame.from_dict(assigned_D).to_csv(assigned_f_name)
        if m:
            m.write(model_f_name)

        # write log
        # text_file = open(os.path.join(log_folder, args.model + '_' + str(nia_id) + '.txt'), "w")
        # text_file.write(log)
        # text_file.close()

        print("D_NIA keys:", list(D_NIA.keys()))  # Print all available keys in D_NIA
        print("Attempting to access nia_id:", nia_id)  # Print the nia_id value before accessing

        # Check if the key exists before accessing it
        if nia_id in D_NIA:
            print("Key exists! Proceeding with access.")
            nia_name_L.append(D_NIA[nia_id]['name'])
        else:
            print(f"Key {nia_id} not found in D_NIA!")
            nia_name_L.append("Unknown")  # Handle missing key case
    
            
        # save summary
        nia_id_L.append(nia_id)

        if args.model in ['OptSingle', 'OptSingleDepth','OptSingleCP','OptSingleDepthCP','GreedySingle','GreedySingleDepth']:
            dist_obj_L.append(dist_obj)
            num_existing_L.append(num_existing)
            if args.k:
                k_L.append(args.k)
                num_allocations_L.append(num_allocation)
            else:
                k_L.append(0)
                num_allocations_L.append(None)

        elif args.model in ['OptMultiple', 'OptMultipleDepth', 'OptMultipleCP', 'GreedyMultipleDepth','OptMultipleDepthCP', 'GreedyMultiple','GreedyMultipleLazy']:
            num_existing_L_grocery.append(num_cur_grocery)
            num_existing_L_restaurant.append(num_cur_restaurant)
            num_existing_L_school.append(num_cur_school)

            dist_obj_L_grocery.append(dist_grocery)
            dist_obj_L_restaurant.append(dist_restaurant)
            dist_obj_L_school.append(dist_school)
            if args.k_array != '0,0,0':
                k_L_grocery.append(k_array[0])
                k_L_restaurant.append(k_array[1])
                k_L_school.append(k_array[2])
                num_allocations_L.append(num_allocation)
            else:
                k_L_grocery.append(0)
                k_L_restaurant.append(0)
                k_L_school.append(0)
                num_allocations_L.append(None)

        obj_L.append(score_obj)
        solving_time_L.append(solving_time)
        num_residents_L.append(num_residents)
        status_L.append(status)

        # plot
        if args.k:
            ax = pednet_nia.plot(figsize=(15, 15), color='blue', markersize=1)
            res = residentials_df.plot(ax=ax,color='green', markersize=80, label='Residential location')
            parking = parking_df.plot(ax=ax,color='gray', markersize=80, label='Parking lot') #,fontsize=20
            if args.amenity:
                if len(amenity_df)>0:
                    amenity_df.plot(ax=ax, color=colors[all_strs.index(args.amenity)], markersize=120, label='Existing')

            allocated_df = parking_df.iloc[allocated_D["allocate_row_id"]]
            allocated_df.plot(ax=ax, color='fuchsia', markersize=80, label='Allocated amenity')

            pink_patch = mpatches.Patch(color='fuchsia', label='Allocated amenity')
            green_patch = mpatches.Patch(color='green', label='Residential location')
            gray_patch = mpatches.Patch(color='gray', label='Parking lot')

            if len(amenity_df)>0:
                    last_patch = mpatches.Patch(color=colors[all_strs.index(args.amenity)], label='Existing amenity')

            if (len(amenity_df)>0):
                    plt.legend(handles=[pink_patch,green_patch,gray_patch,last_patch])
            else:
                plt.legend(handles=[pink_patch,green_patch,gray_patch])


            plt.title("neighbourhood: %s" % (D_NIA[nia_id]['name']))
            if args.k:
                fig_name = "nia_%s_%s_allocation_%s.png" % (nia_id, args.k, args.amenity)
            else:
                fig_name = "nia_%s_%s_allocation_%s.png" % (nia_id, 0, args.amenity)


            print("Saving figure to:", os.path.join(visual_folder, fig_name))
            plt.savefig(os.path.join(visual_folder, fig_name))
            print("Figure saved.")

            plt.savefig(os.path.join(visual_folder,fig_name))

        # save results summary
        os.makedirs(summary_folder, exist_ok=True)
        if args.model in ['OptSingle', 'OptSingleDepth','OptSingleCP','OptSingleDepthCP','GreedySingle','GreedySingleDepth']:
            results_D = {
                "nia_id": nia_id_L,
                "nia_name": nia_name_L,
                "k": k_L,
                "obj": obj_L,
                "dist_obj": dist_obj_L,
                "solving_time": solving_time_L,
                "num_res": num_residents_L,
                "num_parking": num_allocations_L,
                "num_cur": num_existing_L,
                "model_status": status_L
            }
            summary_df_filename = os.path.join(summary_folder, "NIA_%s_%s_%s.csv" % (nia_id, k_name, args.amenity))

        elif args.model in ['OptMultiple', 'OptMultipleDepth','OptMultipleCP','GreedyMultipleDepth','OptMultipleDepthCP', 'GreedyMultiple','GreedyMultipleLazy']:
            results_D = {
                "nia_id": nia_id_L,
                "nia_name": nia_name_L,
                "k_L_grocery": k_L_grocery, "k_L_restaurant": k_L_restaurant, "k_L_school": k_L_school,
                "obj": obj_L,
                "dist_obj_L_grocery": dist_obj_L_grocery, "dist_obj_L_restaurant": dist_obj_L_restaurant, "dist_obj_L_school": dist_obj_L_school,
                "solving_time": solving_time_L,
                "num_res": num_residents_L,
                "num_parking": num_allocations_L,
                "num_existing_L_grocery": num_existing_L_grocery, "num_existing_L_restaurant": num_existing_L_restaurant, "num_existing_L_school": num_existing_L_school,
                "model_status": status_L
            }
            summary_df_filename = os.path.join(summary_folder, "NIA_%s_%s.csv" % (nia_id, k_name))
            
        expected_len = len(next(iter(results_D.values())))
        for key, val in results_D.items():
            if len(val) != expected_len:
                print(f"❌ Mismatch in '{key}': expected {expected_len}, got {len(val)}")


        summary_df = pd.DataFrame(results_D)
        summary_df.to_csv(summary_df_filename,index=False)
        
    
        
        '''
        # Print lengths of all lists in results_D
        list_lengths = {key: len(value) for key, value in results_D.items()}
        print("List lengths:", list_lengths)

            # Find the unique lengths (should be just one unique value)
        unique_lengths = set(list_lengths.values())
        print("Unique lengths:", unique_lengths)

            # Find the maximum list length
        max_len = max(list_lengths.values())

            # Identify lists with incorrect length
        for key, length in list_lengths.items():
            if length != max_len:
                print(f"⚠️ {key} has length {length}, expected {max_len}")    '''

        

