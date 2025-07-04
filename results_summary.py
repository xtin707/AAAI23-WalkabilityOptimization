import glob
import pandas as pd
import os
from map_utils import ct_nia_mapping,get_nias
from graph_utils import pednet_NIA
import matplotlib.pyplot as plt
import numpy as np
import json
import geopandas as gpd
import matplotlib.ticker as ticker 
from matplotlib.ticker import MultipleLocator
from model_latest import opt_single, cur_assignment_single, opt_multiple, opt_single_depth, cur_assignment_single_depth, weights_array, dist_to_score, L_a, L_f_a, weights_array_multi, choice_weights

def ensure_directory_exists(folder_path): 
    """Ensure that a directory exists. If a file path is given, create its parent directory."""
    if not folder_path:
        return  # Prevent errors if an empty path is given

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    
def shifted_geo_mean(L, s=1):
    a = np.array(L)
    shifted = a+s
    return (shifted.prod())**(1.0/len(a)) - s

def get_results_df(results_folder, model_name,amenity_name=None):

    L = []
    if amenity_name:
        all_files = glob.glob(os.path.join(results_folder,"summary",model_name) + "/*" + amenity_name + ".csv")
    else:
        all_files = glob.glob(os.path.join(results_folder, "summary", model_name) + "/*" + ".csv")
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        L.append(df)
    final_df = pd.concat(L, axis=0, ignore_index=True)

    return final_df

def plot_time_vs_size_multiple(results_folder, plot_folder, models, display_names, save_name):

    data_root =  r"C:\Users\annve\Downloads\AAAI23-WalkabilityOptimization"
    D_NIA = ct_nia_mapping(
        os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))
    plt.clf()


    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        results_df = get_results_df(results_folder, model_name)
        results_df = results_df[(results_df["k_L_grocery"]>0) & (results_df["k_L_restaurant"]>0) & (results_df["k_L_school"]>0)]
        for nia in list(D_NIA.keys()):
            nia_df = results_df[results_df["nia_id"] == nia]
            if len(nia_df) < 9:
                    print("missing for",nia)
        size = results_df.groupby("nia_id").mean()["num_res"] + results_df.groupby("nia_id").mean()["num_parking"]
        avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
        new_x, new_y = zip(*sorted(zip(size, avg_time)))

        plt.plot(new_x, new_y, '--o', label=display_name)
        plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Shifted geo mean")
    plt.title(save_name)
    
    # Ensure the directory exists before saving the plot (F1)
    save_path = os.path.join(plot_folder, save_name)
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)
    return

def plot_time_vs_size_single(results_folder, plot_folder, models, display_names, save_name):
    plt.clf()

    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        if "Depth" in model_name:
            amenities = ["restaurant"]
        else:
            amenities = ["grocery","restaurant","school"]

        for amenity in amenities:

            results_df = get_results_df(results_folder, model_name, amenity)
            results_df=results_df[results_df["k"]>0]
            size = results_df.groupby("nia_id").mean()["num_res"] + results_df.groupby("nia_id").mean()["num_parking"]
            avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
            new_x, new_y = zip(*sorted(zip(size, avg_time)))

            plt.plot(new_x, new_y, '--o', label=display_name + "-"+ amenity)
            plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Shifted geo mean")
    plt.title(save_name)
    
    # Ensure the directory exists before saving the plot (F2)
    save_path = os.path.join(plot_folder, save_name)
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)
    

    return


def plot_obj_vs_k(results_df, plot_folder, amenity_name, model_name, display_name):

    data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
    D_NIA = ct_nia_mapping(os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))

    if 'Depth' in model_name:
        for ind in [0,1]:
            plt.clf()
            plt.figure(figsize=(15, 10))
            for nia in list(D_NIA.keys()):
                nia_df = results_df[results_df["nia_id"] == nia]
                nia_df = nia_df.sort_values(by=['k'])
                k_list = nia_df["k"]
                dist_list = [L[ind] for L in [json.loads(x) for x in nia_df["dist_obj"]]]
                dist_list = [None if item == 0 else item for item in dist_list]

                plt.plot(k_list, dist_list, '--o', label=D_NIA[nia]['name'])
                plt.legend(prop={'size': 6})
                plt.xlabel("k")
                plt.ylabel("dist (m)")
            # Ensure the directory exists before saving the plot (F3)
            save_path = os.path.join(plot_folder, f"{amenity_name}_{display_name}_k_vs_dist_{ind+1}.png")
            ensure_directory_exists(os.path.dirname(save_path))
            plt.savefig(save_path)
            



    else:
        plt.clf()
        plt.figure(figsize=(15, 10))
        for nia in list(D_NIA.keys()):
            nia_df=results_df[results_df["nia_id"]==nia]
            nia_df=nia_df.sort_values(by=['k'])
            k_list = nia_df["k"]
            dist_list = nia_df["dist_obj"]

            plt.plot(k_list, dist_list, '--o', label=D_NIA[nia]['name'])
            plt.legend(prop={'size': 6})
            plt.xlabel("k")
            plt.ylabel("dist (m)")
        # Ensure the directory exists before saving the plot (F4)
        save_path = os.path.join(plot_folder, f"{amenity_name}_{display_name}_k_vs_dist.png")
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path)

    plt.clf()
    plt.figure(figsize=(15, 10))

    for nia in list(D_NIA.keys()):
        nia_df = results_df[results_df["nia_id"] == nia]
        if len(nia_df)<10:
            if len(nia_df[nia_df['k']==0])>0:
                print("missing values?")
                print(model_name,amenity_name,nia,len(nia_df))
        nia_df = nia_df.sort_values(by=['k'])
        k_list = nia_df["k"]
        score_list = nia_df["obj"]

        plt.plot(k_list, score_list, '--o', label=D_NIA[nia]['name'])
        plt.legend(prop={'size': 6})
        plt.xlabel("k")
        plt.ylabel("score")
    # Ensure the directory exists before saving the plot (F5)
    save_path = os.path.join(plot_folder, f"{amenity_name}_{display_name}_k_vs_score.png")
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)

    return


def all_instances_obj(data_root, results_folder,processed_folder):
    INCLUDE_BP=False
    D_NIA = ct_nia_mapping( os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # opt single
    print("single case")
    amenities = ["grocery", "restaurant", "school"]
    for amenity in amenities:
        L_nia = []
        L_mip_obj = []
        L_cp_obj = []
        L_greedy_obj = []
        if INCLUDE_BP:
            L_mip_bp_obj = []
            L_cp_bp_obj = []
        L_k = []
        L_mip_status = []
        L_cp_status = []

        for nia in list(D_NIA.keys()):
            print("nia:",nia)
            for k in range(1,10):
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia,k,amenity)

                L_nia.append(nia)
                L_k.append(k)
                #  MIP
                if os.path.exists(os.path.join(results_folder,"sol","OptSingle_False_0",filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder,"sol","OptSingle_False_0",filename), index_col=None, header=0)
                    scores = dist_to_score(np.array(mip_df["dist"]), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_mip_obj.append(score_obj)
                    #L_mip_status.append(mip_df["model_status"])
                else:
                    L_mip_obj.append(None)
                    #L_mip_status.append("null")
                # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename),index_col=None, header=0)
                    scores = dist_to_score(np.array(cp_df["dist"]), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_obj.append(score_obj)
                    #L_cp_status.append(cp_df["model_status"])
                else:
                    L_cp_obj.append(None)
                    #L_cp_status.append("null")

                # Greedy
                if os.path.exists(os.path.join(results_folder, "summary", "GreedySingle_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity))):
                    greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedySingle_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity)), index_col=None,header=0)
                    L_greedy_obj.append(greedy_df["obj"].values[0])
                else:
                    L_greedy_obj.append(None)

        results_D = { "nia":L_nia, "k": L_k ,"mip":L_mip_obj,"cp":L_cp_obj,"greedy":L_greedy_obj,}
        df_filename = os.path.join(processed_folder, "single_%s.csv" % (amenity))
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename,index=False)

    # opt single depth
    print("single depth case")
    amenities = ["restaurant"]
    for amenity in amenities:
        L_nia = []
        L_mip_obj = []
        L_cp_obj = []
        L_greedy_obj = []
        if INCLUDE_BP:
            L_mip_bp_obj = []
            L_cp_bp_obj = []
        L_k = []
        L_mip_status = []
        L_cp_status = []

        for nia in list(D_NIA.keys()):
            print("nia:", nia)
            for k in range(1,10):
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia, k, amenity)

                L_nia.append(nia)
                L_k.append(k)
                #  MIP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleDepth_False_0", filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleDepth_False_0", filename),index_col=None, header=0)
                    choices_dist = [mip_df[str(c)+"_dist"] if (str(c)+"_dist") in mip_df.columns else L_a[-2] for c in range(10)]
                    choices_dist = np.array(choices_dist)
                    weighted_choices = np.dot(np.array(choice_weights), choices_dist)
                    scores = dist_to_score(np.array(weighted_choices), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_mip_obj.append(score_obj)
                    #L_mip_status.append(mip_df["model_status"])
                else:
                    L_mip_obj.append(None)
                    #L_mip_status.append("null")
                    # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptSingleDepthCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleDepthCP_False_0", filename),index_col=None, header=0)
                    choices_dist = [cp_df[str(c) + "_dist"] if (str(c) + "_dist") in cp_df.columns else L_a[-2] for c in range(10)]
                    choices_dist = np.array(choices_dist)
                    weighted_choices = np.dot(np.array(choice_weights), choices_dist)
                    scores = dist_to_score(np.array(weighted_choices), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_obj.append(score_obj)
                    #L_cp_status.append(cp_df["model_status"])
                else:
                    L_cp_obj.append(None)
                    #L_cp_status.append("null")

                # Greedy
                if os.path.exists(os.path.join(results_folder, "summary", "GreedySingleDepth_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity))):
                    greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedySingleDepth_False_0", "NIA_%s_%s_%s.csv" % (nia,k,amenity)),index_col=None, header=0)
                    L_greedy_obj.append(greedy_df["obj"].values[0])
                else:
                    L_greedy_obj.append(None)

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj }
        df_filename = os.path.join(processed_folder, "single_depth_%s.csv" % (amenity))
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)

    # opt multiple
    print("multiple case")
    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    if INCLUDE_BP:
        L_mip_bp_obj = []
    L_k = []
    L_mip_status = []
    L_cp_status = []

    for nia in list(D_NIA.keys()):
        print("nia:", nia)
        for k in range(1,10):
            filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia,k,k,k)

            L_nia.append(nia)
            L_k.append(k)
            #  MIP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultiple_False_0", filename)):
                mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultiple_False_0", filename),index_col=None, header=0)
                dist_grocery = mip_df["dist_grocery"]
                dist_restaurant = mip_df["dist_restaurant"]
                dist_school = mip_df["dist_school"]
                multiple_dist = np.array([dist_grocery, dist_restaurant, dist_school])
                weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_mip_obj.append(score_obj)
                #L_mip_status.append(mip_df["model_status"])
            else:
                L_mip_obj.append(None)
                #L_mip_status.append("null")
            if INCLUDE_BP:

                #  MIP - branch priority
                if os.path.exists(os.path.join(results_folder, "sol", "OptMultiple_True_0", filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultiple_True_0", filename),
                                         index_col=None, header=0)
                    dist_grocery = mip_df["dist_grocery"]
                    dist_restaurant = mip_df["dist_restaurant"]
                    dist_school = mip_df["dist_school"]
                    multiple_dist = np.array([dist_grocery, dist_restaurant, dist_school])
                    weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                    scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_mip_bp_obj.append(score_obj)
                    # L_mip_status.append(mip_df["model_status"])
                else:
                    L_mip_bp_obj.append(None)
                    # L_mip_status.append("null")
            # CP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleCP_False_0", filename)):
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleCP_False_0", filename),index_col=None, header=0)
                dist_grocery = cp_df["dist_grocery"]
                dist_restaurant = cp_df["dist_restaurant"]
                dist_school = cp_df["dist_school"]
                multiple_dist = np.array([dist_grocery, dist_restaurant, dist_school])
                weighted_dist = np.dot(np.array(weights_array), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_cp_obj.append(score_obj)
                #L_cp_status.append(cp_df["model_status"])
            else:
                L_cp_obj.append(None)
                #L_cp_status.append("null")

            # Greedy
            if os.path.exists(os.path.join(results_folder, "summary", "GreedyMultiple_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k))):
                #greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultiple_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k)),index_col=None, header=0)
                file_path = os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0",
                         "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k))
                if os.path.exists(file_path):
                    greedy_df = pd.read_csv(file_path, index_col=None, header=0)
                    L_greedy_obj.append(greedy_df["obj"].values[0])
                else:
                    print(f"Skipping NIA {nia} with k={k} — file not found.")
                               
            else:
                L_greedy_obj.append(None)


    if INCLUDE_BP:

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj,"mip-bp": L_mip_bp_obj}
        df_filename = os.path.join(processed_folder, "multiple.csv")
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy","mip-bp"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)
    else:
        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj}
        df_filename = os.path.join(processed_folder, "multiple.csv" )
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)


    # opt multiple depth
    print("multiple depth case")
    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    if INCLUDE_BP:
        L_mip_bp_obj = []
        L_cp_bp_obj = []
    L_k = []
    L_mip_status = []
    L_cp_status = []

    for nia in list(D_NIA.keys()):
        print("nia:", nia)
        for k in range(1,10):
            filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

            L_nia.append(nia)
            L_k.append(k)
            #  MIP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename)):
                mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename),index_col=None, header=0)
                dist_grocery = mip_df["dist_grocery"]
                choices_dist = [mip_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = mip_df["dist_school"]
                multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_mip_obj.append(score_obj)
                #L_mip_status.append(mip_df["model_status"])
            else:
                L_mip_obj.append(None)
                #L_mip_status.append("null")
            if INCLUDE_BP:

                #  MIP - branch priority
                if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepth_True_0", filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepth_True_0", filename),
                                         index_col=None, header=0)
                    dist_grocery = mip_df["dist_grocery"]
                    choices_dist = [
                        mip_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[
                            -2] for c in range(10)]
                    dist_school = mip_df["dist_school"]
                    multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                    weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                    scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_mip_bp_obj.append(score_obj)
                    # L_mip_status.append(mip_df["model_status"])
                else:
                    L_mip_bp_obj.append(None)
                    # L_mip_status.append("null")

            # CP
            if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename)):
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename), index_col=None, header=0)
                dist_grocery = cp_df["dist_grocery"]
                choices_dist = [cp_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                dist_school = cp_df["dist_school"]
                multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                score_obj = np.mean(scores)
                L_cp_obj.append(score_obj)
                #L_cp_status.append(cp_df["model_status"])
            else:
                L_cp_obj.append(None)
                #L_cp_status.append("null")

            if INCLUDE_BP:

                # CP - branch priority
                if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepthCP_True_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepthCP_True_0", filename),
                                        index_col=None, header=0)
                    dist_grocery = cp_df["dist_grocery"]
                    choices_dist = [
                        cp_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[
                            -2] for c in range(10)]
                    dist_school = cp_df["dist_school"]
                    multiple_dist = np.array([dist_grocery] + choices_dist + [dist_school])
                    weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
                    scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
                    score_obj = np.mean(scores)
                    L_cp_bp_obj.append(score_obj)
                    # L_cp_status.append(cp_df["model_status"])
                else:
                    L_cp_bp_obj.append(None)
                    # L_cp_status.append("null")

            # Greedy
            if os.path.exists(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k))):
                greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", "NIA_%s_%s,%s,%s.csv" % (nia,k,k,k)),index_col=None, header=0)
                L_greedy_obj.append(greedy_df["obj"].values[0])
            else:
                L_greedy_obj.append(None)



    if INCLUDE_BP:

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj,"mip-bp": L_mip_bp_obj, "cp-bp": L_cp_bp_obj}
        df_filename = os.path.join(processed_folder, "multiple_depth.csv")
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy","mip-bp","cp-bp"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)
    else:

        results_D = {"nia": L_nia, "k": L_k, "mip": L_mip_obj, "cp": L_cp_obj, "greedy": L_greedy_obj}
        df_filename = os.path.join(processed_folder, "multiple_depth.csv")
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp","greedy"]].max(axis=1)
        summary_df.to_csv(df_filename, index=False)
    return


def single_aggregate_obj(data_root, results_folder,processed_folder):

    # NOTE: THIS IS A RELATXATION, AS IT ALLOWS COMPETING RESOURCES TO BE TOGETHER AND DISREGARDS THE CAPACITY CONSTRAINTS
    D_NIA = ct_nia_mapping( os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # no depth
    print("single aggregate, no depth")

    L_nia = []
    L_mip_obj = []
    L_cp_obj = []
    L_greedy_obj = []
    L_k = []

    for nia in list(D_NIA.keys()):
        print("nia:",nia)
        for k in range(1,10):
            L_nia.append(nia)
            L_k.append(k)

            amenities = ["grocery", "restaurant", "school"]

            #  MIP
            all_type_dist=[]
            for amenity in amenities:
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia,k,amenity)
                mip_df = pd.read_csv(os.path.join(results_folder,"sol","OptSingle_False_0",filename), index_col=None, header=0)
                all_type_dist.append(mip_df["dist"])
            multiple_dist = np.array(all_type_dist)
            weighted_dist = np.dot(np.array(weights_array), multiple_dist)
            scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
            score_obj = np.mean(scores)
            L_mip_obj.append(score_obj)

            # CP
            all_type_dist = []
            for amenity in amenities:
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia, k, amenity)
                cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptSingleCP_False_0", filename),index_col=None, header=0)
                all_type_dist.append(cp_df["dist"])
            multiple_dist = np.array(all_type_dist)
            weighted_dist = np.dot(np.array(weights_array), multiple_dist)
            scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
            score_obj = np.mean(scores)
            L_cp_obj.append(score_obj)

            # Greedy
            all_type_dist = []
            for amenity in amenities:
                filename = "assignment_NIA_%s_%s_%s.csv" % (nia, k, amenity)
                greedy_df = pd.read_csv(os.path.join(results_folder, "sol", "GreedySingle_False_0", filename), index_col=None,header=0)
                all_type_dist.append(greedy_df["dist"])
            multiple_dist = np.array(all_type_dist)
            weighted_dist = np.dot(np.array(weights_array), multiple_dist)
            scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
            score_obj = np.mean(scores)
            L_greedy_obj.append(score_obj)


        results_D = { "nia":L_nia, "k": L_k ,"mip":L_mip_obj,"cp":L_cp_obj,"greedy":L_greedy_obj}
        df_filename = os.path.join(processed_folder, "single_aggregate_no_depth.csv")
        summary_df = pd.DataFrame(results_D)
        summary_df["best"] = summary_df[["mip", "cp", "greedy"]].max(axis=1)
        summary_df.to_csv(df_filename,index=False)


    return

def plot_quality(processed_folder):

    data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
    D_NIA = ct_nia_mapping( os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))
    plt.clf()

    # single case

    for amenity in ["grocery","restaurant","school"]:
        L_size = []
        L_gap_mip = []
        L_gap_cp = []
        L_gap_greedy = []
        all_obj_df = pd.read_csv(os.path.join(processed_folder, "single_" + amenity+".csv"), index_col=None, header=0)
        for nia in list(D_NIA.keys()):
            results_df = get_results_df(results_folder, "OptSingle_False_0", amenity)
            results_df = results_df[results_df["nia_id"] == nia]
            obj_df = all_obj_df[all_obj_df["nia"]==nia]

            L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
            L_gap_mip.append(np.mean(obj_df['best']-obj_df['mip']))
            L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
            L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

        new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
        new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
        new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

        plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-"+ amenity)
        plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
        plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
        plt.legend(prop={'size': 6})

        plt.xlabel("|M|+|N|")
        plt.ylabel("Relative error")
        plt.title("solution quality - single amenity case")
        # Ensure the directory exists before saving the plot
        save_path = os.path.join(plot_folder, "quality", "single", "single_plot.png")
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path)


    plt.clf()

    for amenity in ["restaurant"]:
        L_size = []
        L_gap_mip = []
        L_gap_cp = []
        L_gap_greedy = []
        all_obj_df = pd.read_csv(os.path.join(processed_folder, "single_depth_" + amenity+".csv"), index_col=None, header=0)
        for nia in list(D_NIA.keys()):
            results_df = get_results_df(results_folder, "OptSingleDepth_False_0", amenity)
            results_df = results_df[results_df["nia_id"] == nia]
            obj_df = all_obj_df[all_obj_df["nia"]==nia]

            L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
            L_gap_mip.append(np.mean(obj_df['best']-obj_df['mip']))
            L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
            L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

        new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
        new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
        new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

        plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-"+ amenity)
        plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
        plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
        plt.legend(prop={'size': 6})

        plt.xlabel("|M|+|N|")
        plt.ylabel("Relative error")
        plt.title("solution quality - single amenity with depth of choice")
        
        # Ensure the directory exists before saving the plot
        save_path = os.path.join(plot_folder, "quality", "single_depth", "single_depth_plot.png")
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path)



    # multiple amenity case
    plt.clf()

    L_size = []
    L_gap_mip = []
    L_gap_cp = []
    L_gap_greedy = []
    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for nia in list(D_NIA.keys()):
        results_df = get_results_df(results_folder, "OptMultiple_False_0")
        results_df = results_df[results_df["nia_id"] == nia]
        obj_df = all_obj_df[all_obj_df["nia"] == nia]

        L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
        L_gap_mip.append(np.mean(obj_df['best'] - obj_df['mip']))
        L_gap_cp.append(np.mean(obj_df['best'] - obj_df['cp']))
        L_gap_greedy.append(np.mean(obj_df['best'] - obj_df['greedy']))

    new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
    new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
    new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

    plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-" + amenity)
    plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
    plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
    plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Relative error")
    plt.title("solution quality - multiple amenity case")
    
    # Ensure the directory exists before saving the plot
    save_path = os.path.join(plot_folder, "quality", "multiple", "multiple_plot.png")
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)


    # multiple amenity case with depth of choice
    plt.clf()
    L_size = []
    L_gap_mip = []
    L_gap_cp = []
    L_gap_greedy = []
    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for nia in list(D_NIA.keys()):
        results_df = get_results_df(results_folder, "OptMultipleDepth_False_0")
        results_df = results_df[results_df["nia_id"] == nia]
        obj_df = all_obj_df[all_obj_df["nia"] == nia]

        L_size.append(results_df["num_res"].mean() + results_df["num_parking"].mean())
        L_gap_mip.append(np.mean((obj_df['best'] - obj_df['mip'])/obj_df['best']))
        L_gap_cp.append(np.mean((obj_df['best'] - obj_df['cp'])/obj_df['best']))
        L_gap_greedy.append(np.mean((obj_df['best'] - obj_df['greedy'])/obj_df['best']))

    new_x, new_gap_mip = zip(*sorted(zip(L_size, L_gap_mip)))
    new_x, new_gap_cp = zip(*sorted(zip(L_size, L_gap_cp)))
    new_x, new_gap_greedy = zip(*sorted(zip(L_size, L_gap_greedy)))

    plt.plot(new_x, new_gap_mip, '--o', label="MILP" + "-" + amenity)
    plt.plot(new_x, new_gap_cp, '--o', label="CP" + "-" + amenity)
    plt.plot(new_x, new_gap_greedy, '--o', label="Greedy" + "-" + amenity)
    plt.legend(prop={'size': 6})

    plt.xlabel("|M|+|N|")
    plt.ylabel("Relative error")
    plt.title("solution quality - multiple amenity with depth of choice")
    
    # Ensure the directory exists before saving the plot
    save_path = os.path.join(plot_folder, "quality", "multiple_depth", "multiple_depth_plot.png")
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)
    return


def network_charac(data_root, results_folder,processed_folder):


    D_NIA = ct_nia_mapping( os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))

    L_nia = []

    L_m = []
    L_n = []

    L_num_grocery = []
    L_num_res = []
    L_num_school = []

    for nia in list(D_NIA.keys()):
        print("nia:",nia)

        filename = "NIA_%s_%s,%s,%s.csv" % (nia,1,1,1)
        L_nia.append(nia)

        #  MIP

        mip_df = pd.read_csv(os.path.join(results_folder,"summary","OptMultiple_False_0",filename), index_col=None, header=0)

        L_m.append(mip_df["num_parking"][0])
        L_n.append(mip_df["num_res"][0])
        L_num_grocery.append(mip_df["num_existing_L_grocery"][0])
        L_num_res.append(mip_df["num_existing_L_restaurant"][0])
        L_num_school.append(mip_df["num_existing_L_school"][0])

    results_D = { "nia":L_nia, "n": L_n ,"m":L_m,"grocery":L_num_grocery,"res":L_num_res,"school":L_num_school}
    df_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.DataFrame(results_D)
    summary_df["m+n"]=summary_df["m"]+summary_df["n"]
    summary_df.to_csv(df_filename,index=False)


    # m_split = 60
    # n_split = 200

    # print("m>=80,n>=200",len(summary_df[(summary_df["m"] >= m_split) & (summary_df["n"]>=n_split)]))
    # print("m>=80,n<200", len(summary_df[(summary_df["m"] >= m_split) & (summary_df["n"] < n_split)]))
    # print("m<80,n>=200", len(summary_df[(summary_df["m"] < m_split) & (summary_df["n"] >= n_split)]))
    # print("m<80,n<00", len(summary_df[(summary_df["m"] < m_split) & (summary_df["n"] < n_split)]))

    # plt.scatter(L_m, L_n, s=np.array(L_num_grocery)**2, alpha=0.5)
    # plt.show()

    output_D = {}

    model = "OptMultiple_False_0"
    group_thres = [0, 200, 400, 600, 2000]


    output_D["group"] = []
    output_D["num_nia"] = []
    output_D["num_ins"] = []
    output_D["avg grocery"] = []
    output_D["avg res"] = []
    output_D["avg school"] = []

    results_df = get_results_df(results_folder, model)
    results_df = results_df[(results_df["k_L_grocery"] > 0) & (results_df["k_L_restaurant"] > 0) & (results_df["k_L_school"] > 0)]
    results_df["m+n"] = results_df["num_res"] + results_df["num_parking"]

    for p in range(len(group_thres) - 1):

        group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p + 1])]
        output_D["num_ins"].append(len(group_df))
        group_df = group_df[group_df["k_L_grocery"]==1]
        output_D["num_nia"].append(len(group_df))

        output_D["group"].append(str(group_thres[p])+"-"+str(group_thres[p+1]))


        output_D["avg grocery"].append(group_df["num_existing_L_grocery"].mean())
        output_D["avg res"].append(group_df["num_existing_L_restaurant"].mean())
        output_D["avg school"].append(group_df["num_existing_L_school"].mean())

    df_filename = os.path.join(processed_folder, "instances_summary.csv")
    output_df = pd.DataFrame(output_D)
    output_df["avg grocery"] = output_df["avg grocery"].map('{:.2f}'.format)
    output_df["avg res"] = output_df["avg res"].map('{:.2f}'.format)
    output_df["avg school"] = output_df["avg school"].map('{:.2f}'.format)
    output_df.to_csv(df_filename, index=False)
    print(output_df.to_latex(index=False))
    return

    return

def plot_time_by_group_multiple(results_folder, plot_folder, models, display_names, save_name):
    plt.figure(dpi=300)
    group_gap = 1
    model_gap = 0.2

    all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
    D_NIA = ct_nia_mapping(
         os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))
    plt.clf()

    group_thres = [0,200,400,600,1115]

    def define_box_properties(plot_name, color_code, label):

        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(plot_name[element], color=color_code)
        plt.setp(plot_name["fliers"], markeredgecolor=color_code)
        # for element in ['medians']:
        #     plt.setp(plot_name[element], color='black')

        # for patch in plot_name['boxes']:
        #     patch.set(facecolor=color_code)

        # use plot function to draw a small line to name the legend.
        #plt.plot([], c=color_code, label=label)
        plt.plot([], label=label)
        plt.legend()


    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        results_df = get_results_df(results_folder, model_name)
        results_df = results_df[(results_df["k_L_grocery"]>0) & (results_df["k_L_restaurant"]>0) & (results_df["k_L_school"]>0)]
        results_df["m+n"]=results_df["num_res"]+results_df["num_parking"]

        # for nia in list(D_NIA.keys()):
        #     nia_df = results_df[results_df["nia_id"] == nia]
        #     if len(nia_df) < 9:
        #         print("missing for",nia)

        group_array=[]
        avg_time = results_df.groupby("nia_id")["solving_time"].apply(shifted_geo_mean)
        for p in range(len(group_thres)-1):

            group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p+1])]
            group_array.append(avg_time[group_df['nia_id'].unique()])

        #model_plot = plt.boxplot(group_array,positions=np.array(np.arange(len(group_array))) * 2.0 + i*0.35, widths=0.3,patch_artist=True)
        model_plot = plt.boxplot(group_array, positions=np.array(np.arange(len(group_array))) * group_gap + i * 0.2,widths=model_gap,medianprops = dict(linewidth=3.5))

        define_box_properties(model_plot, all_colors[i], display_name)

    # set the x label values
    #ticks = ['[0,200)', '[200,400)', '[400,600)]', '[[600,inf)]']
    ticks = ['1', '2', '3', '4']
    plt.xticks(np.arange(0, len(ticks) * group_gap, group_gap), ticks)


    #plt.xlabel("|M|+|N|")
    plt.xlabel("Group")
    plt.ylabel("Shifted Geo Mean (s)")
    #plt.yticks([])


    SMALL_SIZE = 8
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 18
    # if L_all_type_names[ind]=="grocery":
    #     BIGGER_SIZE = 25
    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rcParams["figure.figsize"] = (4, 4)
    #plt.set_size_inches(18.5, 18.5)

    #plt.setp(plt.gca(),ylim=(-5,))

    #plt.title(save_name)
    # Ensure the directory exists before saving the plot
    save_path = os.path.join(plot_folder,  save_name + ".pdf")
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches='tight')
    return


def quality_table_by_group_multiple(processed_folder):

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    num_nia = []
    num_ins = []

    L_MILP_no_depth = []
    L_CP_no_depth = []
    L_greedy_no_depth = []

    L_MILP_depth = []
    L_CP_depth = []
    L_greedy_depth = []

    group_thres = [0, 200, 400, 600, 2000]
    groups_name = ['[0,200)', '[200,400)', '[400,600)]', '[[600,inf)]']

    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        num_nia.append(len(group_df))
        num_ins.append(len(group_df)*9)

    # no depth

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    # depth of choice

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    output_D = { "|M|+|N|":groups_name,
                 "MILP":L_MILP_no_depth, "CP": L_CP_no_depth ,"Greedy":L_greedy_no_depth,
                 "MILP_d":L_MILP_depth, "CP_d": L_CP_depth ,"Greedy_d":L_greedy_depth}
    df_filename = os.path.join(processed_folder, "quality_by_group_multiple.csv")

    output_df = pd.DataFrame(output_D)

    output_df.loc[:, "MILP"] = output_df["MILP"].map('{:.4f}'.format)
    output_df.loc[:, "CP"] = output_df["CP"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy"] = output_df["Greedy"].map('{:.4f}'.format)
    output_df.loc[:, "MILP_d"] = output_df["MILP_d"].map('{:.4f}'.format)
    output_df.loc[:, "CP_d"] = output_df["CP_d"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy_d"] = output_df["Greedy_d"].map('{:.4f}'.format)



    output_df.to_csv(df_filename,index=False)

    return


def quality_table_by_k_multiple(processed_folder):

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    num_nia = []
    num_ins = []

    L_MILP_no_depth = []
    L_CP_no_depth = []
    L_greedy_no_depth = []

    L_MILP_depth = []
    L_CP_depth = []
    L_greedy_depth = []

    groups_name = list(range(1,10))


    # no depth

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for k in range(1,10):
        obj_group_df = all_obj_df[all_obj_df['k']==k]
        L_MILP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_no_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    # depth of choice

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for k in range(1,10):
        obj_group_df = all_obj_df[all_obj_df['k']==k]
        L_MILP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_depth.append(np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_depth.append(np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    output_D = { "k":groups_name,
                 "MILP":L_MILP_no_depth, "CP": L_CP_no_depth ,"Greedy":L_greedy_no_depth,
                 "MILP_d":L_MILP_depth, "CP_d": L_CP_depth ,"Greedy_d":L_greedy_depth}
    df_filename = os.path.join(processed_folder, "quality_by_k_multiple.csv")
    output_df = pd.DataFrame(output_D)
    output_df.to_csv(df_filename,index=False)

    return

def quality_table_by_group_single(processed_folder):

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    num_nia = []
    num_ins = []

    L_MILP_no_depth = []
    L_CP_no_depth = []
    L_greedy_no_depth = []

    L_MILP_depth = []
    L_CP_depth = []
    L_greedy_depth = []

    group_thres = [0, 200, 400, 600, 2000]
    groups_name = ['[0,200)', '[200,400)', '[400,600)]', '[[600,inf)]']

    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        num_nia.append(len(group_df))
        num_ins.append(len(group_df)*9)

    # no depth

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_no_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_no_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_no_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    # depth of choice

    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        L_MILP_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['mip']) / obj_group_df['best']))
        L_CP_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['cp']) / obj_group_df['best']))
        L_greedy_depth.append(100*np.mean((obj_group_df['best'] - obj_group_df['greedy']) / obj_group_df['best']))

    output_D = { "|M|+|N|":groups_name,
                 "MILP":L_MILP_no_depth, "CP": L_CP_no_depth ,"Greedy":L_greedy_no_depth,
                 "MILP_d":L_MILP_depth, "CP_d": L_CP_depth ,"Greedy_d":L_greedy_depth}
    df_filename = os.path.join(processed_folder, "quality_by_group_multiple.csv")
    output_df = pd.DataFrame(output_D)

    output_df.loc[:, "MILP"] = output_df["MILP"].map('{:.4f}'.format)
    output_df.loc[:, "CP"] = output_df["CP"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy"] = output_df["Greedy"].map('{:.4f}'.format)
    output_df.loc[:, "MILP_d"] = output_df["MILP_d"].map('{:.4f}'.format)
    output_df.loc[:, "CP_d"] = output_df["CP_d"].map('{:.4f}'.format)
    output_df.loc[:, "Greedy_d"] = output_df["Greedy_d"].map('{:.4f}'.format)

    output_df.to_csv(df_filename,index=False)

    return output_df


def opt_feas_multiple(results_folder, processed_folder):

    output_D = {}

    models = ["OptMultiple_False_0","OptMultipleCP_False_0","OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0"]
    display_names = ["MILP","CP","MILP,depth","CP,depth"]

    group_thres = [0,200,400,600,2000]

    for i in range(len(models)):

        model_name = models[i]
        display_name = display_names[i]

        output_D[display_name+" opt"]=[]
        output_D[display_name + " feas"] = []

        results_df = get_results_df(results_folder, model_name)
        results_df = results_df[(results_df["k_L_grocery"]>0) & (results_df["k_L_restaurant"]>0) & (results_df["k_L_school"]>0)]
        results_df["m+n"]=results_df["num_res"]+results_df["num_parking"]

        for p in range(len(group_thres)-1):

            group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p+1])]
            num_feas = len(group_df)
            if "CP" in model_name:
                num_opt = len(group_df[group_df["model_status"] == "Optimal"])
            else:
                num_opt = len(group_df[group_df["model_status"].astype(int) == 2])
            output_D[display_name + " opt"].append(num_opt)
            output_D[display_name + " feas"].append(num_feas)

    df_filename = os.path.join(processed_folder, "num_ins_by_group.csv")
    output_df = pd.DataFrame(output_D)
    output_df.to_csv(df_filename, index=False)

    return output_df

def quality_table(results_folder, processed_folder):
    INCLUDE_BP=False
    group_thres = [0, 200, 400, 600, 2000]
    groups_name = ['1', '2', '3', '4']
    if INCLUDE_BP:

        models_name_no_depth = ["mip","cp","greedy","mip-bp"]
        models_name_depth = ["mip", "cp", "greedy", "mip-bp","cp-bp"]
        models_save_name_no_depth = ["OptMultiple_False_0","OptMultipleCP_False_0","GreedyMultiple_False_0","OptMultiple_True_0"]
        models_save_name_depth = ["OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0","GreedyMultipleDepth_False_0","OptMultipleDepth_True_0", "OptMultipleDepthCP_True_0"]
    else:
        models_name_no_depth = ["mip", "cp", "greedy"]
        models_name_depth = ["mip", "cp", "greedy"]
        models_save_name_no_depth = ["OptMultiple_False_0", "OptMultipleCP_False_0", "GreedyMultiple_False_0"]
        models_save_name_depth = ["OptMultipleDepth_False_0", "OptMultipleDepthCP_False_0", "GreedyMultipleDepth_False_0"]

    num_nia = []
    num_ins = []

    L_group = []
    L_model = []

    L_no_depth_mre = []
    L_no_depth_opt = []
    L_no_depth_feas = []

    L_depth_mre = []
    L_depth_opt = []
    L_depth_feas = []

    summary_filename = os.path.join(processed_folder, "nia_summary.csv")
    summary_df = pd.read_csv(summary_filename)

    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        num_nia.append(len(group_df))
        num_ins.append(len(group_df) * 9)
    output_D = {"group": groups_name, "num_nia": num_nia, "num_ins":num_ins}
    output_df = pd.DataFrame(output_D)
    output_df.to_csv(os.path.join(processed_folder,"groups_summary.csv"), index=False)

    # no depth
    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]

        for m in range(len(models_name_no_depth)):
            # group column
            L_group.append(groups_name[p])
            if not INCLUDE_BP:
                # models column
                L_model.append(models_name_no_depth[m])
            # MRE column, no depth
            L_no_depth_mre.append(100 * np.mean((obj_group_df['best'] - obj_group_df[models_name_no_depth[m]]) / obj_group_df['best']))

            results_df = get_results_df(results_folder, models_save_name_no_depth[m])
            results_df = results_df[(results_df["k_L_grocery"] > 0) & (results_df["k_L_restaurant"] > 0) & (results_df["k_L_school"] > 0)]
            results_df["m+n"] = results_df["num_res"] + results_df["num_parking"]
            results_group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p + 1])]
            num_feas = len(results_group_df)
            if "CP" in models_save_name_no_depth[m]:
                num_opt = len(results_group_df[results_group_df["model_status"] == "Optimal"])
            elif "Greedy" in models_save_name_no_depth[m]:
                #num_opt = (obj_group_df['greedy']==obj_group_df['best']).sum()
                num_opt = "N/A"
            else:
                num_opt = len(results_group_df[results_group_df["model_status"].astype(int) == 2])

            # opt column, no depth
            L_no_depth_opt.append(num_opt)
            # feas column, no depth
            L_no_depth_feas.append(num_feas)

    # depth of choice
    all_obj_df = pd.read_csv(os.path.join(processed_folder, "multiple_depth.csv"), index_col=None, header=0)
    for p in range(len(group_thres) - 1):
        group_df = summary_df[(summary_df["m+n"] >= group_thres[p]) & (summary_df["m+n"] < group_thres[p + 1])]
        obj_group_df = all_obj_df[all_obj_df["nia"].isin(group_df['nia'])]
        for m in range(len(models_name_depth)):
            # MRE column, depth
            if INCLUDE_BP:
                L_model.append(models_name_depth[m])
            L_depth_mre.append(100 * np.mean((obj_group_df['best'] - obj_group_df[models_name_depth[m]]) / obj_group_df['best']))

            results_df = get_results_df(results_folder, models_save_name_depth[m])
            results_df = results_df[(results_df["k_L_grocery"] > 0) & (results_df["k_L_restaurant"] > 0) & (results_df["k_L_school"] > 0)]
            results_df["m+n"] = results_df["num_res"] + results_df["num_parking"]
            results_group_df = results_df[(results_df["m+n"] >= group_thres[p]) & (results_df["m+n"] < group_thres[p + 1])]
            num_feas = len(results_group_df)
            if "CP" in models_save_name_depth[m]:
                num_opt = len(results_group_df[results_group_df["model_status"] == "Optimal"])
            elif "Greedy" in models_save_name_depth[m]:
                #num_opt = (obj_group_df['greedy']==obj_group_df['best']).sum()
                num_opt = "N/A"
            else:
                num_opt = len(results_group_df[results_group_df["model_status"].astype(int) == 2])

            # opt column, depth
            L_depth_opt.append(num_opt)
            # feas column, depth
            L_depth_feas.append(num_feas)
    if INCLUDE_BP:

        output_D_temp = {"group": L_group, "method": L_model,
                    "MRE,d": L_depth_mre, "feas, d": L_depth_feas, "opt, d": L_depth_opt}
        output_df_temp = pd.DataFrame(output_D_temp)

    output_D = {"group": L_group,"method":L_model,
                "MRE,no d": L_no_depth_mre, "feas, no d": L_no_depth_feas, "opt, no d": L_no_depth_opt,
                "MRE,d": L_depth_mre, "feas, d": L_depth_feas, "opt, d": L_depth_opt}
    output_df = pd.DataFrame(output_D)

    # output format
    output_df["MRE,no d"] = output_df["MRE,no d"].map('{:.2f}'.format)
    output_df["MRE,d"] = output_df["MRE,d"].map('{:.2f}'.format)
    df_filename = os.path.join(processed_folder, "quality table.csv")
    output_df.to_csv(df_filename, index=False)
    print(output_df.to_latex(index=False))
    return

def hist_distances(data_root, results_folder,processed_folder,plot_folder):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 16
        # if L_all_type_names[ind]=="grocery":
        #     BIGGER_SIZE = 25

        # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
        # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
    D_NIA = ct_nia_mapping( os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # opt multiple depth
    print("multiple depth case")

    dist_grocery_before = []
    dist_res_1_before = []
    dist_res_2_before = []
    dist_school_before = []

    dist_grocery_after = {}
    dist_res_1_after = {}
    dist_res_2_after = {}
    dist_school_after = {}

    #k = 4
    # ORIG: all_k = [2,3]
    all_k = [3]
    use = 'mip'
    for k in all_k:
        dist_grocery_after[k] = []
        dist_res_1_after[k] = []
        dist_res_2_after[k] = []
        dist_school_after[k] = []
        for nia in list(D_NIA.keys()):
            print("nia:", nia)

            filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

            if use == 'mip':
                #  MIP
                if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename)):
                    mip_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepth_False_0", filename),index_col=None, header=0)
                    dist_grocery = mip_df["dist_grocery"]
                    choices_dist = [mip_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                    dist_school = mip_df["dist_school"]
                    dist_res_1 = list(mip_df["0_dist_restaurant"])
                    dist_res_2 = list(mip_df["1_dist_restaurant"])

                elif os.path.exists(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename)):
                    # if not feasible, use greedy solution 
                        greedy_df = pd.read_csv(
                            os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename),
                            index_col=None, header=0)
                        dist_grocery = greedy_df["dist_grocery"]
                        dist_school = greedy_df["dist_school"]
                        dist_res_1 = list(greedy_df["0_dist_restaurant"])
                        dist_res_2 = list(greedy_df["1_dist_restaurant"])
                else: 
                    print(f"File not found for NIA {nia}, skipping...")
                    continue

                dist_grocery_after[k] += list(dist_grocery)
                dist_school_after[k] += list(dist_school)
                dist_res_1_after[k] += list(dist_res_1)
                dist_res_2_after[k] += list(dist_res_2)

            else:
                # CP
                if os.path.exists(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename)):
                    cp_df = pd.read_csv(os.path.join(results_folder, "sol", "OptMultipleDepthCP_False_0", filename), index_col=None, header=0)
                    dist_grocery = cp_df["dist_grocery"]
                    choices_dist = [cp_df[str(c) + "_dist_restaurant"] if (str(c) + "_dist_restaurant") in mip_df.columns else L_a[-2] for c in range(10)]
                    dist_school = cp_df["dist_school"]

                else: 
                    print(f"File not found for NIA {nia}, skipping...")
                    continue

                dist_grocery_after[k] += list(dist_grocery)
                dist_school_after[k] += list(dist_school)
                dist_res_1_after[k] += list(cp_df["0_dist_restaurant"])
                dist_res_2_after[k] += list(cp_df["1_dist_restaurant"])

    k = 0


    for nia in list(D_NIA.keys()):
        print("nia:", nia)

        filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

        if os.path.exists(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename)):
            greedy_df = pd.read_csv(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename),
                                 index_col=None, header=0)
            dist_grocery = greedy_df["dist_grocery"]
            dist_school = greedy_df["dist_school"]
        else:
            print("????????")

        dist_grocery_before += list(dist_grocery)
        dist_school_before += list(dist_school)
        dist_res_1_before += list(greedy_df["0_dist_restaurant"])
        dist_res_2_before += list(greedy_df["1_dist_restaurant"])

    speed = 1.2
    transp = 0.5
    colors = ['blue','orange','green']
    colors_line = ['blue', 'red', 'green']

    bins = np.linspace(0.0, 70.0, 100)

    time_grocery_before = (np.array(dist_grocery_before) /speed) / 60
    time_res_1_before = (np.array(dist_res_1_before) / speed) / 60
    time_res_2_before = (np.array(dist_res_2_before) / speed) / 60
    time_school_before = (np.array(dist_school_before) / speed) / 60

    time_grocery_after = {}
    time_res_1_after = {}
    time_res_2_after = {}
    time_school_after = {}

    for k in all_k:
        time_grocery_after[k] = (np.array(dist_grocery_after[k]) / speed) / 60
        time_res_1_after[k] = (np.array(dist_res_1_after[k]) / speed) / 60
        time_res_2_after[k] = (np.array(dist_res_2_after[k]) / speed) / 60
        time_school_after[k] = (np.array(dist_school_after[k]) / speed) / 60

    L_all_before = [time_grocery_before, time_res_1_before, time_res_2_before, time_school_before]
    L_all_after = [[time_grocery_after[k] for k in all_k], [time_res_1_after[k] for k in all_k], [time_res_2_after[k] for k in all_k], [time_school_after[k] for k in all_k]]
    L_all_save_names = ["nearest grocery.png","nearest 1 res.png","nearest 2 res.png","nearest school.png"]
    L_all_type_names = ["grocery", "res1", "res2", "school"]


    for ind in range(len(L_all_type_names)):
        print(L_all_type_names[ind])
        plt.clf()
        plt.figure(dpi=300)
        arr_before = L_all_before[ind]
        arr_before = arr_before[~np.isin(arr_before, [2400/speed/60])]
        print("max value",L_all_type_names[ind],"before",np.max(arr_before))
        bins = np.linspace(0.0, int(np.max(arr_before)), 100)
        plt.hist(arr_before, bins, alpha=transp, label='k=0')

        plt.axvline(arr_before.mean(), color='blue', linestyle='solid', linewidth=1.5,label='Mean (k=0): {:.2f}'.format(arr_before.mean()))
        min_ylim, max_ylim = plt.ylim()
        #plt.text(arr_before.mean() * 1.1, max_ylim * 0.8, 'Mean: {:.2f}'.format(arr_before.mean()))
        plt.axvline(np.quantile(arr_before, 0.75), color='blue', linestyle='dashdot', linewidth=1.5, label='75% (k=0): {:.2f}'.format(np.quantile(arr_before, 0.75)))
        plt.axvline(np.max(arr_before), color='blue', linestyle=('dotted'), linewidth=1.5, label='Max: (k=0): {:.2f}'.format(np.max(arr_before)))
        #plt.text(np.quantile(arr_before, 0.75) * 1.1, max_ylim * 0.9, '75%: {:.2f}'.format(np.quantile(arr_before, 0.75)))

        for ind2 in range(len(all_k)):
            print(k)
            arr_after = L_all_after[ind][ind2]
            arr_after=arr_after[~np.isin(arr_after, [2400/speed/60])]
            print("max value", L_all_type_names[ind], k, np.max(arr_after))

            plt.hist(arr_after, bins,alpha=transp, label='k={}'.format(all_k[ind2]))
            plt.axvline(arr_after.mean(), color=colors_line[ind2+1], linestyle='solid', linewidth=1.5,label='Mean (k={}): {:.2f}'.format(all_k[ind2],arr_after.mean()))
            #plt.text(arr_after.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(arr_after.mean()))

            plt.axvline(np.quantile(arr_after, 0.75), color=colors_line[ind2 + 1], linestyle='dashdot', linewidth=1.5, label='75% (k={}): {:.2f}'.format(all_k[ind2],np.quantile(arr_after, 0.75)))
            plt.axvline(np.max(arr_after), color=colors_line[ind2 + 1], linestyle=('dotted'), linewidth=1.5, label='Max (k={}): {:.2f}'.format(all_k[ind2], np.max(arr_after)))
            #plt.text(np.quantile(arr_after, 0.75) * 1.1, max_ylim * 0.9, '75%: {:.2f}'.format(np.quantile(arr_after, 0.75)))
        plt.legend(loc='upper right')
        plt.xlabel("Walking Time (Minutes)")
        #plt.ylabel("Frequency")

        # Ensure the directory exists before saving the plot
        save_path = os.path.join(plot_folder,"final_eval", L_all_save_names[ind])
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
    return

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

def nia_avg_walking_time(data_root, plot_folder, results_folder, preprocessing_folder):
    SIZE = 7
    plt.figure(dpi=300)

    '''Plot code reference: https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb'''

    nia_shape = get_nias(data_root)

    # Ensure geometries are in a projected CRS before computing centroids
    #if nia_shape.crs is None or nia_shape.crs.to_epsg() != 3857:
    #    nia_shape = nia_shape.to_crs(epsg=3857)  # Convert to a projected CRS

    nia_shape["center"] = nia_shape["geometry"].centroid  # Now safe to compute centroids
    nia_points = nia_shape.copy()
    nia_points.set_geometry("center", inplace=True)

    speed = 1.2

    dist_grocery = []
    dist_res1 = []
    dist_res2 = []
    dist_school = []
    walk_obj = []

    for nia in [int(item) for item in nia_shape["area_sh11"]]:
        
        print("nia:", nia)
        k = 0
        filename = f"assignment_NIA_{nia}_{k},{k},{k}.csv"
        greedy_path = os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename)
        summary_path = os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", f"NIA_{nia}_{k},{k},{k}.csv")

        if os.path.exists(greedy_path) and os.path.exists(summary_path):
            greedy_df = pd.read_csv(greedy_path, index_col=None, header=0)
            greedy_df_result = pd.read_csv(summary_path, index_col=None, header=0)
        else:
            print("???????? File not found:", greedy_path, "or", summary_path)
            greedy_df = pd.DataFrame()  # Assign an empty DataFrame to avoid UnboundLocalError
            greedy_df_result = pd.DataFrame()

        # Check if the required columns exist before accessing them
        if not greedy_df.empty and "dist_grocery" in greedy_df.columns:
            dist_grocery.append(np.mean(greedy_df["dist_grocery"]))
            dist_res1.append(np.mean(greedy_df["0_dist_restaurant"]))
            dist_res2.append(np.mean(greedy_df["1_dist_restaurant"]))
            dist_school.append(np.mean(greedy_df["dist_school"]))
        else:
            print(f"Warning: Missing required columns in {greedy_path}")
            dist_grocery.append(np.nan)
            dist_res1.append(np.nan)
            dist_res2.append(np.nan)
            dist_school.append(np.nan)

        if not greedy_df_result.empty and "obj" in greedy_df_result.columns:
            walk_obj.append(np.mean(greedy_df_result["obj"]))
        else:
            print(f"Warning: Missing 'obj' column in {summary_path}")
            walk_obj.append(np.nan)

    nia_shape["dist_grocery"] = (np.array(dist_grocery) / speed) / 60
    nia_shape["dist_res1"] = (np.array(dist_res1) / speed) / 60
    nia_shape["dist_res2"] = (np.array(dist_res2) / speed) / 60
    nia_shape["dist_school"] = (np.array(dist_school) / speed) / 60
    nia_shape["walk_obj"] = np.array(walk_obj)

    ax = nia_shape.plot(column="walk_obj", legend=True, legend_kwds={"shrink": 0.5}, cmap="OrRd")
    texts = []
    for x, y, label, id in zip(nia_points.geometry.x, nia_points.geometry.y, nia_points["area_na13"], nia_points["area_sh11"]):
        texts.append(plt.text(x - 0.01, y + 0.01, int(id), fontsize=SIZE, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none")))

    plt.rc("font", size=SIZE)
    plt.rc("axes", titlesize=SIZE)
    plt.rc("axes", labelsize=SIZE)
    plt.rc("xtick", labelsize=SIZE)
    plt.rc("ytick", labelsize=SIZE)
    plt.rc("legend", fontsize=SIZE)
    plt.tight_layout()

    # Ensure the directory exists before saving the plot
    save_dir = os.path.join(plot_folder, "final_eval")
    ensure_directory_exists(save_dir) 

    plt.savefig(os.path.join(save_dir, "cur_score.pdf"))
    plt.savefig(os.path.join(save_dir, "cur_score.png"))

    return


def nia_avg_walking_time_2(data_root,plot_folder,results_folder,preprocessing_folder):
    
    SIZE=7
    plt.figure(dpi=300)

    '''plot code reference: https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb'''

    nia_shape = get_nias(data_root)

    # reading pednet file
    # pednet_path = os.path.join(data_root, "pednet.zip")
    # pednet = gpd.read_file(pednet_path)
    #nia_shape["geometry"] = nia_shape["geometry"].to_crs(epsg=3857)  # NEW ADDED Replace EPSG code with your target projected CRS
    nia_shape["center"] = nia_shape["geometry"].centroid
    nia_points = nia_shape.copy()
    nia_points.set_geometry("center", inplace=True)

    for k in [0,3]:

        walk_obj = []

        for nia in [int(item) for item in nia_shape["area_sh11"]]:

            ###############
            print("nia:", nia)
            filename = "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)
            
            obj = np.nan  # ✅ Set a default value to prevent UnboundLocalError
            
            #  MIP
            if os.path.exists(os.path.join(results_folder, "summary", "OptMultipleDepth_False_0", filename)):
                mip_df = pd.read_csv(os.path.join(results_folder, "summary", "OptMultipleDepth_False_0", filename),index_col=None, header=0)
                obj=mip_df["obj"]
            else:
                # if not feasible, use greedy solution
                if os.path.exists(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", filename)):
                    greedy_df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", filename), index_col=None, header=0)
                    obj=greedy_df["obj"]

            walk_obj.append(obj)
            ###############
            
            # ADDED Debug fixing inhomogenous shapes
            cleaned_walk_obj = []
            for w in walk_obj:
                if isinstance(w, pd.Series):
                    cleaned_walk_obj.append(w.iloc[0])  # extract the float value from the Series
                else:
                    cleaned_walk_obj.append(w)

        nia_shape["walk_obj_" + str(k)] = np.array(cleaned_walk_obj)

    nia_shape["delta"] = nia_shape["walk_obj_3"]-nia_shape["walk_obj_0"]

    # ALTERED ax=nia_shape.plot(column='delta',legend=True, legend_kwds={'shrink': 0.5},cmap='viridis', edgecolor='white',linewidth=0.5)
    vmin, vmax = 10, 55
    ax = nia_shape.plot(
        column='delta',
        cmap='viridis',
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={'shrink': 0.5},
        edgecolor='white',
        linewidth=0.5
    )
    
    # ALTERED cbar = ax.get_figure().get_axes()[1] cbar.yaxis.set_major_locator(ticker.MultipleLocator(10))
    # Colorbar tick formatting
    cbar = ax.get_figure().get_axes()[1]
    cbar.yaxis.set_major_locator(ticker.MultipleLocator(10))
    cbar.set_ylim(vmin, vmax)
    
    
    texts = []
    for x, y, label, id in zip(nia_points.geometry.x, nia_points.geometry.y, nia_points["area_na13"],nia_points["area_sh11"]):
        # can instead plot id too?

        texts.append(plt.text(x - 0.01, y+0.01, int(id), fontsize=SIZE, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
        # names
        #texts.append(plt.text(x-0.01, y, label[:-5], fontsize=7, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')))
    #plt.show()
    #ax.yaxis.set_ticks(np.arange(43.6, 43.8, 0.25))
    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)  # legend fontsize
    plt.tight_layout()
    
    # Ensure the directory exists before saving the plot
    save_dir = os.path.join(plot_folder, "final_eval")
    ensure_directory_exists(save_dir)  
    plt.savefig(os.path.join(save_dir, "score_delta_k3.pdf"))
    plt.savefig(os.path.join(save_dir, "score_delta_k3.png"))
    
    return

def avg_obj_vs_k_multi(results_folder, plot_folder):

    SMALL_SIZE = 8
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 22

    data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
    D_NIA = ct_nia_mapping( os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "processed_TSNS 2020 NIA Census Tracts.xlsx"))

    # dist obj
    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.figure(dpi=300)

    L_k = []
    L_grocery = []
    L_res_0 = []
    L_res_1 = []
    L_res_2 = []
    L_school = []
    L_obj = []
    for k in range(10):
        L_grocery_temp = []
        L_res_0_temp = []
        L_res_1_temp = []
        L_res_2_temp = []
        L_school_temp = []
        L_obj_temp = []
        for nia in list(D_NIA.keys()):


            filename = "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)


            if (os.path.exists(os.path.join(results_folder, "summary", "OptMultipleDepth_False_0", filename))):
                df = pd.read_csv(os.path.join(results_folder, "summary", "OptMultipleDepth_False_0", filename), index_col=None, header=0)
                if np.isnan(df["dist_obj_L_grocery"][0]):
                    pass
                else:
                    L_grocery_temp.append(df["dist_obj_L_grocery"])
                L_res_0_temp.append([L[0] for L in [json.loads(x) for x in df["dist_obj_L_restaurant"]]])
                L_res_1_temp.append([L[1] for L in [json.loads(x) for x in df["dist_obj_L_restaurant"]]])
                L_res_2_temp.append([L[1] for L in [json.loads(x) for x in df["dist_obj_L_restaurant"]]])
                L_school_temp.append(df["dist_obj_L_school"])
                L_obj_temp.append(df["obj"])
            elif (k>0) and (not os.path.exists(os.path.join(results_folder, "summary", "OptMultipleDepth_False_0", filename))):
                #df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0","NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)), index_col=None, header=0)
                file_path = os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0",
                         "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k))

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=None, header=0)
                    # your processing here
                    L_grocery_temp.append(df["dist_obj_L_grocery"])
                    L_res_0_temp.append([L[0] for L in [json.loads(x) for x in df["dist_obj_L_restaurant"]]])
                    L_res_1_temp.append([L[1] for L in [json.loads(x) for x in df["dist_obj_L_restaurant"]]])
                    L_res_1_temp.append([L[2] for L in [json.loads(x) for x in df["dist_obj_L_restaurant"]]])
                    L_school_temp.append(df["dist_obj_L_school"])
                    L_obj_temp.append(df["obj"])
                else:
                    print(f"Skipping missing file: {file_path}")
            else:
                print("????????")
        L_grocery.append(np.mean(L_grocery_temp))
        L_res_0.append(np.mean(L_res_0_temp))
        L_res_1.append(np.mean(L_res_1_temp))
        L_res_2.append(np.mean(L_res_2_temp))
        L_school.append(np.mean(L_school_temp))
        L_obj.append(np.mean(L_obj_temp))
        L_k.append(k)

    plt.rcParams.update({'font.size': BIGGER_SIZE})

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('k')
    ax1.set_ylabel('Distance (Meters)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax1.plot(L_k,L_grocery , label="Grocery",marker="^")
    ax1.plot(L_k,L_res_0 , label="Restaurant (1st)",marker="^")
    ax1.plot(L_k, L_school, label="School",marker="^")
    ax1.plot(L_k, L_res_1, label="Restaurant (2nd)",marker="^")
    #ax1.plot(L_k, L_res_2, label="res3", marker="^")

    ax2 = ax1.twinx()
    ax2.set_ylabel('Score', color='purple')
    ax2.set_ylim([50, 95])
    ax2.yaxis.set_major_locator(MultipleLocator(10)) 
    ax2.plot(L_k, L_obj, color='purple', label="score",linestyle="dashdot",marker="^")
    ax2.tick_params(axis='y', labelcolor='purple')


    ax1.legend(prop={'size': BIGGER_SIZE})
    ax2.legend(prop={'size': BIGGER_SIZE},loc='upper left')
    #plt.xlabel("k")
    #plt.ylabel("dist (m)")

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.tight_layout()
    
    # Ensure the directory exists before saving the plot
    quality_folder = os.path.join(plot_folder, "quality")
    ensure_directory_exists(quality_folder)
    # plt.savefig(save_path, "quality/avg_dist_vs_k_depth.pdf")
    save_path = os.path.join(plot_folder, "quality", "avg_dist_vs_k_depth.pdf")
    plt.savefig(save_path)


    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.figure(dpi=300)

    L_k = []
    L_grocery = []
    L_res = []
    L_school = []
    L_obj = []
    for k in range(10):

        L_grocery_temp = []
        L_res_temp = []
        L_school_temp = []
        L_obj_temp = []
        for nia in list(D_NIA.keys()):
            filename = "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)

            if (os.path.exists(os.path.join(results_folder, "summary", "OptMultiple_False_0", filename))):
                df = pd.read_csv(os.path.join(results_folder, "summary", "OptMultiple_False_0", filename), index_col=None,header=0)
                if np.isnan(df["dist_obj_L_grocery"][0]):
                    pass
                else:
                    L_grocery_temp.append(df["dist_obj_L_grocery"])
                L_res_temp.append(df["dist_obj_L_restaurant"])
                L_school_temp.append(df["dist_obj_L_school"])
                L_obj_temp.append(df["obj"])
            elif (k > 0) and (not os.path.exists(os.path.join(results_folder, "summary", "OptMultiple_False_0", filename))):
                #df = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultiple_False_0", "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)),index_col=None, header=0)
                file_path = os.path.join(results_folder, "summary", "GreedyMultiple_False_0",
                                        "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k))

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=None, header=0)
                    # your processing here
                    L_grocery_temp.append(df["dist_obj_L_grocery"])
                    L_res_temp.append(df["dist_obj_L_restaurant"])
                    L_school_temp.append(df["dist_obj_L_school"])
                    L_obj_temp.append(df["obj"])
                else:
                    print(f"Skipping missing file: {file_path}")
                
        L_k.append(k)
        L_grocery.append(np.mean(L_grocery_temp))
        L_res.append(np.mean(L_res_temp))
        L_school.append(np.mean(L_school_temp))
        L_obj.append(np.mean(L_obj_temp))

    fig, ax1 = plt.subplots()
    plt.rcParams.update({'font.size': BIGGER_SIZE})
    ax1.set_xlabel('k')
    ax1.set_ylabel('Distance (Meters)', color='black')
    ax1.yaxis.set_major_locator(MultipleLocator(100)) 
    ax1.tick_params(axis='y', labelcolor='black')

    ax1.plot(L_k, L_grocery, label="Grocery",marker="^")
    ax1.plot(L_k, L_res, label="Restaurant",marker="^")
    ax1.plot(L_k, L_school, label="School",marker="^")

    ax2 = ax1.twinx()
    ax2.set_ylabel('score', color='purple')
    ax2.set_ylim([50, 95])
    ax2.yaxis.set_major_locator(MultipleLocator(10)) 
    ax2.plot(L_k, L_obj, color='purple', label="score", linestyle="dashdot",marker="^")
    ax2.tick_params(axis='y', labelcolor='purple')

    ax1.legend(prop={'size': BIGGER_SIZE})
    ax2.legend(prop={'size': BIGGER_SIZE},loc='upper left')

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize

    plt.tight_layout()
    # Ensure the directory exists before saving the plot
    save_dir = os.path.join(plot_folder, "quality") 
    ensure_directory_exists(save_dir)  

    plt.savefig(os.path.join(save_dir, "avg_dist_vs_k_no_depth.pdf"))  
    
    
    


    # # score obj
    #
    # plt.clf()
    # plt.figure(figsize=(15, 10))
    #
    # for nia in list(D_NIA.keys()):
    #     nia_df = results_df[results_df["nia_id"] == nia]
    #     if len(nia_df)<10:
    #         if len(nia_df[nia_df['k']==0])>0:
    #             print("missing values?")
    #             print(model_name,amenity_name,nia,len(nia_df))
    #     nia_df = nia_df.sort_values(by=['k'])
    #     k_list = nia_df["k"]
    #     score_list = nia_df["obj"]
    #
    #     plt.plot(k_list, score_list, '--o', label=D_NIA[nia]['name'])
    #     plt.legend(prop={'size': 6})
    #     plt.xlabel("k")
    #     plt.ylabel("score")
    # plt.savefig(os.path.join(plot_folder, amenity_name + "_"+ display_name + "_k_vs_score.png"))

    return



if __name__ == "__main__":

    results_folder = "results"
    plot_folder = "results_plot"
    data_root =  "/Users/annve/Downloads/AAAI23-WalkabilityOptimization"
    processed_folder= "processed_results"
    preprocessing_folder = "./preprocessing"
    
    nia_avg_walking_time(data_root,plot_folder,results_folder,preprocessing_folder)
    nia_avg_walking_time_2(data_root, plot_folder, results_folder, preprocessing_folder)
    hist_distances(data_root, results_folder,processed_folder,plot_folder)
    avg_obj_vs_k_multi(results_folder, plot_folder)
    #plot_time_by_group_multiple(results_folder, plot_folder, models, display_names, save_name)

