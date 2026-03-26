    SIZE=7
    plt.figure(dpi=300)

    '''plot code reference: https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb'''

    nia_shape = get_nias(data_root)

    # reading pednet file
    # pednet_path = os.path.join(data_root, "pednet.zip")
    # pednet = gpd.read_file(pednet_path)

    nia_shape["center"] = nia_shape["geometry"].centroid
    nia_points = nia_shape.copy()
    nia_points.set_geometry("center", inplace=True)

    speed = 1.2
    dist_grocery=[]
    dist_res1=[]
    dist_res2=[]
    dist_school=[]
    walk_obj=[]

    for nia in [int(item) for item in nia_shape["area_s_cd"]]:

        #pednet_nia = pednet_NIA(pednet, nia, preprocessing_folder)

        print("nia:", nia)
        k=0
        filename = "assignment_NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)
        if os.path.exists(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename)):
            greedy_df = pd.read_csv(os.path.join(results_folder, "sol", "GreedyMultipleDepth_False_0", filename),
                                    index_col=None, header=0)
            greedy_df_result = pd.read_csv(os.path.join(results_folder, "summary", "GreedyMultipleDepth_False_0", "NIA_%s_%s,%s,%s.csv" % (nia, k, k, k)),
                                    index_col=None, header=0)
        else:
            print("????????")
        dist_grocery.append(np.mean(greedy_df["dist_grocery"]))
        dist_res1.append(np.mean(greedy_df["0_dist_restaurant"]))
        dist_res2.append(np.mean(greedy_df["1_dist_restaurant"]))
        dist_school.append(np.mean(greedy_df["dist_school"]))
        walk_obj.append(greedy_df_result["obj"])

    nia_shape["dist_grocery"] = (np.array(dist_grocery)/speed)/60
    nia_shape["dist_res1"] = (np.array(dist_res1)/speed)/60
    nia_shape["dist_res2"] = (np.array(dist_res2)/speed)/60
    nia_shape["dist_school"] = (np.array(dist_school)/speed)/60
    nia_shape["walk_obj"] = np.array(walk_obj)

    ax=nia_shape.plot(column='walk_obj',legend=True, legend_kwds={'shrink': 0.5}, cmap='OrRd')
    texts = []
    for x, y, label, id in zip(nia_points.geometry.x, nia_points.geometry.y, nia_points["area_name"],nia_points["area_s_cd"]):
        # can instead plot id too?
        texts.append(plt.text(x - 0.01, y+0.01, int(id), fontsize=SIZE,bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
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
    plt.savefig(os.path.join(plot_folder,"final_eval", "cur_score.pdf"))
    plt.savefig(os.path.join(plot_folder, "final_eval", "cur_score.png"))
    return