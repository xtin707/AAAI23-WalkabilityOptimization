from graph_utils import *
from map_utils import *

from pathlib import Path
import argparse
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt


# def of tags: https://wiki.openstreetmap.org/wiki/Map_features#Others
#POI['geometry'] some are points, some are polygon


def pednet_preprocessing():
    '''
    PedNet preprocessing: get the corresponding NIA for edges in PedNet
    '''
    # get end points of roads
    road_points(data_root,outputfile=os.path.join(preprocessing_folder,"road_end_point.txt"))

    # road and nia mapping
    road_nia_mapping(data_root, preprocessing_folder, os.path.join(preprocessing_folder,"road_nia_mapping.txt"))
    return

def nia_preprocessing(nia_id):
    
    '''
    Data needed for each NIA instance
    '''
    print("processing nia", nia_id)
    tag_parking={"amenity":"parking"}
    tag_residential={"building":["apartments","bungalow","cabin","detached","dormitory","farm","ger","hotel","house","houseboat","residential","semidetached_house","static_caravan","terrace"]}
    tag_school={"amenity":"school"}
    tag_restaurant={"amenity":"restaurant"}
    
    tag_grocery= {"shop":["supermarket", "convenience"]}
    tag_department_store = {
    "shop": ["mall", "department_store"],
    "amenity": "marketplace",
    "landuse": "retail"}
    tag_cafe= {"amenity":"cafe", "cuisine":"coffee_shop"}
    pednet = load_pednet(data_root)


    #D_NIA = ct_nia_mapping(os.path.join(data_root,"neighbourhood-improvement-areas-wgs84/processed_TSNS 2020 NIA Census Tracts.xlsx"))

    prec=2
    pednet_nia = pednet_NIA(pednet,nia_id,preprocessing_folder)
    print(f"Input data (pednet_nia): {pednet_nia}")

    # networkx graph
    G_raw = create_graph(pednet_nia, precision=prec)
    
    subgraphs = list(G_raw.subgraph(c) for c in nx.connected_components(G_raw))
    # get the largest connect component
    sugbraph_index = [len(g.nodes) for g in subgraphs].index(max([len(g.nodes) for g in subgraphs]))
    G=subgraphs[sugbraph_index]

    # shortest paths
    SP_mat=nx.floyd_warshall_numpy(G,weight='length')
    SP_filename = "NIA_%s_prec_%s.txt" % (nia_id, prec)
    np.savetxt(os.path.join(sp_save_path, SP_filename), SP_mat)

    # pandana net
    net_filename="NIA_%s_prec_%s.hd5" % (nia_id, prec)
    transit_ped_net = get_pandana_net(G,graph_save_path)
    transit_ped_net.save_hdf5(os.path.join(net_save_path, net_filename))


    residentials_df = query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values, tag_residential)
    department_store_df=query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values,tag_department_store)
    parking_df=query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values,tag_parking)
    grocery_df=query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values,tag_grocery)
    school_df=query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values,tag_school)
    cafe_df = query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values, tag_cafe)
    restaurant_df = query_ox(get_NIAs_boundary(nia_id, "ox", data_root).geometry.values, tag_restaurant)


    all_dfs=[residentials_df, department_store_df, parking_df, grocery_df, school_df, cafe_df, restaurant_df]
    all_strs = ['residential', 'department_store', 'parking', 'grocery', 'school', 'cafe','restaurant']
    colors = ['g','lightcoral','grey','red','yellow','brown','orange']

    ax = pednet_nia.plot(figsize=(15, 15), color='blue', markersize=1)

    for i in range(len(all_dfs)):
        df = all_dfs[i]

        if len(df)>0:
            # add node id to each amenity location
            df['x']=df['geometry'].apply(centroid,args=('x',))
            df['y']=df['geometry'].apply(centroid,args=('y',))
            x, y = df.x, df.y
            df["node_ids"] = transit_ped_net.get_node_ids(x, y)

            # plot
            df.plot(ax=ax, color=colors[i], markersize=80)

        df_filename = "NIA_%s_%s.pkl" % (nia_id, all_strs[i])
        df.to_pickle(os.path.join(df_save_path, df_filename))


    plt.savefig(os.path.join(data_visual_save_path,"nia_%s.png" % (nia_id)))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument("cc", help="run on compute canada?",
                        type=int)
    parser.add_argument("--nia", help="proprocess data for nia",
                        type=int)

    args = parser.parse_args()

    if args.cc == 1:
        data_root = "/home/huangw98/projects/def-khalile2/huangw98/walkability_data"
    else:
        data_root = "C:\\Users\\annve\\Downloads\\AAAI23-WalkabilityOptimization"
    

    preprocessing_folder = "./preprocessing"
    Path(preprocessing_folder).mkdir(parents=True, exist_ok=True)
    net_save_path = os.path.join(preprocessing_folder, 'saved_nets')
    graph_save_path = os.path.join(preprocessing_folder, 'networx_graph')
    df_save_path = os.path.join(preprocessing_folder, 'saved_dfs')
    sp_save_path = os.path.join(preprocessing_folder, 'saved_SPs')
    data_visual_save_path = os.path.join(preprocessing_folder, 'saved_visual')
    Path(net_save_path).mkdir(parents=True, exist_ok=True)
    Path(df_save_path).mkdir(parents=True, exist_ok=True)
    Path(sp_save_path).mkdir(parents=True, exist_ok=True)
    Path(data_visual_save_path).mkdir(parents=True, exist_ok=True)

    if args.nia:
        nia_preprocessing(args.nia)
    else:
        pednet_preprocessing()



