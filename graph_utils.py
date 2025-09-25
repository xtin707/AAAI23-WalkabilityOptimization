import pandas as pd
import geopandas as gpd
import pandas as pd
import os
import shapely
import numpy as np
import networkx as nx
import pandana as pdna
from shapely.ops import nearest_points
import h5py
from shapely.geometry import Polygon, LineString, Point, box
from sqlalchemy import *
from shapely.geometry import *
import json

preprocessing_folder="preprocessing_pickle4"

def load_pednet(data_root):
    pednet = gpd.read_file(os.path.join(data_root,"pednet.zip")).to_crs("EPSG:2019")
    print(pednet.crs)
   
    pednet = pednet[
        ['OBJECTID', 'road_type', 'sdwlk_code', 'sdwlk_desc', 'crosswalk', 'cwalk_type', 'px', 'px_type', 'geometry']]
    return pednet

def create_graph(gdf, precision=3):

    '''
    Modified from publicly available PedNet data:
    City of Toronto. 2019a. Toronto Walkability Project. https:
    //github.com/gcc-dav-official-github/dav cot walkability.
    Accessed: 2022-07-17.

    Create a networkx given a GeoDataFrame of lines. Every line will
    correspond to two directional graph edges, one forward, one reverse. The
    original line row and direction will be stored in each edge. Every node
    will be where endpoints meet (determined by being very close together) and
    will store a clockwise ordering of incoming edges.
    '''

    G = nx.Graph()

    def make_node(coord, precision):
        return tuple(np.round(coord, precision))

    # Edges are stored as (from, to, data), where from and to are nodes.
    def add_edges(row, G):
        geometry = row.geometry
        coords = list(geometry.coords)
        
        start = make_node(coords[0], precision)
        end = make_node(coords[-1], precision)
        
        # Add forward edge
        fwd_attr = {k: v for k, v in row.items()}
        fwd_attr['forward'] = 1
        
        # fwd_attr['geometry']=  geometry
        fwd_attr['length'] = geometry.length
        fwd_attr['visited'] = 0

        G.add_edge(start, end, **fwd_attr)

    gdf.apply(add_edges, axis=1, args=[G])

    return G

def get_pandana_net(G,save_path):
    ''' convert a network graph to pandana graph'''

    if not os.path.exists(save_path):
        all_nodes = list(G.nodes)
        all_edges_dist = nx.get_edge_attributes(G, 'length')
        from_list = [all_nodes.index(node1) for (node1, node2) in list(all_edges_dist.keys())]
        to_list = [all_nodes.index(node2) for (node1, node2) in list(all_edges_dist.keys())]
        nodes_x = [x for (x,y) in all_nodes]
        nodes_y = [y for (x, y) in all_nodes]

        transit_ped_net = pdna.Network(nodes_x, nodes_y, from_list,
                     to_list,
                     pd.DataFrame(list(all_edges_dist.values())),
                     twoway=True)
    else:
        transit_ped_net = pdna.Network.from_hdf5(save_path)

    return transit_ped_net

def pednet_CTs(pednet,CTs,mapping=os.path.join(preprocessing_folder,'pednet_points/road_CT_mapping.txt')):
    with open(mapping, 'r') as f:
        D = json.load(f)

    df_road=pd.DataFrame.from_dict(D)
    df_road=df_road[df_road["CTNAME"].isin(CTs)]
    pednet_ct = pednet[pednet['OBJECTID'].isin(list(df_road["roadID"].values))]

    return pednet_ct.reset_index()

def pednet_NIA(pednet,nia,preprocessing_folder):
    mapping=os.path.join(preprocessing_folder,"road_nia_mapping.txt")
    with open(mapping, 'r') as f:
        D = json.load(f)
    df_road=pd.DataFrame.from_dict(D)
    df_road = df_road[df_road["niaID"]==nia]
    pednet_nia = pednet[pednet['OBJECTID'].isin(list(df_road["roadID"].values))]

    return pednet_nia.reset_index()

def nodes_census(pednet,ct,mapping=os.path.join(preprocessing_folder,'pednet_points/road_CT_mapping.txt')):
    with open(mapping, 'r') as f:
        D = json.load(f)
    CTs = D['CTNAME']
    x_p = D['x_p']
    y_p = D['y_p']
    roadID = D['roadID']

    roads_ct = []
    nodes_ct = []
    for i in range(len(CTs)):
        if CTs[i] == ct:
            roads_ct.append(roadID[i])
            nodes_ct.append(Point(x_p[i], y_p[i]))
    # simplification: take one end of the road as nodes
    return nodes_ct[::2]

def nodes_from_pandana_net(transit_ped_net):
    nodes_df = transit_ped_net.nodes_df
    gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.x, nodes_df.y), crs="EPSG:2019")
    return gdf

def nearest_panana_net(item, nodes):
    pts=nodes.geometry.unary_union

    if isinstance(item, shapely.geometry.polygon.Polygon):
        point = item.centroid
    elif isinstance(item, shapely.geometry.point.Point):
        point = item
    else:
        print("Unkown origin type !!!")
        return "unknown"

    return np.where(nodes.geometry == nearest_points(point, pts)[1])[0][0]

def get_SP(transit_ped_net,save_path):
    '''
    return a matrix with pre-computed SPs
    '''

    if not os.path.exists(save_path):
        print("starting computing SP")
        gdf = nodes_from_pandana_net(transit_ped_net)
        num_nodes = len(gdf)
        mat = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                mat[i,j]=transit_ped_net.shortest_path_length(i, j)
        print("finish computing SP")
        np.savetxt(save_path, mat)
    else:
        mat=np.loadtxt(save_path)
    return mat


if __name__ == "__main__":
    # creating network graph
    pednet = gpd.read_file(file_path = "C:\\Users\\annve\\Downloads\\Walkability For All\\pednet.zip").to_crs("EPSG:2019")
    print("PedNet CRS:", pednet.crs)
   


