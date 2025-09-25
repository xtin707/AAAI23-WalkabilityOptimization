import json
import math
import pandas as pd
import geopandas as gpd
import pandas as pd
import osmnx as ox
import shapely
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box
from sqlalchemy import *
from shapely.geometry import *
import os



def get_nias(data_root):
    file_path = os.path.join(data_root,"Neighbourhood Improvement Areas - 4326", "Neighbourhood Improvement Areas - 4326.shp")
    nia = gpd.read_file(file_path).to_crs("EPSG:2019")
    nia.columns = map(str.lower, nia.columns)
    nia = nia[['area_id2', 'area_sh11', 'area_na13', 'geometry']]
    print("nias file crs is: " + str(nia.crs))
    return nia


def get_CTs_boundary(CTs, use, file_path="data/lct_000b16a_e/lct_000b16a_e.shp"):
    boundary = gpd.read_file(file_path).to_crs("EPSG:2019")
    print("CT file CRS is: "+ str(boundary.crs))
    boundary = boundary[boundary["CMANAME"] == "Toronto"]
    
    exclude = ['0806.00', '0400.22', '0803.04', '0803.03', '0801.02', '0412.02', '0576.40', '0531.02', '0528.41',
               '0528.35', '0510.00', '0530.02', '0531.02', '0525.02', '0527.04', '0500.02']
    if CTs is not None:
        boundary = boundary[boundary["CTNAME"].isin(CTs)]
    boundary = boundary[~boundary["CTNAME"].isin(exclude)]

    return boundary

def get_NIAs_boundary(nia, use, data_root):
    boundary = gpd.read_file(os.path.join(data_root, "Neighbourhood Improvement Areas - 4326", "Neighbourhood Improvement Areas - 4326.shp")).to_crs("EPSG:2019")

    print("NIA boundary CRS is: "+ str(boundary.crs))
   
    boundary.columns = map(str.lower, boundary.columns)
    boundary["area_sh11"] = boundary["area_sh11"].apply(lambda x: int(x))

    boundary = boundary[boundary["area_sh11"] == nia]

    return boundary

def query_ox(polygons,tags):
    frames=[]
    for polygon in polygons:
        # Convert polygon to WGS84 for OSM query
        poly_wgs84 = gpd.GeoSeries([polygon], crs="EPSG:2019").to_crs("EPSG:4326").iloc[0]
        result = ox.features.features_from_polygon(poly_wgs84, tags)
        # Always reproject results back to EPSG:2019
        result = result.to_crs("EPSG:2019")
        frames.append(result) #['unique_id', 'osmid', 'element_type', 'building', 'geometry']
    result = pd.concat(frames,ignore_index=True)

    return pd.concat(frames, ignore_index=True)

def centroid(item, value):

    if isinstance(item,shapely.geometry.polygon.Polygon):
        x = item.centroid.x
        y = item.centroid.y
    elif isinstance(item,shapely.geometry.point.Point):
        x = item.x
        y = item.y
    else:
        x = item.centroid.x
        y = item.centroid.y
    if value=="x":
        return x
    if value == "y":
        return y

def nia_filename(nias):
    NIAs=sorted(nias)
    NIAs_name = ''
    for i in range(len(NIAs)):
        NIAs_name+=str(NIAs[i])
        if i != (len(NIAs)-1):
            NIAs_name += "_"
    return NIAs_name

def road_points(root,outputfile="./preprocessing/road_end_points.txt",prec=2):

    # original copy in test.py
    # pednet_path="zip://data/pednet.zip"
    pednet_path = os.path.join(root,"pednet.zip")

    print("Checking pednet path:", pednet_path)
    print("File exists?", os.path.exists(pednet_path))


    # reading pednet file
    pednet = gpd.read_file(pednet_path).to_crs("EPSG:2019")
    print(pednet.crs)
    
    d = {'roadID': [], 'x': [], 'y': []}

    for i in range(len(pednet)):
        if i % 500 == 0:
            print("processing", i)
        x_list, y_list = pednet.iloc[i]["geometry"].coords.xy
        
        # get the two end points of each road segment
        d['x'].append(np.round(x_list[0], prec))
        d['y'].append(np.round(y_list[0], prec))
        d['roadID'].append(i + 1)
        d['x'].append(np.round(x_list[-1], prec))
        d['y'].append(np.round(y_list[-1], prec))
        d['roadID'].append(i + 1)

    with open(outputfile, 'w') as file:
        file.write(json.dumps(d))
    return

def road_nia_mapping(data_root, preprocessing_folder, outputfile):
    # code reference: https://github.com/gcc-dav-official-github/dav_cot_walkability/blob/master/code/TTC%20Walkability%20Tutorial.ipynb
    nia = gpd.read_file(os.path.join(data_root,"Neighbourhood Improvement Areas - 4326.zip")).to_crs("EPSG:2019")
    print("Available columns in nia:", list(nia.columns))
    nia.columns = map(str.lower, nia.columns)
    nia = nia[['area_id2', 'area_sh11', 'area_na13', 'geometry']]

    
    nia_d = {'niaID': [], 'x_p': [], 'y_p': [], 'roadID': []}

    with open(os.path.join(preprocessing_folder,"road_end_point.txt"), 'r') as f:
        D = json.load(f)
    roadID = D['roadID']
    x_list = D['x']
    y_list = D['y']


    # assign road segments (end points) to census tract
    for row in range(len(nia)):
        nia_id = int(nia.iloc[row]["area_sh11"]) # nia name
        poly = nia.iloc[row]['geometry']  # nia_boundary

        for j in range(len(x_list)):
            p = Point(x_list[j], y_list[j])
            if p.within(poly):
                nia_d['niaID'].append(nia_id)
                nia_d['x_p'].append(x_list[j])
                nia_d['y_p'].append(y_list[j])
                nia_d['roadID'].append(roadID[j])

    with open(outputfile, 'w') as file:
        file.write(json.dumps(nia_d))
    return


def road_CT_mapping(CT_boundary_path="data/lct_000b16a_e/lct_000b16a_e.shp",
                    end_points_path="./preprocessing/pednet_points/end_points.txt",
                    save_path='./preprocessing/pednet_points/road_CT_mapping.txt'):
    boundary = gpd.read_file(CT_boundary_path).to_crs("EPSG:2019")
    print(boundary.crs)
    boundary = boundary[boundary["CMANAME"] == "Toronto"]

    ct_d = {'CTNAME': [], 'x_p': [], 'y_p': [], 'roadID': []}

    with open(end_points_path, 'r') as f:
        D = json.load(f)
    roadID = D['roadID']
    x_list = D['x']
    y_list = D['y']

    all_CTs = list(boundary['CTNAME'].unique())

    # CMAUID=535 for all of these

    # assign road segments (end points) to census tract
    for row in range(len(boundary)):
        name = boundary.iloc[row]['CTNAME']  # census tract name
        poly = boundary.iloc[row]['geometry']  # census tract boundary
        print(row, name)
        for j in range(len(x_list)):
            p = Point(x_list[j], y_list[j])
            if p.within(poly) == True:
                ct_d['CTNAME'].append(name)
                ct_d['x_p'].append(x_list[j])
                ct_d['y_p'].append(y_list[j])
                ct_d['roadID'].append(roadID[j])

    with open(save_path, 'w') as file:
        file.write(json.dumps(ct_d))


def ct_nia_mapping(nia_path):
    df = pd.read_excel(nia_path)
    D={}
    for i in range(len(df)):
        id = int(df.iloc[i][0])
        ct = str(df.iloc[i][2])
        if not id in D.keys():
            D[id]={"name":df.iloc[i][1],"CTs":[ct[-6:-2]+'.'+ct[-2:]]}
        else:
            D[id]["CTs"].append(ct[-6:-2]+'.'+ct[-2:])
    return D

def map_back_allocate(allocations,df_to):
    allocated_nodes = [df_to.iloc[j]["node_ids"] for j in allocations]
    allocated_df = df_to.iloc[allocations]
    return allocated_nodes, allocated_df

def map_back_assign(assignments, df_from, df_to, dict):
    # assignments
    i_s = []
    j_s = []
    d_s = []
    i_id = []
    j_id = []
    a_id = []
    for (i, j, a) in assignments:
        i_s.append(i)
        j_s.append(j)
        d_s.append(dict[(i, j)])
        i_id.append(df_from.iloc[i]["node_ids"])
        j_id.append(df_to.iloc[j]["node_ids"])
        a_id.append(a)
    assign_D = {
        "i": i_s,
        "j": j_s,
        "i_id": i_id,
        "j_id": j_id,
        "a_id": a_id,
        "d_s": d_s,
    }
    return assign_D

if __name__ == "__main__":
    
    data_root = r"C:\Users\annve\Downloads\Walkability For All" #NEW ALTERED





