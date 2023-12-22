'''
@File    :   pytrackEngine.py
@Time    :   2023/10/04 17:22:10
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   A toolkit of map matching based on PyTrack.
'''



import copy
import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree
from filecontrol import blockprint

from pytrack.graph import graph, distance, utils
from pytrack.matching import mpmatching_utils, mpmatching


def mapMatch(
        lon,
        lat,
        road_graph,
        node_gdf,
        enlarge_dist=300,
        interp_dist=30,
        radius=30,
):
    """
    Perfrom map-matching.
    lon, lat: longitude and latitude, either list or ndarray.
    road_graph: [NetworkX.MultiDiGraph]
    node_gdf: GeoDataFrame of nodes
    enlarge_dist: distance for bounding box enlarging, default = 100 meters.
    interp_dist: interpolate dist between two adjacent nodes. The smaller the interp_dist, the greater the precision and the longer the computational time.
    radius: radius of the candidate search circle.
    return: results [dict]
    """
    points = [(lat_, lon_) for lat_, lon_ in zip(lat, lon)]

    # Create BBOX
    north, east = np.max(np.array([*points]), 0)
    south, west = np.min(np.array([*points]), 0)
    
    # Enlarge bounding box
    north, south, west, east = distance.enlarge_bbox(north, south, west, east, enlarge_dist)

    # Generate G
    # if road_graph & node_gdf:
    # if road graph is given, generate subgraph from the enlarged bounding box
    G = road_graph.subgraph(node_gdf.clip(mask=[west, south, east, north])['osmid'].to_list())
    
    # elif road_graph:
    #     # if no nodes gdf, generate one
    #     node_gdf = utils.graph_to_gdfs(road_graph, nodes=True, edges=False)
    #     G = road_graph.subgraph(node_gdf.clip(mask=[west, south, east, north]))
    
    # else:
    #     # extract road graph from osm
    #     G = graph.graph_from_bbox(north, south, west, east, simplify=True, network_type='drive')

    # Generate candidates
    G_interp, candidates, no_cands = get_candidates(G, points, interp_dist=interp_dist, closest=True, radius=radius)

    # Extract trellis DAG graph
    trellis = mpmatching_utils.create_trellis(candidates)
    
    # Perform the map-matching process
    path_prob, predecessor = mpmatching.viterbi_search(G_interp, trellis, "start", "target")

    # Get matching results
    match_results = elab_candidate_results(candidates, predecessor, no_cands)

    return match_results

def infoGrasp(
        results:dict,
        edge_gdf=None,
        road_info=False,
):
    """
    Grasp information from map-matching results.
    results: map-matching results, [dict].
    edge_gdf: road network information, DataFrame or GeoDataFrame.
    road_info: True if road information is needed.
    """
    # if no roadnet files are given, don't extract road info
    if edge_gdf is None:
        road_info = False

    if road_info:
        info = {
            'edge_osmid': [],
            'road_type': [],
            'bridge': [],
            'tunnel': [],
        }
    else:
        info = {
            'edge_osmid': [],
        }

    for result in results.values():

        if result['candidates']:  # has candidates

            # get candidate id
            id = np.argwhere(result['candidate_type'] == True)[0][0]

            # edge osmid
            edge_osmid = result['edge_osmid'][id]  # edge_osmid
            info['edge_osmid'].append(edge_osmid)

            # # matched coordinates
            # info['lat'].append(result['candidates'][id][0])
            # info['lon'].append(result['candidates'][id][1])

            if road_info:
                road = edge_gdf[edge_gdf['osmid'] == edge_osmid]
                if road.shape[0] == 0:
                    road_type, bridge, tunnel = np.nan, np.nan, np.nan 
                else:
                    road = road.iloc[0]
                    road_type, bridge, tunnel = road['highway'], road['bridge'], road['tunnel'],
                
                info['road_type'].append(road_type)
                info['bridge'].append(bridge)
                info['bridge'].append(tunnel)

        else:  # no candidates

            info['edge_osmid'].append(np.nan)
            # info['lat'].append(result['observation'][0])
            # info['lon'].append(result['observation'][1])

            if road_info:
                info['road_type'].append(np.nan)
                info['bridge'].append(np.nan)
                info['bridge'].append(np.nan)

    return info

@blockprint
def pytrackMatch(
        lon,
        lat,
        road_graph=None,
        node_gdf=None,
        edge_gdf=None,
        enlarge_dist=300,
        interp_dist=30,
        radius=30,
        road_info=False,
):
    """
    Perform map-matching and generate info.
    lon, lat: longitude and latitude, either list or ndarry.
    road_graph: [NetworkX.MultiDiGraph]
    node_gdf: GeoDataFrame of nodes
    edge_gdf: road network information, DataFrame or GeoDataFrame.
    enlarge_dist: distance for bounding box enlarging, default = 100 meters.
    interp_dist: interpolate dist between two adjacent nodes. The smaller the interp_dist, the greater the precision and the longer the computational time.
    radius: radius of the candidate search circle.
    road_info: True if road information is needed.
    return: results [dict]
    """

    # perform map-matching
    results = mapMatch(
        lon, lat,
        road_graph,
        node_gdf,
        enlarge_dist,
        interp_dist,
        radius
    )

    # generate matched info
    info_dict = infoGrasp(
        results,
        edge_gdf,
        road_info
    )

    return info_dict


"""
Modified PyTrack
"""


def get_candidates(G, points, interp_dist=30, closest=True, radius=30):
    """ 
    [Modified function from PyTrack to avoid ``no candidates error``]
    Extract candidate points for Hidden-Markov Model map-matching approach.

    Parameters
    ----------
    G: networkx.MultiDiGraph
        Street network graph.
    points: list
        The actual GPS points.
    interp_dist: float, optional, default: 1
        Step to interpolate the graph. The smaller the interp_dist, the greater the precision and the longer
        the computational time.
    closest: bool, optional, default: True
        If true, only the closest point is considered for each edge.
    radius: float, optional, default: 10
        Radius of the search circle.
    Returns
    -------
    G: networkx.MultiDiGraph
        Street network graph.
    results: dict
        Results to be used for the map-matching algorithm.
    no_cands_dict: dict
        Points that have no candidates.
    """

    G = G.copy()

    # if interp_dist:
    #     if G.graph["geometry"]:
    #         G = distance.interpolate_graph(G, dist=interp_dist)
    #     else:
    #         _ = utils.graph_to_gdfs(G, nodes=False)
    #         G = distance.interpolate_graph(G, dist=interp_dist)

    _ = utils.graph_to_gdfs(G, nodes=False)
    G = distance.interpolate_graph(G, dist=interp_dist)

    geoms = utils.graph_to_gdfs(G, nodes=False).set_index(["u", "v"])[["osmid", "geometry"]]

    uv_xy = [[u, osmid, np.deg2rad(xy)] for uv, geom, osmid in
             zip(geoms.index, geoms.geometry.values, geoms.osmid.values)
             for u, xy in zip(uv, geom.coords[:])]

    index, osmid, xy = zip(*uv_xy)

    nodes = pd.DataFrame(xy, index=osmid, columns=["x", "y"])[["y", "x"]]

    ball = BallTree(nodes, metric='haversine')

    idxs, dists = ball.query_radius(np.deg2rad(points), radius / distance.EARTH_RADIUS_M, return_distance=True)
    dists = dists * distance.EARTH_RADIUS_M  # radians to meters

    if closest:
        results = dict()
        for i, (point, idx, dist) in enumerate(zip(points, idxs, dists)):
            df = pd.DataFrame({"osmid": list(np.array(index)[idx]),
                               "edge_osmid": list(nodes.index[idx]),
                               "coords": tuple(map(tuple, np.rad2deg(nodes.values[idx]))),
                               "dist": dist})

            df = df.loc[df.groupby('edge_osmid')['dist'].idxmin()].reset_index(drop=True)

            results[i] = {"observation": point,
                          "osmid": list(df["osmid"]),
                          "edge_osmid": list(df["edge_osmid"]),
                          "candidates": list(df["coords"]),
                          "candidate_type": np.full(len(df["coords"]), False),
                          "dists": list(df["dist"])}

    else:
        results = {i: {"observation": point,
                       "osmid": list(np.array(index)[idx]),
                       "edge_osmid": list(nodes.index[idx]),
                       "candidates": list(map(tuple, np.rad2deg(nodes.values[idx]))),
                       "candidate_type": np.full(len(nodes.index[idx]), False),
                       "dists": list(dist)} for i, (point, idx, dist) in enumerate(zip(points, idxs, dists))}

    no_cands = [node_id for node_id, cand in results.items() if not cand["candidates"]]
    no_cands_dict = {point: results[point] for point in no_cands}

    if no_cands:
        for cand in no_cands:
            del results[cand]
        # print(f"A total of {len(no_cands)} points has no candidates: {*no_cands,}")
    return G, results, no_cands_dict

def elab_candidate_results(results, predecessor, no_cands_dict):
    """ 
    [Modified function from PyTrack to avoid ``no candidate error``]
    Elaborate results of ``candidate.get_candidates`` method. It selects which candidate best matches the actual
    GPS points.

    Parameters
    ----------
    results: dict
        Output of ``candidate.get_candidates`` method.
    predecessor: dict
        Output of ``mpmatching.viterbi_search`` method.
    no_cands_dict: dict
        Points that have no candidates.
    Returns
    -------
    results: dict
        Elaborated results.
    """
    results = copy.deepcopy(results)
    for key_can, key_pre in zip(list(results.keys()), reversed(list(predecessor.keys()))):
        win_cand_idx = int(predecessor[key_pre].split("_")[1])
        results[key_can]["candidate_type"][win_cand_idx] = True
    
    # for points without candidates, add nan value
    results.update(no_cands_dict)

    return results
