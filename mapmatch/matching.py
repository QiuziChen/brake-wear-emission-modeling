'''
@File    :   matching.py
@Time    :   2023/10/12 19:37:22
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Define a class `matcher` that contains map-matching framework.
'''


import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from mapmatch.pytrackEngine import pytrackMatch
from pytrack.graph import graph, distance, utils


ENGINE_DICT = {
    'pytrack': pytrackMatch
}


class Matcher():
    """
    Map matching object that contains tools for map-matching.
    """
    def __init__(
            self,
            engine='pytrack'
    ):
        """
        road_graph: [NetworkX.MultiDiGraph]
        engine: mapmatching engine.
        """
        self.matchEngine = ENGINE_DICT[engine]
    
    def match(
            self,
            traj:pd.DataFrame,
            lonCol='lon',
            latCol='lat',
            tripIDCol='tripID',
            dropCoord=True,
            **kwargs
    ):
        """
        Perfrom map-matching and grasp road information.
        ==========
        lonCol: column name of longitude
        latCol: column name of latitude
        tripIDCol: column name of tripID to perform map matching for each trip seperately. If None, perform interpolation for the whole traj.df.
        dropCoord: True if no need for keeping mapping coordinates.
        """
        traj = traj.copy()
        if dropCoord:
            traj.loc[:, 'osmid'] = None
        else:
            traj.loc[:, 'mapLon'] = None
            traj.loc[:, 'mapLat'] = None
            traj.loc[:, 'osmid'] = None
        
        # generate road graph info
        print("Graph extracting...")
        north, south, west, east = traj[latCol].max(), traj[latCol].min(), traj[lonCol].min(), traj[lonCol].max()
        north, south, west, east = distance.enlarge_bbox(north, south, west, east, 500)

        # custom filter
        custom_filter = ('["highway"]["area"!~"yes"]["access"!~"private"]'
                 '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|'
                 'elevator|escalator|footway|path|pedestrian|planned|platform|proposed|raceway|steps|track"]'
                 '["service"!~"emergency_access|private"]')

        road_graph = graph.graph_from_bbox(north, south, west, east, simplify=True, custom_filter=custom_filter)
        node_gdf, edge_gdf = utils.graph_to_gdfs(road_graph)
        node_gdf = gpd.GeoDataFrame(node_gdf, geometry="geometry")
        edge_gdf = gpd.GeoDataFrame(edge_gdf, geometry="geometry")

        # perform map-matching for each trip seperately
        if tripIDCol:  
            
            for id in tqdm(traj[tripIDCol].unique(), desc="Map-matching"):  
                # select trip from traj.df
                trip = traj[traj[tripIDCol] == id].copy()

                try:
                    info_dict = self.matchEngine(
                        lon=trip[lonCol],
                        lat=trip[latCol],
                        road_graph=road_graph,
                        edge_gdf=edge_gdf,
                        node_gdf=node_gdf,
                        **kwargs
                    )
                    
                    # update df
                    traj.loc[trip.index, 'osmid'] = info_dict['edge_osmid']

                    if not dropCoord:
                        traj.loc[trip.index, 'mapLon'] = info_dict['lon']
                        traj.loc[trip.index, 'mapLat'] = info_dict['lat']
                except:
                    pass
        
        # perform map-matching for the whole traj.df
        else:  
            # road info
            info_dict = self.matchEngine(
                lon=traj[lonCol],
                lat=traj[latCol],
                road_graph=road_graph,
                edge_gdf=edge_gdf,
                node_gdf=node_gdf,
                **kwargs
            )
            
            # update df
            traj.loc[:, 'osmid'] = info_dict['edge_osmid']

            if not dropCoord:
                traj.loc[:, 'mapLon'] = info_dict['lon']
                traj.loc[:, 'mapLat'] = info_dict['lat']
        
        return traj