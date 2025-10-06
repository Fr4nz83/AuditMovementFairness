import pandas as pd
import geopandas as gpd
import numpy as np


class StopExplorer() :

    ### CLASS CONSTRUCTOR ###
    
    def __init__(self, stop_df_path : str) :
        
        # Read the dataframe containing the stops detected by scikit-mobility
        self.gdf_stops = pd.read_parquet(stop_df_path)
        self.gdf_stops = gpd.GeoDataFrame(self.gdf_stops, 
                                          geometry=gpd.points_from_xy(self.gdf_stops.lng, self.gdf_stops.lat), 
                                          crs="EPSG:4326")
        del self.gdf_stops['lng'], self.gdf_stops['lat']



    ### PROTECTED METHODS ###



    ### PUBLIC METHODS ###

    def get_df_stops(self) -> pd.DataFrame :
        ''' 
        Return a reference to the dataframe containing the stops detected by scikit-mobility.
        '''
        return self.gdf_stops
    
    def get_df_stops_users(self, user_id : int) -> pd.DataFrame  :
        '''
        Returns a dataframe containing the stops associated with a specific user ID.
        '''

        return self.gdf_stops.loc[self.gdf_stops['uid'] == user_id]
        