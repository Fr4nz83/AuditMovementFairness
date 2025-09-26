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


        # Enrich the original dataframe by extracting some temporal information.
        self._initial_enrichment()



    ### PROTECTED METHODS ###

    def _initial_enrichment(self) :
        # Enrich the original dataframe by extracting some temporal information.
        self.gdf_stops['hour_start'] = self.gdf_stops['datetime'].dt.hour.astype(np.uint8)
        self.gdf_stops['hour_end'] = self.gdf_stops['leaving_datetime'].dt.hour.astype(np.uint8)
        self.gdf_stops['weekday'] = self.gdf_stops['datetime'].dt.weekday.astype(np.uint8)



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

    def get_stops_temporal_intervals_freqs(self) -> pd.DataFrame  :
        ''' 
        Return a dataframe containing the frequency of all the (hour_start, hour_stop) intervals that
        have been found associated with the stop segments.
        '''
        
        return self.gdf_stops[['hour_start','hour_end']].value_counts(normalize = True)
        