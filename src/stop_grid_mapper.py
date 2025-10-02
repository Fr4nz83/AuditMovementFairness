import pandas as pd
import geopandas as gpd
import numpy as np
from .grid_partitioning import Grid
from .stop_explorer import StopExplorer


class StopGridMapper:

    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, grid : Grid, stop_explorer : StopExplorer):
        """
        Initialize a StopGridMapper instance by mapping the stop segments of 'stop_explorer' to the cells of 'grid'.
        """

        # Compute a spatial join between 'other_geo_df' and the cells of the grid.
        self.join = gpd.sjoin(stop_explorer.get_df_stops(), grid.get_grid(), how='inner', predicate='intersects')
        
        # Rename and set the type of the column containing the index of the grid's cells.
        self.join.rename(columns={'index_right':'cell_id'}, inplace=True)
        self.join['cell_id'] = self.join['cell_id'].astype(np.uint32)
        self.join['duration_mins'] = self.join['duration_secs'] / 60



    ### PUBLIC METHODS ###

    def get_join(self) -> gpd.GeoDataFrame :
        '''
        Return a reference to the GeoDataFrame containing the result of the spatial join between the stops and the grid cells.
        '''
        return self.join


    def compute_statistics_cells_users(self) -> pd.DataFrame :
        '''
        Compute several statistics for each pair (user_id, cell_id).
        '''

        # Here we build the dataframe that will contain the mapping between users
        # and grid cells, with several statistics associated with them.
        stats_pairs_cell_uid = {'num_stops' : pd.NamedAgg(column='uid', aggfunc='size'),
                                'mean_duration_mins' : pd.NamedAgg(column='duration_mins', aggfunc='mean'),
                                'median_duration_mins' : pd.NamedAgg(column='duration_mins', aggfunc='median'),
                                'mean_hour_start' : pd.NamedAgg(column='hour_start', aggfunc='mean'),
                                'median_hour_start' : pd.NamedAgg(column='hour_start', aggfunc='median'),
                                'mean_hour_end' : pd.NamedAgg(column='hour_end', aggfunc='mean'),
                                'median_hour_end' : pd.NamedAgg(column='hour_end', aggfunc='median')}
        
        agg_cell_uid = self.join.groupby(['uid', 'cell_id']).agg(**stats_pairs_cell_uid)


        # For each pair (user_id, cell_id), compute the fraction of stops associated with the weekday and the weekend.
        weekend_analysis = self.join.groupby(['uid', 'cell_id'])['weekend'].value_counts(normalize=True)

        # Now, some pairs (user_id, cell_id) might not have the fraction concerning the weekday or the weekend (this
        # happens when all the stops happened during the weekend or during the weekday). We thus want to reindex and simplify
        # the series, so that it simply contains the fraction of time a pair (user_id, cell_id) spent during the weekend.
        pairs = weekend_analysis.index.droplevel('weekend').unique() # 1 - Get all the unique "(user_id, cell_id)" pairs.
        weekend_idx = pd.MultiIndex.from_tuples([(u, c, 1) for u, c in pairs],      # 2 - Create a new multiindex with the "weekend" level 
                                                names=weekend_analysis.index.names) #     set to 1. This will be used to reindex the series.                                                                 
        weekend_analysis = weekend_analysis.reindex(weekend_idx).fillna(0) # 3 - Reindex the original series. This will drop the pairs that
                                                                        #     do not have the "weekend" level equal to 1. We also fill 
                                                                        #     the NaN values, which are those pairs that did not occur
                                                                        #    during the weekend, with 0s. 
        weekend_analysis.index = weekend_analysis.index.droplevel('weekend') # 4 - Drop the "weekend" level, as it is not needed anymore.
        # display(weekend_analysis)

        # Update 'agg_cell_uid' with the final information.
        agg_cell_uid['frac_time_weekend'] = weekend_analysis

        return agg_cell_uid
    

    def compute_statistics_cells(self) -> pd.DataFrame :
        '''
        Compute several statistics for each cell of the grid .
        '''

        stats_config = {'num_stops' : pd.NamedAgg(column='uid', aggfunc='size'),
                        'num_users' : pd.NamedAgg(column='uid', aggfunc='nunique'),
                        'mean_duration_mins' : pd.NamedAgg(column='duration_mins', aggfunc='mean'),
                        'median_duration_mins' : pd.NamedAgg(column='duration_mins', aggfunc='median')}
        
        return self.join.groupby('cell_id').agg(**stats_config)