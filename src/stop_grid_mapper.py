import pandas as pd
import geopandas as gpd
import numpy as np

import folium
import branca.colormap as cm

from .grid_partitioning import Grid
from .stop_explorer import StopExplorer


class StopGridMapper:

    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self, grid : Grid, stop_explorer : StopExplorer):
        """
        Initialize a StopGridMapper instance by mapping the stop segments of 'stop_explorer' to the cells of 'grid'.
        """

        self.grid = grid

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


    def compute_statistics_cells_users(self) :
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
        
        self.agg_cell_uid = self.join.groupby(['uid', 'cell_id']).agg(**stats_pairs_cell_uid)


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
        self.agg_cell_uid['frac_time_weekend'] = weekend_analysis

        return self.agg_cell_uid
    

    def compute_statistics_cells(self) :
        '''
        Compute several statistics for each cell of the grid .
        '''

        # Compute several stats of the grid's cells.
        stats_config = {'num_stops' : pd.NamedAgg(column='uid', aggfunc='size'),
                        'num_users' : pd.NamedAgg(column='uid', aggfunc='nunique'),
                        'mean_duration_mins' : pd.NamedAgg(column='duration_mins', aggfunc='mean'),
                        'median_duration_mins' : pd.NamedAgg(column='duration_mins', aggfunc='median')} 
        stats_cells_grid = self.join.groupby('cell_id').agg(**stats_config)

        # Create a GeoDataframe that represents the original GeoDataFrame of the grid, augmented with the statistics.
        self.augmented_grid = self.grid.get_grid().join(stats_cells_grid, how='left').fillna(0)
        return self.augmented_grid
    

    def generate_augmented_grid_heatmap(self, target_column : str, desc_target_column : str, dic_fields_tooltip : dict) -> folium.Map :
        '''
        Generate a heatmap of the grid, where each cell is colored according to the values in 'value_col'.
        '''

        gdf = self.augmented_grid


        # Instantiate a Folium Map.
        minx, miny, maxx, maxy = gdf.total_bounds
        m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], 
                       zoom_start=14, tiles="CartoDB positron", prefer_canvas = True)


        # Define a colormap for the heatmap
        vmin = float(gdf[target_column].min())
        vmax = float(gdf[target_column].max())
        if vmin == vmax:  # avoid degenerate scale
            vmin, vmax = (0.0, vmin if vmax > 0 else 1.0)

        cmap = cm.LinearColormap(
            colors=["white", "red"],
            vmin=vmin, vmax=vmax
        )
        cmap.caption = desc_target_column
        cmap.add_to(m)


        # Define the grid layer. Each grid's cell is filled according to the colormap defined below AND the # of users it contains.
        folium.GeoJson(
            data=gdf.to_json(),
            style_function=lambda f: {
                "fillColor": cmap(f["properties"][target_column]), # polygon fill color
                "color": "black", # polygon border color
                "weight": 0.5,    # polygon border color's weight
                "fillOpacity": 0.7, # polygon fill color's opacity
            },
            tooltip=folium.GeoJsonTooltip(
                fields = list(dic_fields_tooltip.keys()),
                aliases = list(dic_fields_tooltip.values()),
                localize=True
            ),
            name=f"Grid heatmap (by # of {target_column})"
        ).add_to(m)

        # Add a layer control to the map (to be able to turn on/off the grid layer).
        folium.LayerControl().add_to(m)

        # Instruct the map to tightly fit the render's zoom on the dataset bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(30, 30))
        
        return m