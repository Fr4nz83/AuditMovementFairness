import pandas as pd
import geopandas as gpd
import numpy as np
import folium

from shapely.geometry import box
from pyproj import Transformer


class Grid() :

    ### CLASS CONSTRUCTOR ###

    def __init__(self, grid_cell_length_meters : float) :

        self.cell_length_meters = grid_cell_length_meters



    ### PROTECTED METHODS ###

    def _project_bounds_from_to_crs(self, orig_crs : str, dest_crs : str, bbox : tuple[float,float,float,float]) -> tuple[float,float,float,float]:
        '''
        Given a bounding box 'bbox' projected in the 'orig_crs' coordinate reference system, reproject it to the 'dest_crs' 
        CRS.

        Parameters
        ----------
        orig_crs : str
            The original CRS of the bounding box.
        dest_crs : str
            The destination CRS to which the bounding box must be reprojected.
        bbox : tuple[float,float,float,float]
            The bounding box to be reprojected, in the form (minx, miny, maxx, maxy).

        Returns
        ------- 
        tuple[float,float,float,float]
            The reprojected bounding box, in the form (minx, miny, maxx, maxy).
        '''

        # Reproject the bounding box's coords from the original CRS to a metric CRS.
        tr = Transformer.from_crs(orig_crs, dest_crs, always_xy=True)
        return tr.transform_bounds(*bbox)



    ### PUBLIC METHODS ###

    def compute_grid_for_geodata(self, geo_df : gpd.GeoDataFrame) -> gpd.GeoDataFrame :
        
        # Compute the bounding box of the objects in the geodataframe
        bbox = geo_df.total_bounds  # return a tuple having the form '(minx, miny, maxx, maxy)'
        self.orig_crs = geo_df.crs # Get the original CRS of the geodataframe.
        metric_crs = geo_df.estimate_utm_crs() # Estimate a metric CRS we can use for the area covered by the geodataframe.

        # Reproject the bounding box's coords from the original CRS to a metric CRS.
        bbox_metric = self._project_bounds_from_to_crs(self.orig_crs, metric_crs, bbox)

        # Compute a uniform grid with side 'self.cell_length_meters' over the bounding box.
        self.grid = self.compute_grid_meters(bbox_metric, self.cell_length_meters).set_crs(crs=metric_crs)

        # Reproject the grid to the original CRS.
        self.grid = self.grid.to_crs(crs=self.orig_crs)

        return self.grid
    

    def get_grid(self) -> gpd.GeoDataFrame :
        '''
        Return a reference to the GeoDataFrame containing the grid cells.
        '''
        return self.grid


    def compute_grid_meters(self, bbox : tuple[float,float,float,float], step : float) -> gpd.GeoDataFrame :
        
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
        
        list_cells = []
        for curr_lon in np.arange(min_lon, max_lon, step):
            next_lon = min(curr_lon + step, max_lon)
            for curr_lat in np.arange(min_lat, max_lat, step):
                next_lat = min(curr_lat + step, max_lat)
                
                # build shapely box
                geom = box(curr_lon, curr_lat, next_lon, next_lat)
                # record bounds + geometry
                list_cells.append({"geometry": geom})

        # Create a final GeoDataFrame holding the grid cells.
        return gpd.GeoDataFrame.from_dict(list_cells)
    

    def compute_grid_step(self, bbox : tuple[float,float,float,float], divisions : int, crs : str) -> gpd.GeoDataFrame :
        
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
        
        step_lon = (max_lon - min_lon) / divisions
        step_lat = (max_lat - min_lat) / divisions
        list_cells = []
        for curr_lon in range(min_lon, max_lon, step_lon):
            next_lon = curr_lon + step_lon
            for curr_lat in range(min):
                next_lat = curr_lat + step_lat
                
                # build shapely box
                geom = box(curr_lon, curr_lat, next_lon, next_lat)

                # record bounds + geometry
                list_cells.append({"geometry": geom})

        # Create a final GeoDataFrame holding the grid cells.
        return gpd.GeoDataFrame.from_dict(list_cells).set_crs(crs=crs)
    

    def generate_grid_map(self):

        # Create a Folium Map object.
        bbox = self.grid.total_bounds
        m = folium.Map(location=[(bbox[3]+bbox[1])/2, (bbox[2]+bbox[0])/2,], 
                       zoom_start=13, control_scale=True, prefer_canvas=True)
        
        # Plot the grid's cells' polygons on the map.
        folium.GeoJson(self.grid).add_to(m)


        # 5. Display or save
        return m
        # m.save("trajectory_time_slider.html")