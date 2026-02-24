from matplotlib.pyplot import step
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import pickle
import math

import shapely
from pyproj import Transformer


class Grid() :

    ### CLASS CONSTRUCTOR ###

    def __init__(self,
                 bbox : tuple[float,float,float,float],
                 orig_crs : str,
                 metric_crs : str, 
                 grid_cell_length_meters : float,
                 offset : float = 0.0) :
        '''
        Create a grid over 'bbox'.

        Parameters
        ----------
        bbox : tuple[float,float,float,float]
            The bounding box over which the grid must be computed, in the form (minx, miny, maxx, maxy).
        orig_crs : str
            The original CRS of the bounding box.
        metric_crs : str
            A metric CRS (i.e., with coordinates in meters) to which the bounding box will be reprojected before computing the grid.
        grid_cell_length_meters : float
            The length of the side of the grid's cells, in meters.
        offset : float, optional
            offset is a bbox-relative phase shift of the internal grid lines. It does not shift the bbox itself.
            The value must be in meters and fall in [0, grid_cell_length_meters).
        '''


        # Check bbox, step and offset correctness.
        min_x, min_y, max_x, max_y = map(float, bbox)
        if not (max_x > min_x and max_y > min_y):
            raise ValueError("The bounding box does not satisfy 'max_x > min_x' or 'max_y > min_y'")
        if grid_cell_length_meters <= 0:
            raise ValueError("grid_cell_length_meters must be > 0")
        if not (0 <= offset < grid_cell_length_meters):
            raise ValueError("offset must be in [0, grid_cell_length_meters)")
        

        # Compute the grid over the bounding box 'bbox'.
        self.cell_length_meters = grid_cell_length_meters
        self.offset = offset
        self.grid = self._compute_grid_over_bbox(bbox, orig_crs, metric_crs)



    ### PROTECTED METHODS ###

    def _project_bounds_from_to_crs(self, 
                                    orig_crs : str, dest_crs : str, 
                                    bbox : tuple[float,float,float,float]) -> tuple[float,float,float,float]:
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
    

    def _compute_grid_meters(self, 
                             bbox: tuple[float, float, float, float], 
                             step: float, offset: float = 0.0) -> gpd.GeoDataFrame:
        # TODO: Investigare l'uso di "geohexgrid" per la creazione di griglie uniforme esagonali.
        '''
        Compute a uniform grid of square cells with side 'step' (in meters) over the bounding box 'bbox', 
        which is given in the form (minx, miny, maxx, maxy).
        The grid is returned as a GeoDataFrame of Shapely Polygons, each representing a cell of the grid.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            The bounding box over which the grid must be computed, in the form (minx, miny, maxx, maxy).
            NOTE: The bounding box must be given in a metric CRS, i.e., with coordinates in meters. 
        step : float
            The length of the side of the grid's cells, in meters. 
        offset : float, optional
            offset is a bbox-relative phase shift of the internal grid lines. It does not shift the bbox itself.
            The value must be in meters and fall in [0, step).
        '''

        # Unpack the bounding box.
        min_x, min_y, max_x, max_y = map(float, bbox)


        # Compute the coordinates of the bottom-left corner of the bounding box with the offset applied.
        offset_min_x = min_x - ((step - offset) % step)
        offset_min_y = min_y - ((step - offset) % step)

        # Compute the number of cells in the x and y directions, rounding up to ensure full coverage 
        # of the original bounding box.
        nx = math.ceil((max_x - offset_min_x) / step)
        ny = math.ceil((max_y - offset_min_y) / step)

        # Flattened coordinates of the bottom-left corners of the cells, taking into account the applied offset.
        x0 = offset_min_x + step * np.repeat(np.arange(nx, dtype=float), ny)
        y0 = offset_min_y + step * np.tile(np.arange(ny, dtype=float), nx)

        # 1D flattened coordinates of the top-right corners of the cells.
        # Also, clip the coordinates of the top-right corners of the cells, so that they stay within
        # the original bounding box boundaries.
        x1 = np.minimum(x0 + step, max_x)
        y1 = np.minimum(y0 + step, max_y)

        # Clip the coordinates of the bottom-left corners of the cells, so that they stay within the original bounding box boundaries.
        x0 = np.maximum(x0, min_x)
        y0 = np.maximum(y0, min_y)

        # Create the Shapely grid (note: box vectorized in Shapely 2.x).
        geoms = shapely.box(x0, y0, x1, y1)
        return gpd.GeoDataFrame(geometry=geoms)


    def _compute_grid_over_bbox(self, bbox, orig_crs, metric_crs) -> gpd.GeoDataFrame :
        
        # Reproject the bounding box's coords from the original CRS to a metric CRS.
        # Also add a small buffer of 'enlarge_meters' meters to the original bounding box, ensuring we leave a little
        # space around the original area covered in 'geo_df'.
        enlarge_meters = 5
        bbox_metric = self._project_bounds_from_to_crs(orig_crs, metric_crs, bbox)
        bbox_metric_enlarged = shapely.box(*bbox_metric).buffer(enlarge_meters).bounds

        # Compute a uniform grid with side 'self.cell_length_meters' over the bounding box.
        grid = self._compute_grid_meters(bbox_metric_enlarged, self.cell_length_meters, self.offset).set_crs(crs=metric_crs)

        # Reproject the grid to the original CRS. Finally, return it.
        return grid.to_crs(crs=orig_crs)
    


    ### PUBLIC METHODS ###

    def get_df_grid(self) -> gpd.GeoDataFrame :
        '''
        Return a reference to the GeoDataFrame containing the grid cells.
        '''
        return self.grid
    

    def get_grid_cell_length_meters(self) -> float :
        '''
        Return the length of the grid's cells' side, in meters.
        '''
        return self.cell_length_meters


    def get_grid_offset_meters(self) -> float :
        '''
        Return the offset applied to the grid's cells' 'x' and 'y' coordinates, in meters.
        '''
        return self.offset
    

    def save_to_file(self, filepath : str) -> None :
        '''
        Save the Grid instance to a file, using pickle serialization.
        The file will be created at the path specified by 'filepath'.
        '''

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load_from_pickle(cls, filepath : str) -> 'Grid' :
        '''
        Alternative class constructor: load a Grid instance from a file, using pickle serialization.
        The file must be located at the path specified by 'filepath'.
        '''

        with open(filepath, "rb") as f:
            return pickle.load(f)
    

    def generate_grid_map(self):
        '''
        Generate a simple Folium map showing the grid's cells.
        '''

        # Create a Folium Map object.
        bbox = self.grid.total_bounds
        m = folium.Map(location=[(bbox[3]+bbox[1])/2, (bbox[2]+bbox[0])/2,], 
                       zoom_start=13, control_scale=True, prefer_canvas=True)
        
        # Plot the grid's cells' polygons on the map.
        folium.GeoJson(self.grid).add_to(m)


        # 5. Display or save
        return m