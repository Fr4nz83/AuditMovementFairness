import geopandas as gpd

def create_traj_partitioning(gdf_traj: gpd.GeoDataFrame,
                             lon_n: int, lat_n: int) -> tuple[dict, dict, list]:
    """
    Partition a geographic area into a grid and return the grid information, a mapping from grid cell coordinates
    to partition indices, and a list of partitions. Each partition is a dict with:
      - 'grid_loc': (j, i) coordinate of the cell.
      - 'trajectories': list of trajectory indices (from gdf_traj) that intersect the cell.
      
    Parameters:
      gdf_traj: GeoDataFrame containing trajectories.
      lon_min, lon_max, lat_min, lat_max: bounding coordinates of the area.
      lon_n: number of grid cells along the longitude (x-axis).
      lat_n: number of grid cells along the latitude (y-axis).
    
    Note:
      The function assumes gdf_traj is in a CRS where x corresponds to longitude and y to latitude.
    """

    # Find the bounding box of the trajectory dataset.
    lon_min, lat_min, lon_max, lat_max = gdf_traj.total_bounds
    print(f'{lon_min}, {lat_min}, {lon_max}, {lat_max}')

    grid_info = {
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lat_n': lat_n,
        'lon_n': lon_n,
    }

    grid_loc2_idx = {}  # Maps each grid cell (j, i) to its index in partitions.
    partitions = []     # List to store each cell's partition info.

    # Precompute cell dimensions.
    cell_width = (lon_max - lon_min) / lon_n
    cell_height = (lat_max - lat_min) / lat_n

    for i in range(lat_n):
        lat_start = lat_min + i * cell_height
        lat_end = lat_start + cell_height
        
        for j in range(lon_n):
            lon_start = lon_min + j * cell_width
            lon_end = lon_start + cell_width

            # Use .cx to find the trajectories that intersect this cell (spatial join).
            list_trajs = gdf_traj.cx[lon_start:lon_end, lat_start:lat_end].index.tolist()
            
            # Create a partition dict in which we store the grid location and the list of trajectory indices.
            # NOTE: we name the field 'points' instead of 'trajectories' to maintain compatibility with existing code.
            partition = {
                'grid_loc': (j, i),
                'points': list_trajs,
            }

            # Map the coordinates of this cell to the index in the partitions list.
            grid_loc2_idx[(j, i)] = len(partitions)

            # Append this parttion to the partitions list.
            partitions.append(partition)
    
    return grid_info, grid_loc2_idx, partitions