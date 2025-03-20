import geopandas as gpd
import folium
from .functions import get_simple_stats


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


def show_traj_grid_regions(traj_df, grid_info, types, normal_regions, significant_regions):

    lon_min = grid_info['lon_min']
    lon_max = grid_info['lon_max']
    lat_min = grid_info['lat_min']
    lat_max = grid_info['lat_max']

    lon_n = grid_info['lon_n']
    lat_n = grid_info['lat_n']


    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, prefer_canvas = True, tiles='cartodbpositron')

    for region in normal_regions:
        i, j = region['grid_loc']

        lon_start = lon_min + (i/lon_n)*(lon_max - lon_min)
        lon_end = lon_min + ((i+1)/lon_n)*(lon_max - lon_min)

        lat_start = lat_min + (j/lat_n)*(lat_max - lat_min)
        lat_end = lat_min + ((j+1)/lat_n)*(lat_max - lat_min)

        n, p, rho = get_simple_stats(region['points'], types)

        folium.Rectangle([(lat_start,lon_start), (lat_end,lon_end)],
                         weight=0, #removes border
                         fill=True,
                         fill_color="blue",     # Choose any color you want
                         fill_opacity=0.2,      # Adjust the opacity as needed
                         tooltip=f'# examples={n}, # positives={p}, ratio={rho:.2f}').add_to(mapit)

    for region in significant_regions:
        i, j = region['grid_loc']

        lon_start = lon_min + (i/lon_n)*(lon_max - lon_min)
        lon_end = lon_min + ((i+1)/lon_n)*(lon_max - lon_min)

        lat_start = lat_min + (j/lat_n)*(lat_max - lat_min)
        lat_end = lat_min + ((j+1)/lat_n)*(lat_max - lat_min)

        n, p, rho = get_simple_stats(region['points'], types)

        folium.Rectangle([(lat_start,lon_start), (lat_end,lon_end)],
                  weight=0, #removes border
                  fill=True,
                  fill_color="red",     # Choose any color you want
                  fill_opacity=0.5,      # Adjust the opacity as needed
                  tooltip=f'# examples={n}, # positives={p}, ratio={rho:.2f}').add_to(mapit)

    # TODO: Draw the trajectories.

    mapit.fit_bounds([(lat_min, lon_min), (lat_max, lon_max)])

    return mapit