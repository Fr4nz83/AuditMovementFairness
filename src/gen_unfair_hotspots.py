import pandas as pd
import geopandas as gpd
import numpy as np
import random

from shapely.affinity import rotate, scale

from tqdm import tqdm


### Constants used while generating hotspots... ###
_INITIAL_HIGH_SCALE : float = 1.
_GROWTH : float = 2.0
_TOL : float = 0.0001


### FUNCTIONS ###

def eval_scale(in_poly, anchor_point, sindex, uid_values : np.ndarray,
               scale_factor : float, min_stops_object : int):
        
    # Apply a scaling factor to 'in_poly'
    out_poly = scale(in_poly, xfact=scale_factor, yfact=scale_factor, origin=anchor_point)
    
    # Count the number of stop segment centroids that fall in 'out_poly'
    idx = sindex.query(out_poly, predicate="contains", sort=False)

    # Retrieve the users' IDs.
    list_uids = uid_values[idx]

    # For each ID that occurs in 'list_uids', count the number of its occurrences.
    # Then, determine the number of distinct objects that have more than X stops in the polygon: these
    # will be associated with the polygon.
    values, freq_values = np.unique(list_uids, return_counts=True)
    selected_values = values[freq_values >= min_stops_object]
        
    return selected_values, out_poly


def gen_hotspot(df_polygons : gpd.GeoDataFrame, rtree_stops, stop_uid_values : np.ndarray,
                target_num_objs : int, min_stops_object : int) :
    """
    Generate a synthetic hotspot of unfairness.
    """

    # Randomly pick a census block from those that contain at least one stop segment.
    polygon_base = df_polygons.sample(1).geometry.iloc[0]
    # polygon_base.plot()
    
    # Pick a random point within 'polygon_base': we'll use it as anchor/origin for the polygon's transformed geometry.
    anchor_point = polygon_base.representative_point()

    # Rotate the original polygon by a random amount of degrees.
    polygon =  rotate(polygon_base,
                      angle=random.uniform(0, 360),
                      origin=anchor_point,
                      use_radians=False)
    

    ### Now we rescale the rotated polygon, anchored on 'anchor_point' (this equals to a translation of the position of ###
    ### the original block geom.) until it becomes associated with the desired number of objects.                       ###


    # 1 - Set the lower bound for the scaling.
    low_scale = 0.


    # 2 - Initial expansion of the upper bound, until we go beyond the target number of objects to associate.
    high_scale = _INITIAL_HIGH_SCALE
    high_list_objs, _ = eval_scale(polygon, anchor_point, rtree_stops, stop_uid_values,
                                   high_scale, min_stops_object)
    while high_list_objs.size < target_num_objs:
        low_scale = high_scale
        high_scale *= _GROWTH
        high_list_objs, _ = eval_scale(polygon, anchor_point, rtree_stops, stop_uid_values,
                                      high_scale, min_stops_object)
    # print(f"DEBUG: Initial high scale: {high_scale}, initial num objs high scale: {num_high_objs}")


    # 3 - Binary search for the smallest scale with count >= target_objs. 
    best_diff_target = abs(target_num_objs - high_list_objs.size)
    while (high_scale - low_scale) > _TOL:
        
        # Find out how many objects associate with the polygon scaled with 'mid_scale'.
        mid_scale = 0.5 * (low_scale + high_scale)
        # print(f"DEBUG: Current interval: [{low_scale},{high_scale}], mid_scale: {mid_scale}")
        mid_list_objs, poly_mid = eval_scale(polygon, anchor_point, rtree_stops, stop_uid_values,
                                                  mid_scale, min_stops_object)


        # Update the boundaries of the search interval.
        if mid_list_objs.size >= target_num_objs:
            # print(f"DEBUG: Over the target: {num_mid_scale_objs}>={target_num_objs}")
            high_scale = mid_scale
        else:
            # print(f"DEBUG: Under the target: {num_mid_scale_objs}<{target_num_objs}")
            low_scale = mid_scale


        # Update the best polygon found if the absolute difference with the target num_objs is lower
        # than what has been previously found.
        diff_target = abs(target_num_objs - mid_list_objs.size)
        if(diff_target < best_diff_target) :
            # print(f"DEBUG: Better polygon found, difference: {diff_target} ({num_mid_scale_objs})")
            best_poly, best_list_objs, best_diff_target = poly_mid, mid_list_objs, diff_target
        

        # If we have associated exactly the number of desired objects, exit the loop.
        if diff_target == 0 : break

    # Return the best scaled polygon found, as well as the number of objects it is associated with.
    return best_poly, best_list_objs


def gen_unfair_labels(num_objs : int, list_objs_unfair : np.ndarray,
                      global_pos_rate : float, hotspots_pos_rate : float) :
    
    # 1 - Generate the "fair" labels for a given number of objects. 
    labels = np.random.default_rng().binomial(n=1, p=global_pos_rate, size=num_objs).astype(np.int8)

    # 2 - Now, generate the unfair labels to be applied to a selected set of objects.
    num_penalized_objects = list_objs_unfair.size
    penalized_labels_obj = np.random.default_rng().binomial(n=1, p=hotspots_pos_rate, size=num_penalized_objects).astype(np.int8)

    # 3 - Apply the unfair labels to the objects associated with 'penalized candidate'.
    labels[list_objs_unfair] = penalized_labels_obj

    return labels




def gen_set_hotspots(df_polygons : gpd.GeoDataFrame, df_stops : gpd.GeoDataFrame,
                    target_num_hotspots : int, target_num_objs : int,
                    global_pos_rate : float, hotspots_pos_rate : float) :
    
    # Count the total number of objects
    tot_num_objs = df_stops["uid"].nunique()
    
    # Get the rtree associated with df_stops.
    rtree_stops = df_stops.sindex

    # Store the user IDs associated with the stops in a numpy array.
    stop_uid_values = df_stops["uid"].to_numpy()

    list_hotspots = []
    for idx_hotspot in tqdm(range(target_num_hotspots), desc="Generating hotspots...") :
    # for idx_hotspot in range(target_num_hotspots) :

        # Generate the polygon of a hotspot that associates with a 'target_num_objs' objects.
        poly_hotspot, list_objs_hotspot = gen_hotspot(df_polygons, rtree_stops, stop_uid_values, 
                                                      target_num_objs, min_stops_object=2)
        
        # Generate the dataset of labels, injecting unfairness in the objects associated with the hotspot.
        unfair_labels = gen_unfair_labels(tot_num_objs, list_objs_hotspot, global_pos_rate, hotspots_pos_rate)

        # Add the unfair dataset to the list.
        list_hotspots.append( (poly_hotspot, list_objs_hotspot, unfair_labels) )
        # break

    return list_hotspots