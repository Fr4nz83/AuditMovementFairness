import pandas as pd
import geopandas as gpd
import numpy as np
import random
import math

from shapely.affinity import rotate, translate
from shapely import Polygon, MultiPolygon, Point, maximum_inscribed_circle

from tqdm import tqdm


### Constants used while generating hotspots... ###
_INITIAL_HIGH_BUFFER : float = 1.
_GROWTH : float = 2.0
_TOL : float = 0.1


### FUNCTIONS ###

def rotate_translate_poly(poly : Polygon) :

    # Determine the original polygon's bounding box.
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny

    # Rotate the original polygon by a random amount of degrees, using as a pivot one
    # of its internal points (picked up randomly).
    polygon =  rotate(poly,
                      angle=random.uniform(0, 360),
                      use_radians=False)
    
    # Translate the rotated polygon in a random 2D direction but not too far,
    # i.e., the x and y offsets are chosen according to the rotated polygon's bbox.
    polygon = translate(polygon,
                        xoff=random.uniform(-0.5 * w, 0.5 * w),
                        yoff=random.uniform(-0.5 * h, 0.5 * h))
    
    return polygon


def eval_buffer(in_poly : Polygon, sindex, uid_values : np.ndarray,
                buffer : float, min_stops_object : int) -> tuple[np.ndarray, Polygon]:
        
    # Apply a buffer to 'in_poly'
    out_poly = in_poly.buffer(buffer)
    
    # Count the number of stop segment centroids that fall in 'out_poly'
    # print(f"DEBUG: Querying rtree...")
    idx = sindex.query(out_poly, predicate="contains", sort=False)

    # Retrieve the users' IDs.
    list_uids = uid_values[idx]

    # For each ID that occurs in 'list_uids', count the number of its occurrences.
    # Then, determine the number of distinct objects that have more than X stops in the polygon: these
    # will be associated with the polygon.
    values, freq_values = np.unique(list_uids, return_counts=True)
    selected_values = values[freq_values >= min_stops_object]
        
    return selected_values, out_poly


def gen_hotspot(polygon_base : Polygon, rtree_stops, stop_uid_values : np.ndarray,
                target_num_objs : int, min_stops_object : int) :
    """
    Generate a synthetic hotspot of unfairness.
    """

    # 1 - Rotate and translate the original polygon.
    polygon =  rotate_translate_poly(polygon_base)



    ### 2 - Now apply an expanding buffer to the rotated polygon, until it becomes associated with the
    ###     desired number of objects.

    # 2.1 - Set the lower bound for the buffer (this equals to the poly collapsing to a point).
    low_buffer = -maximum_inscribed_circle(polygon).length

    # print(f"DEBUG: Initializing high buffer...")
    # 2.2 - Initial expansion of the upper bound, until we go beyond the target number of objects to associate.
    high_buffer = _INITIAL_HIGH_BUFFER
    high_list_objs, init_poly = eval_buffer(polygon, rtree_stops, stop_uid_values,
                                            high_buffer, min_stops_object)
    while high_list_objs.size < target_num_objs:
        low_buffer = high_buffer
        high_buffer *= _GROWTH
        high_list_objs, init_poly = eval_buffer(polygon, rtree_stops, stop_uid_values,
                                                high_buffer, min_stops_object)
        # print(f"DEBUG: init: current high buffer: {high_buffer}")
    # print(f"DEBUG: Initial high buffer: {high_buffer}, initial num objs high buffer: {high_list_objs.size}")
    best_poly, best_list_objs = init_poly, high_list_objs


    # 2.3 - Binary search for the smallest buffer with count >= target_objs. 
    best_diff_target = abs(target_num_objs - high_list_objs.size)
    while (high_buffer - low_buffer) > _TOL:
        
        # Find out how many objects associate with the polygon buffered with 'mid_buffer'.
        mid_buffer = 0.5 * (low_buffer + high_buffer)
        # print(f"DEBUG: Current interval: [{low_buffer},{high_buffer}], mid_buffer: {mid_buffer}")
        mid_list_objs, poly_mid = eval_buffer(polygon, rtree_stops, stop_uid_values,
                                              mid_buffer, min_stops_object)


        # Update the boundaries of the search interval.
        if mid_list_objs.size >= target_num_objs:
            # print(f"DEBUG: Over the target: {mid_list_objs.size}>={target_num_objs}")
            high_buffer = mid_buffer
        else:
            # print(f"DEBUG: Under the target: {mid_list_objs.size}<{target_num_objs}")
            low_buffer = mid_buffer


        # Update the best polygon found if the absolute difference with the target num_objs is lower
        # than what has been previously found.
        diff_target = abs(target_num_objs - mid_list_objs.size)
        if(diff_target < best_diff_target) :
            # print(f"DEBUG: Better buffered polygon found, difference: {diff_target} ({mid_list_objs.size})")
            best_poly, best_list_objs, best_diff_target = poly_mid, mid_list_objs, diff_target
        

        # If we have associated exactly the number of desired objects, exit the loop.
        if diff_target == 0 : break

    # Return the best buffered polygon found, as well as the number of objects it is associated with.
    return best_poly, best_list_objs


def gen_unfair_labels(num_objs : int, list_objs_unfair : np.ndarray,
                      global_pos_rate : float, hotspots_pos_rate : float) :
    
    # 1 - Generate the "fair" labels for a given number of objects. 
    labels = np.random.default_rng().binomial(n=1, p=global_pos_rate, size=num_objs).astype(np.int8)

    # 2 - Now, generate the unfair labels to be applied to a selected set of objects.
    num_penalized_objects = list_objs_unfair.size
    penalized_labels_obj = (np.random.default_rng().binomial(n=1, p=hotspots_pos_rate, size=num_penalized_objects)
                                                   .astype(np.int8))

    # 3 - Apply the unfair labels to the objects associated with 'penalized candidate'.
    labels[list_objs_unfair] = penalized_labels_obj

    return labels




def gen_unfair_datasets(df_polygons : gpd.GeoDataFrame, df_stops : gpd.GeoDataFrame,
                        num_unfair_datasets : int, 
                        num_hotspots_dataset : int, target_num_objs : int,
                        global_pos_rate : float, hotspots_pos_rate : float) :
    
    # Count the total number of objects
    tot_num_objs = df_stops["uid"].nunique()
    
    # Get the rtree associated with df_stops.
    rtree_stops = df_stops.sindex

    # Store the user IDs associated with the stops in a numpy array.
    stop_uid_values = df_stops["uid"].to_numpy()

    list_unfair_datasets = []
    for idx_dataset in tqdm(range(num_unfair_datasets), desc="Generating unfair datasets of labels...") :
    # for idx_dataset in range(num_unfair_datasets) :

        # Randomly pick a census block from those that contain at least one stop segment.
        polygon_base = df_polygons.sample(1)
        # print(f"DEBUG: sampled polygon idx: {polygon_base.index.values[0]}")
        polygon_base = polygon_base.geometry.iloc[0]
        # polygon_base.plot()

        # Generate the polygon of a hotspot that associates with a 'target_num_objs' objects.
        poly_hotspot, list_objs_hotspot = gen_hotspot(polygon_base, rtree_stops, stop_uid_values, 
                                                      target_num_objs, min_stops_object=2)
        
        # Generate the dataset of labels, injecting unfairness in the objects associated with the hotspot.
        unfair_labels = gen_unfair_labels(tot_num_objs, list_objs_hotspot, global_pos_rate, hotspots_pos_rate)

        # Add the unfair dataset to the list.
        list_unfair_datasets.append( (poly_hotspot, list_objs_hotspot, unfair_labels) )
        # break


    return list_unfair_datasets