import geopandas as gpd
import numpy as np
import random
from shapely import affinity, maximum_inscribed_circle
from tqdm import tqdm

from shapely import Polygon, MultiPolygon
from geopandas.sindex import SpatialIndex


### Constants used while generating hotspots... ###
_INITIAL_HIGH_BUFFER : float = 1.
_GROWTH : float = 4.0
_TOL : float = 0.1


### FUNCTIONS ###

def rotate_translate_poly(poly : Polygon) -> Polygon :
    """
    Randomly rotate and translate a polygon.

    The function applies a random rotation to the input polygon and then
    translates the rotated geometry by a random offset along both axes.
    The translation magnitude is bounded by half of the width and height
    of the input polygon's bounding box.

    Parameters
    ----------
    poly : Polygon
        Input polygon to transform.

    Returns
    -------
    Polygon
        A new polygon obtained by randomly rotating and translating
        the input geometry.

    Notes
    -----
    The transformation preserves the polygon's shape and size, changing
    only its orientation and position.
    """

    # Rotate the original polygon by a random amount of degrees, using as a pivot one
    # of its internal points (picked up randomly).
    polygon = affinity.rotate(poly,
                              angle=random.uniform(0, 360),
                              use_radians=False)
    
    # Translate the rotated polygon in a random 2D direction but not too far,
    # i.e., the x and y offsets are chosen according to the rotated polygon's bbox.
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    polygon = affinity.translate(polygon,
                                 xoff=random.uniform(-0.5 * w, 0.5 * w),
                                 yoff=random.uniform(-0.5 * h, 0.5 * h))
    
    return polygon


def eval_buffer(in_poly : Polygon, sindex : SpatialIndex, uid_values : np.ndarray,
                buffer : float, min_stops_object : int) -> tuple[np.ndarray, Polygon]:
    """
    Buffer a polygon and determine which objects are associated with it.

    The function expands or contracts the input polygon by the specified
    buffer distance, queries the spatial index to find the stop centroids
    contained in the buffered polygon, and then selects the distinct object
    IDs whose number of contained stops is at least `min_stops_object`.

    Parameters
    ----------
    in_poly : Polygon
        Input polygon to buffer.
    sindex : geopandas.sindex.SpatialIndex
        Spatial index built on the stop geometries to be queried.
    uid_values : np.ndarray
        Array of object IDs aligned with the geometries indexed in `sindex`.
    buffer : float
        Buffer distance applied to `in_poly`. Positive values enlarge the
        polygon, while negative values shrink it.
    min_stops_object : int
        Minimum number of stops that an object must contribute inside the
        buffered polygon in order to be associated with it.

    Returns
    -------
    tuple[np.ndarray, Polygon]
        A tuple containing:

        - selected_values : np.ndarray
            Distinct object IDs associated with the buffered polygon.
        - out_poly : Polygon
            Buffered version of the input polygon.

    Notes
    -----
    This function assumes that `uid_values[i]` corresponds to the same stop
    geometry referenced by index `i` in `sindex`.
    """
        
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
    values, freq_values = np.unique(list_uids, return_counts=True, sorted=False)
    selected_values = values[freq_values >= min_stops_object]
        
    return selected_values, out_poly


def gen_hotspot(polygon : Polygon, rtree_stops : SpatialIndex, stop_uid_values : np.ndarray,
                target_num_objs : int, min_stops_object : int) -> tuple[Polygon, np.ndarray]:
    """
    Generate a synthetic hotspot associated with approximately a target number
    of objects.

    Starting from a base polygon, the function searches for a buffer distance
    that makes the polygon associated with about `target_num_objs` distinct
    objects. It first expands the upper search bound exponentially until the
    target is reached or exceeded, and then refines the result with a binary
    search. The best polygon found during the search is returned.

    Parameters
    ----------
    polygon : Polygon
        Base polygon used to generate the hotspot.
    rtree_stops : geopandas.sindex.SpatialIndex
        Spatial index built on the stop geometries.
    stop_uid_values : np.ndarray
        Array of object IDs aligned with the geometries indexed in
        `rtree_stops`.
    target_num_objs : int
        Desired number of distinct objects to associate with the hotspot.
    min_stops_object : int
        Minimum number of contained stops required for an object to be
        considered part of the hotspot.

    Returns
    -------
    tuple[Polygon, np.ndarray]
        A tuple containing:
        - best_poly : Polygon
            Buffered polygon whose associated number of objects is the best
            approximation found for the target.
        - best_list_objs : np.ndarray
            Distinct object IDs associated with `best_poly`.

    Notes
    -----
    The search starts from a lower bound obtained by shrinking the polygon
    up to the radius of its maximum inscribed circle, and stops when the
    buffer interval is smaller than the tolerance `_TOL` or when the target
    is matched exactly.
    """

    ### Apply an expanding buffer to the rotated polygon, until it becomes associated with the
    ### desired number of objects.

    # 1 - Set the lower bound for the buffer (this equals to the poly collapsing to a point).
    low_buffer = -maximum_inscribed_circle(polygon).length

    # print(f"DEBUG: Initializing high buffer...")
    # 2 - Initial expansion of the upper bound, until we go beyond the target number of objects to associate.
    high_buffer = _INITIAL_HIGH_BUFFER
    high_list_objs, init_poly = eval_buffer(polygon, rtree_stops, stop_uid_values,
                                            high_buffer, min_stops_object)
    while high_list_objs.size < target_num_objs:
        low_buffer = high_buffer
        high_buffer *= _GROWTH
        high_list_objs, init_poly = eval_buffer(polygon, rtree_stops, stop_uid_values,
                                                high_buffer, min_stops_object)
        # print(f"DEBUG: init: current high buffer: {high_buffer}")
    best_poly, best_list_objs = init_poly, high_list_objs
    # print(f"DEBUG: Initial high buffer: {high_buffer}, initial num objs high buffer: {high_list_objs.size}")


    # 3 - Binary search for the smallest buffer with count >= target_objs. 
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


def gen_unfair_labels(num_objs : int, list_lists_objs_unfair : np.ndarray,
                      global_pos_rate : float, hotspots_pos_rate : float) -> np.ndarray :
    """
    Generate binary labels with synthetic unfairness injected in hotspot objects.

    The function first generates baseline binary labels for all objects using
    the global positive rate. It then identifies the union of the objects
    associated with the unfair hotspots and overwrites their labels with new
    samples drawn using the hotspot-specific positive rate.

    Parameters
    ----------
    num_objs : int
        Total number of objects for which labels must be generated.
    list_lists_objs_unfair : np.ndarray
        Collection of arrays, where each array contains the object IDs
        associated with one unfair hotspot.
    global_pos_rate : float
        Bernoulli success probability used to generate the baseline labels
        for all objects.
    hotspots_pos_rate : float
        Bernoulli success probability used to regenerate the labels of the
        objects belonging to at least one unfair hotspot.

    Returns
    -------
    np.ndarray
        One-dimensional array of dtype `np.int8` containing the generated
        binary labels.

    Notes
    -----
    Objects that belong to multiple hotspots are penalized only once, since
    the function applies the hotspot effect to the union of hotspot object IDs.
    """
    
    # 1 - Generate the "fair" labels for a given number of objects. 
    labels = np.random.default_rng().binomial(n=1, p=global_pos_rate, size=num_objs).astype(np.int8)

    # 2 - Now, generate the unfair labels to be applied to a selected set of objects.
    list_objs_unfair = np.unique(np.concatenate(list_lists_objs_unfair), sorted=False)
    num_penalized_objects = list_objs_unfair.size
    penalized_labels_obj = (np.random.default_rng().binomial(n=1, p=hotspots_pos_rate, size=num_penalized_objects)
                                                   .astype(np.int8))

    # 3 - Apply the unfair labels to the objects associated with the various hotspots.
    labels[list_objs_unfair] = penalized_labels_obj

    return labels




def gen_unfair_datasets(df_polygons : gpd.GeoDataFrame, df_stops : gpd.GeoDataFrame,
                        num_unfair_datasets : int, 
                        num_hotspots_per_dataset : int, target_num_objs : int,
                        global_pos_rate : float, hotspots_pos_rate : float) -> list[tuple[list[Polygon],
                                                                                          list[list[np.ndarray]],
                                                                                          np.ndarray]] :
    """
    Generate multiple synthetic datasets containing unfair spatial hotspots.

    For each dataset, the function samples a set of base polygons from
    `df_polygons`, randomly rotates and translates them, grows each polygon
    into a hotspot associated with approximately `target_num_objs` objects,
    and finally generates a binary label vector in which unfairness is
    injected into the hotspot objects.

    Parameters
    ----------
    df_polygons : gpd.GeoDataFrame
        GeoDataFrame containing the candidate base polygons used to build
        the hotspots.
    df_stops : gpd.GeoDataFrame
        GeoDataFrame containing stop geometries and a `uid` column
        identifying the object to which each stop belongs.
    num_unfair_datasets : int
        Number of synthetic unfair datasets to generate.
    num_hotspots_per_dataset : int
        Number of hotspots to generate in each dataset.
    target_num_objs : int
        Target number of distinct objects to associate with each hotspot.
    global_pos_rate : float
        Bernoulli success probability for the baseline label generation.
    hotspots_pos_rate : float
        Bernoulli success probability applied to the objects belonging to
        the generated hotspots.

    Returns
    -------
    list
        List of dataset tuples. Each tuple has the form "(list_poly_hotspots, list_lists_objs_hotspots, unfair_labels)",
        where:
        - `list_poly_hotspots` is the list of hotspot polygons,
        - `list_lists_objs_hotspots` is the list of object-ID arrays
        associated with each hotspot,
        - `unfair_labels` is the generated binary label vector.
    """


    # Count the total number of objects
    tot_num_objs = df_stops["uid"].nunique()
    
    # Get the rtree associated with df_stops.
    rtree_stops = df_stops.sindex

    # Store the user IDs associated with the stops in a numpy array.
    stop_uid_values = df_stops["uid"].to_numpy()

    # Randomly pick 'num_unfair_datasets * num_hotspots_per_dataset' census block (with repetitions) from those that contain
    # at least one stop segment. They will be used as base polygons to create the hotspots.
    list_unfair_datasets = []
    for idx_dataset in tqdm(range(num_unfair_datasets), desc="Generating unfair datasets of labels...") :
    # for idx_dataset in range(num_unfair_datasets) :

        # Pick 'num_hotspots_per_dataset' randomly from 'df_polygons', and then transform each of them applying
        # a rotation and translation.
        base_polygons = list(map(rotate_translate_poly, df_polygons.sample(num_hotspots_per_dataset).geometry.to_list()))

        # Generate the hotspots using the sampled and transformed polygons, associating each of them with 
        # 'target_num_objs' objects.
        list_poly_hotspots = []
        list_lists_objs_hotspots = []
        for idx_hotspot in range(num_hotspots_per_dataset) :
            polygon_base = base_polygons[idx_hotspot]
            poly_hotspot, list_objs_hotspot = gen_hotspot(polygon_base, rtree_stops, stop_uid_values, 
                                                          target_num_objs, min_stops_object=2)
            
            # Append the information related to the hotspot just created.
            list_poly_hotspots.append(poly_hotspot)
            list_lists_objs_hotspots.append(list_objs_hotspot)
        
        # Generate the dataset of labels, injecting unfairness in the objects associated with the hotspots.
        unfair_labels = gen_unfair_labels(tot_num_objs, list_lists_objs_hotspots, global_pos_rate, hotspots_pos_rate)

        # Add the unfair dataset to the list.
        unfair_dataset = (list_poly_hotspots, list_lists_objs_hotspots, unfair_labels)
        list_unfair_datasets.append(unfair_dataset)
        # break


    return list_unfair_datasets