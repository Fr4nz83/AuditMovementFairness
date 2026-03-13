import pandas as pd
import geopandas as gpd
import numpy as np
import random
from shapely import affinity, maximum_inscribed_circle
from tqdm import tqdm
from itertools import combinations
from functools import reduce

from shapely import Polygon, MultiPolygon
from geopandas.sindex import SpatialIndex


### Constants used while generating hotspots... ###
_INITIAL_HIGH_BUFFER : float = 1.
_GROWTH : float = 2.0
_TOL : float = 0.1


### FUNCTIONS ###

def eval_buffer_multiregion(in_multipoly : MultiPolygon, sindex : SpatialIndex, uid_values : np.ndarray,
                            buffer : float, min_stops_object : int) -> tuple[np.ndarray, Polygon] | None :
        
    # Apply a buffer to 'in_poly'
    out_multipoly = in_multipoly.buffer(buffer)


    # Check that we have a MultiPolygon, and that the number of polygons in 'out_multipoly' is equal to 'in_multipoly':
    # if this is not the case, then it means that at least a pair of enlarged polygons intersect after buffering, which 
    # in turn makes Shapely simplify them into Polygons.
    if not isinstance(out_multipoly, MultiPolygon): 
        # print(f"DEBUG: buffered multipolygon is not a polygon anymore! {type(out_multipoly)}")
        return None
    if len(in_multipoly.geoms) != len(out_multipoly.geoms) : 
        # print(f"DEBUG: buffered multipolygon has less polygons! {len(in_multipoly.geoms)}vs{len(out_multipoly.geoms)}")
        return None
    

    # Find out the stop segment centroids that fall in any of the polygons of 'out_poly'
    # print(f"DEBUG: Querying rtree...")
    list_selected_values = []
    for poly in out_multipoly.geoms:
        # Find the stops' centroids and then the associated user IDs.
        idx_stop = sindex.query(poly, predicate="contains", sort=False)
        uids_stop = uid_values[idx_stop]

        # Determine the user IDs that have at least 'min_stops_object' in this polygon.
        values, freq_values = np.unique(uids_stop, return_counts=True)
        list_selected_values.append(values[freq_values >= min_stops_object])


    # Find out the set of users that appear in ALL the polygons. 
    uid_intersection = (
        reduce(np.intersect1d, list_selected_values)
        if list_selected_values
        else np.array([], dtype=uid_values.dtype)
    )
        
    return uid_intersection, out_multipoly


def init_upper_bound_multiregion(base_multipolygon: MultiPolygon,
                                 rtree_stops: SpatialIndex,
                                 stop_uid_values: np.ndarray,
                                 target_num_objs: int,
                                 min_stops_object: int,
                                 low_buffer: float) -> tuple[float, float, np.ndarray, MultiPolygon] | None :
    """
    Initialize the search bounds for gen_multiregion_hotspot.
    """

    # Start from the unbuffered multipolygon.
    # NOTE: its polygons surely do not intersect.
    high_list_objs, init_multipoly = eval_buffer_multiregion(
        base_multipolygon, rtree_stops, stop_uid_values, 0.0, min_stops_object
    )

    # If target is already reached at buffer 0, keep the original lower bound.
    if high_list_objs.size >= target_num_objs:
        return low_buffer, 0.0, high_list_objs, init_multipoly

    # Exponential expansion until:
    #   1) we reach the target, or
    #   2) buffering makes polygons intersect.
    last_valid_buffer = 0.0
    probe_buffer = _INITIAL_HIGH_BUFFER

    while True:
        res = eval_buffer_multiregion(base_multipolygon, rtree_stops, stop_uid_values,
                                      probe_buffer, min_stops_object)

        # 1 - We hit a buffer that makes polygons intersect while exponentially enlarging the buffer: trigger
        # a binary search!
        if res is None:
            left = last_valid_buffer           # valid, but still below target
            right = probe_buffer              # invalid (intersection)
            best_found = None                 # (buffer, list_objs, multipoly)

            # There may still be a valid solution inside (last_valid_buffer, probe_buffer), so binary-search
            # in this region.
            while (right - left) > _TOL:
                mid = 0.5 * (left + right)
                mid_res = eval_buffer_multiregion(base_multipolygon, rtree_stops, stop_uid_values,
                                                  mid, min_stops_object)

                # Mid buffer intersects -> move leftward.
                if mid_res is None:
                    right = mid
                    continue
                
                # Mid buffer multipolygon's polygons do not intersect: retrieve the mid infos.
                mid_list_objs, mid_multipoly = mid_res

                # Check if we reached the target number of objects.
                # 1.1 - Valid and reaches target -> keep it as candidate upper bound,
                # and keep searching left for a smaller valid one.
                if mid_list_objs.size >= target_num_objs:
                    best_found = (mid, mid_list_objs, mid_multipoly)
                    right = mid
                # 1.2 - Valid but still below target.
                else: left = mid

            # If no suitable buffer has been found, it means it is impossible to enlarge the polygons
            # AND associate a sufficient number of objects without incurring in polygons intersecting.
            if best_found is None: return None

            # A suitable buffer has been found: return it.
            high_buffer, high_list_objs, init_multipoly = best_found
            return left, high_buffer, high_list_objs, init_multipoly


        # 2 - We hit a buffer that does not make polygons intersect.
        cand_list_objs, cand_multipoly = res

        # 2.1 - First valid upper bound with count >= target found.
        if cand_list_objs.size >= target_num_objs:
            return last_valid_buffer, probe_buffer, cand_list_objs, cand_multipoly

        # 2.2 - Still below target, and still no intersections: keep exponential expansion.
        last_valid_buffer = probe_buffer
        high_list_objs = cand_list_objs
        init_multipoly = cand_multipoly
        probe_buffer = (probe_buffer + 1.0) * _GROWTH


def gen_multiregion_hotspot(base_multipolygon : MultiPolygon, rtree_stops : SpatialIndex, stop_uid_values : np.ndarray,
                            target_num_objs : int, min_stops_object : int) -> tuple[Polygon, np.ndarray] | None :
    """
    Generate a synthetic multi-region hotspot associated with approximately a target number
    of objects.
    """

    ### Apply an expanding buffer to the rotated polygon, until it becomes associated with the
    ### desired number of objects.

    # 1 - Set the lower bound for the buffer. This is equal to buffer needed to make one polygon in the multipolygon
    #     collapse to a point, which requires to find the minimum 'maximum_inscribed_circle' across the polygons
    #     in the multipolygon.
    low_buffer = -min([maximum_inscribed_circle(poly).length for poly in base_multipolygon.geoms])


    # print(f"DEBUG: Initializing high buffer...")
    # 2 - Initial expansion of the upper bound, until we go beyond the target number of objects to associate
    # #   or until a pair.
    res = init_upper_bound_multiregion(base_multipolygon, rtree_stops, stop_uid_values,
                                       target_num_objs, min_stops_object, low_buffer)
    
    # Find out if it is impossible to build an hotspot from the selected regions because 
    # we can't expand them AND associate a sufficient number of objects without the expanded regions
    # intersecting.
    if res is None :
        # print("DEBUG: INIT: impossible to build a multiregion hotspot from the considered regions!") 
        return None
    
    low_buffer, high_buffer, best_list_objs, best_multipoly = res
    # print(f"DEBUG: Initial high buffer: {high_buffer}, low buffer: {low_buffer}, initial num objs high buffer: {best_list_objs.size}")


    # 3 - Binary search for the smallest buffer with count >= target_objs. 
    best_diff_target = abs(target_num_objs - best_list_objs.size)
    while (high_buffer - low_buffer) > _TOL:
        
        # Find out how many objects associate with the polygon buffered with 'mid_buffer'.
        mid_buffer = 0.5 * (low_buffer + high_buffer)
        # print(f"DEBUG: Current interval: [{low_buffer},{high_buffer}], mid_buffer: {mid_buffer}")
        res = eval_buffer_multiregion(base_multipolygon, rtree_stops, stop_uid_values,
                                      mid_buffer, min_stops_object)

        # Check if res is None: if this is the case, then it means that one of the polygons in the
        # multipolygon collapsed due to negative buffer erosion. Set the lower bound to the search
        # value that has been used, and try in the new interval.
        if res is None :
            # print(f"DEBUG: Invalid multipolygon during binary search: {mid_buffer}")
            low_buffer = mid_buffer
            continue

        # Update the boundaries of the search interval.
        mid_list_objs, mid_multipoly = res
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
            best_multipoly, best_list_objs, best_diff_target = mid_multipoly, mid_list_objs, diff_target
        

        # If we have associated exactly the number of desired objects, exit the loop.
        if diff_target == 0 : break


    # Return the best buffered multipolygon found, as well as the number of objects it is associated with.
    return best_multipoly, best_list_objs


def rotate_translate_nointersect_multipoly(list_polygons : list[Polygon]) -> MultiPolygon | None :
    MAX_NUM_ATTEMPTS = 5

    # For each polygon in the multipolygon, apply a random rotation and translation.
    check = False
    count = 0
    new_mp = None
    while not check :
        # print(f"DEBUG: Transforming and turning the polys into a multipoly, attempt {count}...")
        
        # Try to find a suitable set of non-intersecting polygons within 'MAX_NUM_TRIES' iterations.
        if count >= MAX_NUM_ATTEMPTS : return None

        new_mp = []
        for polygon in list_polygons :

            # Determine the width and height of the original polygon.
            minx, miny, maxx, maxy = polygon.bounds
            w, h = maxx - minx, maxy - miny

            # Rotate the original polygon by a random amount of degrees.
            new_polygon = affinity.rotate(polygon,
                                          angle=random.uniform(0, 360),
                                          use_radians=False)
            
            # Translate the rotated polygon in a random 2D direction but not too far,
            # i.e., the x and y offsets are chosen according to the original polygon's bbox.
            new_polygon = affinity.translate(new_polygon,
                                            xoff=random.uniform(-0.5 * w, 0.5 * w),
                                            yoff=random.uniform(-0.5 * h, 0.5 * h))
            
            # Append the transformed polygon.
            new_mp.append(new_polygon)
            

        # Check if the transformed polygons do not intersect.
        check = all(a.disjoint(b) for a, b in combinations(new_mp, 2))
        count += 1
    

    # If we arrive here, it means we have a suitable multipolygon.
    return MultiPolygon(new_mp)


def sample_regions_multiregion_hotspot(df_polygons : gpd.GeoDataFrame,
                                       map_uid_blocks : pd.DataFrame,
                                       seed_uids : np.ndarray,
                                       num_regions_per_hotspot) -> tuple[list, MultiPolygon] :
    
    multipoly = None
    while multipoly is None :
        # Pick a random uid to be used as the seed for the multiregion hotspot.
        seed_uid = seed_uids[random.randint(0, seed_uids.size - 1)]
        # print(f"DEBUG: Seed uid: {seed_uid}")

        # Pick 'num_regions_per_hotspot' regions from those with which 'seed_uid' is associated with.
        id_regions = map_uid_blocks.loc[seed_uid].sample(num_regions_per_hotspot).index.to_list()
        # print(f"DEBUG: list regions: {list_regions}")

        # Retrieve the geometries of these regions.
        list_regions = df_polygons.loc[id_regions].geometry.to_list()
        # print(f"DEBUG: List polygons regions: {list_regions}")

        # Try to generate a multipolygon from the regions' geometries
        multipoly = rotate_translate_nointersect_multipoly(list_regions)

    return multipoly


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


def gen_unfair_datasets_multiregion_hotspots(df_polygons : gpd.GeoDataFrame, 
                                             map_uid_blocks : pd.DataFrame,
                                             stop_uid_values : np.ndarray, 
                                             rtree_stops : SpatialIndex,
                                             num_unfair_datasets : int, 
                                             num_hotspots_per_dataset : int,
                                             num_regions_per_hotspot : int,
                                             num_objs_per_hotspot : int,
                                             global_pos_rate : float, hotspots_pos_rate : float) -> list[tuple[list[Polygon],
                                                                                                            list[list[np.ndarray]],
                                                                                                            np.ndarray]] :
    """
    Generate multiple synthetic datasets containing unfair multiregion hotspots.
    """

    # For each uid, determine in how many blocks it has at least a stop segment. Useful when building hotspots made of
    # distinct regions, as we can use uids that have at least a stop segment in more than one block as 'seeds' to build such
    # hotspots.
    seed_uids = map_uid_blocks.groupby(level='uid').size()

    # Count the total number of objects.
    tot_num_objs = seed_uids.index.nunique()
    print(tot_num_objs)

    # Select the user IDs that have at least a stop centroid in 'num_regions_per_hotspot' or more separate regions.
    seed_uids = seed_uids.loc[seed_uids >= num_regions_per_hotspot].index.to_numpy()
    # print(f"DEBUG: Seed uids: {seed_uids}")


    list_unfair_datasets = []
    # for _ in tqdm(range(num_unfair_datasets), desc="Generating unfair datasets of labels with multiregion hotspots...") :
    for _ in range(num_unfair_datasets) :        

        # Generate the hotspots using the sampled and transformed polygons, associating each of them with 
        # 'target_num_objs' objects.
        list_multipolys_hotspots = []
        list_lists_objs_hotspots = []
        for _ in range(num_hotspots_per_dataset) :

            # Generate an hotspot made of separate 'num_regions_per_hotspot' regions. The loop below attempts to create
            # one until the hotspot's regions do not intersect the regions of previously created hotspots.
            while True :
                # Pick 'num_regions_per_hotspot' separate regions randomly from 'df_polygons', 
                # and then transform each of them applying a rotation and translation of their polygons. The sampling process
                # may be repeated multiple times, until we get a suitable number of polygons that do not intersect and can serve
                # as the basis for the multi-region hotspot.
                multipolygon_base = sample_regions_multiregion_hotspot(df_polygons,
                                                                       map_uid_blocks, 
                                                                       seed_uids, 
                                                                       num_regions_per_hotspot)
                # print(f"DEBUG: Base multipolygon: {multipolygon_base}")
                # break
                
                # Generate an hotspot made of 'num_regions_per_hotspot' separate regions.
                res = gen_multiregion_hotspot(multipolygon_base, rtree_stops, stop_uid_values, 
                                              num_objs_per_hotspot, min_stops_object=1)
                
                # If it is not possible to build an hotspot from the sampled and transformed regions,
                # we have to discard this multipolygon.
                if res is None : continue
                    
                # We have succesfully built a multiregion hotspot!
                multipoly_hotspot, list_objs_hotspot = res
                # print(f"DEBUG: multiregion hotspot built! Need to check if it can coexist with previous ones.")
                
                # Now we need to check that the new hotspot's multipolygon does not intersect those of the previously
                # created hotspots.
                if not all(multipoly_hotspot.disjoint(other) for other in list_multipolys_hotspots) :
                    # print(f"DEBUG: this multiregion hotspot intersects a previously built one, discard it.")
                    continue
                
                # If we arrive here, it means the multiregion hotspot is OK and does not intersect the previously
                # created ones! Break out the loop.
                # print(f"DEBUG: this multiregion hotspot is ok! Num.objs: {list_objs_hotspot.size}")
                break
            
            # Hotspot successfully created! Append its information to the relevant lists.
            list_multipolys_hotspots.append(multipoly_hotspot) # Add the hotspot's multipoly.
            list_lists_objs_hotspots.append(list_objs_hotspot) # Add the object IDs associated with this multi-region hotspot.


        # Generate the dataset of labels, injecting unfairness in the objects associated with the hotspots.
        # print(f"DEBUG: generating unfair labels!")
        unfair_labels = gen_unfair_labels(tot_num_objs, list_lists_objs_hotspots, global_pos_rate, hotspots_pos_rate)

        # Add the unfair dataset to the list.
        unfair_dataset = (list_multipolys_hotspots, list_lists_objs_hotspots, unfair_labels)
        list_unfair_datasets.append(unfair_dataset)
        # break


    return list_unfair_datasets