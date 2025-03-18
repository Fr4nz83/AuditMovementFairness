from rtree import index
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import math
import folium
import multiprocessing
from functools import partial


#### The dataframe should have columns lat, lon, label
def load_data(filename):
    '''
    This function creates a dataframe from a csv dataset. The dataset's rows are expected to have three 
    columns: lat, lon, label. Essentially, it contains the outcomes that a classifier gave to individuals
    in space w.r.t. a target variable.
    '''
    
    df = pd.read_csv(filename, index_col=0)
    df.reset_index(drop=True, inplace=True)

    return df


def get_stats(df, label):
    '''
    This function counts the number of examples, and those that have a positive label.
    Here positive is considered "1", different values are considered negative.
    '''
    N = len(df)
    P = df.loc[df[label] == 1, label].count()

    return N, P


def create_rtree(df):
    '''
    Builds an r-tree that indexes the points representing the examples in df.
    
    TODO: The rtree creation can be hugely optimized by using bulk loading when instantiating
          the rtree.
    '''
    
    # First, create an empty r-tree index
    rtree = index.Index()

    # Then, insert the points in the index, one by one.
    # There are far more optimized ways to do it, but the library used by the authors seems limited.
    for idx, row in df.iterrows():
        left, bottom, right, top = row['lon'], row['lat'], row['lon'], row['lat']
        rtree.insert(idx, (left, bottom, right, top))
    
    return rtree


def create_rtree_v2(df):
    '''
    Builds an r-tree that indexes the points representing the examples in df
    using bulk loading for better efficiency.
    '''
    
    # Create an iterable of (id, (left, bottom, right, top)) tuples
    items = (
        (idx, (row['lon'], row['lat'], row['lon'], row['lat']), None) for idx, row in df.iterrows()
    )
    # print("Number of items to bulk-load:", len(list(items)))
    
    # Build the r-tree in one step using bulk loading
    rtree = index.Index(items)
    
    return rtree


def filterbbox(df, min_lon, min_lat, max_lon, max_lat):
    '''
    This function selects the points within the specified bounding box.
    
    NOTE: Can be probably simplified using geopandas.
    '''
    
    df = df.loc[df['lon'] >= min_lon]
    df = df.loc[df['lon'] <= max_lon]
    df = df.loc[df['lat'] >= min_lat]
    df = df.loc[df['lat'] <= max_lat]
    df.reset_index(drop=True, inplace=True)    
    
    return df


def get_true_types(df, label):
    '''
    This function is used to manage the LAR dataset, and appears to remap the label 3 of 
    the target variable to 0, thus reducing the number of classes to two.
    '''
    
    array = np.array(df[label].values.tolist())
    array[array==3] = 0 ## replace entries with label 3 to have label 0 (for the LAR dataset)
    return array


def get_random_types(N, P):
    '''
    The function numpy.random.binomial(n, p, size=None) generates random samples from a binomial distribution, representing the number of successes in n independent trials, each with a success probability p. The size parameter determines the number N of sets of trials. 
    The way the authors use the function below, means that we have N sets of just 1 trial, and in that trial the probability of success is N/P. So, this means that across the N sets of trials, a fraction of ~P sets will be successful.
    
    This function is used when performing simulations.
    '''
    
    return np.random.binomial(size=N, n=1, p=P/N)



def get_simple_stats(points, types):
    '''
    Computes the fraction of positive labels p/n.
    '''
    
    n = len(points)
    p = types[points].sum()
    
    if n>0:
        rho = p/n
    else:
        rho = np.nan

    return (n, p, rho)


def compute_pos_rate(points, types):
    '''
    Same as above.
    '''
    
    n = len(points)
    p = types[points].sum()
    return p/n


def id2loc(df, point_id):
    '''
    Get the coordinates of a point id.
    '''
    
    lat = df.loc[[point_id]]['lat'].values[0]
    lon = df.loc[[point_id]]['lon'].values[0]
    return (lat, lon)



def query_range_box(df, rtree, xmin, xmax, ymin, ymax):
    '''
    Perform a range query within the r-tree.
    '''
    
    left, bottom, right, top = xmin, ymin, xmax, ymax

    result = list( rtree.intersection((left, bottom, right, top)) )

    return result


def query_range(df, rtree, center, radius):
    '''
    Retrieves the location of a point (via its ID), then defines a square buffer around it (with side length equal to twice the radius). Then compute a range query between this square within the r-tree.
    '''
    
    lat, lon = id2loc(df, center)

    left, bottom, right, top = lon - radius, lat - radius, lon + radius, lat + radius
    result = list( rtree.intersection((left, bottom, right, top)) )

    return result


def query_nn(df, rtree, center, k):
    '''
    Performs a k-NN query within the r-tree.
    '''
    
    lat, lon = id2loc(df, center)
    return list(rtree.nearest( [lon, lat], k))


def create_seeds(df, rtree, n_seeds):
    '''
    Given a set of points, cluster them and then consider the clusters' centroids. Then, for each
    centroid find the nearest point in the r-tree: this point will be used as a seed.
    '''
    
    # Compute clusters over the points in df with k-means.
    X = df[['lon', 'lat']].to_numpy()
    kmeans = KMeans(n_clusters=n_seeds, n_init='auto').fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # For each cluste centroid, find the nearest point from the real data, and use it as seed.
    seeds = []
    for c in cluster_centers:
        seeds.append(list(rtree.nearest([c[0], c[1]], 1))[0])
 
    return seeds


def compute_max_likeli(n, p, N, P):
    '''
    Computes l1_max. Used in function 'compute_statistics' (see below)
    '''
    
    ## l1max =  p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
    ## handle extreme cases

    rho = P/N
    l0max = P*math.log(rho) + (N-P)*math.log(1-rho) # Equation in page 488 of paper.


    ### Compute l^1_{max} ###

    # Case in which a region contains zero points or it represents the entire space: l1_max collapses to l0_max.
    if n == 0 or n == N: ## rho_in == 0/0 or rho_out == 0/0
        l1max = l0max
        return l1max


    rho_in = p/n            # Fraction (probability) of positive examples within a given region.  Corresponds to \rho_0 in the paper.
    rho_out = (P-p)/(N-n)   # Fraction (probability) of positive examples outside a given region. Corresponds to \rho_1 in the paper.

    # Manage the various corner cases, which occur when we have a logarithm of 0, which is undefined.
    ## Case 1: rho_in == 0, thus we have a log(0).
    if p == 0: l1max = P*math.log(rho_out) + (N-n - P)*math.log(1-rho_out)
    ## Case 2: a region contains only positives, and it contains ALL the positives in the space. 
    #          In this case, rho_in == 1, rho_out == 0, and thus from the general formula we would have 0 * inf + 0 * inf = 0
    elif p == n and p == P: l1max = 0
    ## Case 3: rho_in == 1, hence p*math.log(rho_in) + (n-p)*math.log(1-rho_in) = 0 because we have a log(1) == 0.
    elif p == n: l1max = (P-p)*math.log(rho_out) + (N-P)*math.log(1-rho_out)
    ## Case 4: rho_out == 0, hence (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out) because we have a log(1) == 0
    elif p == P: l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in)
    # DEFAULT normal case: 
    else: l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)

    return l1max



def compute_statistic(n, p, N, P, direction='both', verbose=False):
    '''
    Computes the difference (l1max – l0max), which is the (logarithm of the) likelihood ratio statistic.

    The parameter 'direction' lets you test one-sided hypotheses (for example, “inside rate < outside rate” or vice versa) versus a two-sided alternative.
    
    Purpose: This statistic is used to detect regions where the observed rate deviates significantly from the overall rate—indicative of potential spatial unfairness.
    '''
    
    ## l1max - l0max

    if verbose:
        print(f'{n=}, {p=}')


    if n == 0 or n == N: ## rho_in == 0/0 or rho_out == 0/0
        return 0 


    rho = P/N
    rho_in = p/n
    rho_out = (P-p)/(N-n)

    if verbose:
        print(f'{rho=}, {rho_in=}, {rho_out=}')
    
    l0max = P*math.log(rho) + (N-P)*math.log(1-rho)    

    # Case 1: One-sided hypotesis test in which only interested in cases where the inside positive rate is lower than the outside positive rate,
    # i.e., a region is being disadvantaged w.r.t. the rest. 
    # NOTE: Not used in the source code.
    if direction == 'less_in':
        ### inside < outside
        if rho_in < rho_out:
            l1max = compute_max_likeli(n, p, N, P)
        else:
            l1max = l0max
    # Case 2: One-sided hypotesis test in which only interested in cases where the inside positive rate is higher than the outside positive rate, i.e.,
    # a region is being advantaged w.r.t. the rest. 
    # NOTE: Not used in the source code.
    elif direction == 'less_out':
       ### inside > outside
        if rho_in > rho_out:
            l1max = compute_max_likeli(n, p, N, P)
        else:
            l1max = l0max
    # Case 3: Two-sided hypotesis test.
    # NOTE: this is the only option used in the source code.
    else:
        ### inside != outside
        l1max = compute_max_likeli(n, p, N, P)

    # This difference represents the computation of the logarithmic max likelihood ratio.
    statistic = l1max - l0max

    if verbose:
        print(f'{l0max=}, {l1max=}, {statistic=}')

    return statistic 




def create_regions(df, rtree, seeds, radii):
    '''
    Given a set of seed point IDs and a list of radii, this function creates candidate regions. For each seed and radius, it queries the spatial index (using query_range) to get all points in that region and packages the information into a dictionary. Note that each region is a square, centered on a point, with side
    equal to the radius.
    Purpose: To generate many regions whose fairness (or lack thereof) can be audited.
    '''
    
    regions = []
    for seed in seeds:
        for radius in radii:
            # Retrieve the points within the ball of radius 'radius' centered on 'seed'.
            points = query_range(df, rtree, seed, radius)
            region = {
                'points' : points,
                'center' : seed,
                'radius' : radius,  
            }
            regions.append(region)
    
    return regions


def scan_regions(regions, types, N, P, direction='both', verbose=False):
    '''
    Iterates over the candidate regions, computes the likelihood statistic for each (using get_simple_stats and compute_statistic), and returns:

    - The region with the highest statistic.
    - The maximum statistic value found.
    - The list of statistics for all regions.

    Purpose: To “scan” the geographical space for the most anomalous (or unfair) region.
    '''
    
    statistics = []

    # For each region R, compute:
    # 1) the number of points in the region, n
    # 2) the number of points with a positive label, p
    # 3) the fraction of points with a positive label, rho = p/n.
    # 4) The likelihood ratio computed as log(\frac{l_1^{max}(R)}{l_0^{max}}) = log(l_1^{max}(R)) - log(l_0^{max})
    for region in regions:
        # Computes the fraction of positive labels p/n.
        n, p, rho = get_simple_stats(region['points'], types)
        statistics.append(compute_statistic(n, p, N, P, direction=direction))
    

    # Find the region with the highest likelihood ratio, i.e., the region which contradicts the most
    # the null hypotesis for which a region is spatially fair.
    # NOTE: This will be used compared against the likelihood ratios achieved by the MonteCarlo simulation 
    # implemented in 'scan_alt_worlds'.
    idx = np.argmax(statistics)
    max_likelihood = statistics[idx]


    # DEBUG: print info on screen if verbose is True.
    if verbose:
        print('range', np.amin(statistics), np.amax(statistics))
        print('max likelihood', max_likelihood)

        # NOTE: The print below concerns the region with the highest max likelihood ratio,
        #       but it was already commented in the original source.
        #n, p, rho = get_simple_stats(regions[idx]['points'], types)
        # print(f"at ({regions[idx]['center']}, {regions[idx]['radius']})" )
        #compute_statistic(n, p, N, P, direction=direction, verbose=verbose)
    
    return regions[idx], max_likelihood, statistics



def scan_alt_worlds(n_alt_worlds, regions, N, P, verbose=False):
    """ returns all alt worlds sorted by max likelihood, and the max likelihood """
    
    # Below we conduct hypotesis testing on #n_alt_worlds simulated worlds.
    alt_worlds = []
    for _ in range(n_alt_worlds):
        # Step 1: assign an outcome to the N examples, according to the binomial distribution where the positive label has probability P.
        #         Recall that the probability P of success is computed by looking at all the space, and that we are assuming that the 
        #         prob. distribution behind the labels is binomial.  
        alt_types = get_random_types(N, P)
        
        # Step 2: conduct the hypotesis test.
        # NOTE: the 'direction' input parameter here is set implicitly to 'both'.
        alt_best_region, alt_max_likeli, _ = scan_regions(regions, alt_types, N, P, verbose=verbose)
        
        # Step 3; take note of the labels assigned in this simulation, the max likelihood ratio, and the region behind it.
        alt_worlds.append((alt_types, alt_best_region, alt_max_likeli))

    # Sort the results from the simulations, according to their max likelihood ratio.
    alt_worlds.sort(key=lambda x: -x[2])

    return alt_worlds, alt_worlds[0][2]


def simulation(regions, N, P, verbose, _):
    alt_types = get_random_types(N, P)
    alt_best_region, alt_max_likeli, _ = scan_regions(regions, alt_types, N, P, verbose=verbose)
    return (alt_types, alt_best_region, alt_max_likeli)

def scan_alt_worlds_parallel(n_alt_worlds, regions, N, P, verbose=False):
    # Use a process pool to run the simulations in parallel.
    with multiprocessing.Pool() as pool:
        # Use partial to fix the other arguments for our helper.
        func = partial(simulation, regions, N, P, verbose)
        alt_worlds = pool.map(func, range(n_alt_worlds))
    
    # Sort the results and return the highest max likelihood ratio.
    alt_worlds.sort(key=lambda x: -x[2])
    return alt_worlds, alt_worlds[0][2]


def get_signif_threshold(signif_level, n_alt_worlds, regions, N, P, parallel = False):
    """ 
    Returns a statistic value such any region with statistic above that value is unfair at significance level `signif_level`; i.e., has p-value lower than `signif_level`  
    """
    
    # Step 1 - Conduct 'n_alt_worlds' simulations. For each of them, we find the max likelihood ratio. We store and sort all these ratios
    #          in a list: this list represents an empirical distribution of the max likelihood ratio
    alt_worlds, _ = scan_alt_worlds_parallel(n_alt_worlds, regions, N, P) if parallel else scan_alt_worlds(n_alt_worlds, regions, N, P)
    k = int(signif_level * n_alt_worlds)
    signif_thresh = alt_worlds[k][2] ## get the max likelihood at position k

    return signif_thresh



######## partioning-based scan

def scan_partitioning(regions, types):
    '''
    Rather than using likelihood ratios, this function computes a “score” for each region based on the squared difference between its positive rate and the mean rate across all regions. It returns the region with the maximum score and the array of scores.
    
    The purpose is to offer an alternative (partitioning-based) metric for detecting regions with anomalous rates.
    '''
    
    rhos = []
    for region in regions:
        n = len(region['points'])
        p = types[region['points']].sum()
        if n>0:
            rho = p/n
        else:
            rho = np.nan
        rhos.append(rho)
    mean_rho = np.nanmean(rhos)
    rhos = np.array(rhos)
    # print('mean_rho', mean_rho)
    # print(rhos[:10])
    
    
    # For each region, we compute the squared difference between its positive rate and the mean rate across all regions.
    scores = (rhos - mean_rho)**2
    
    # Find the score and ID of the region yielding the maximum score.
    max_score = np.nanmax(scores)
    idx = np.nanargmax(scores)

    # print(scores[:10])
    # print(np.nanmax(scores), np.nanargmax(scores) )
    return regions[idx], max_score, scores



######## create synthetic datasets #######

def create_points(n, rho):
    '''
    Generates a synthetic dataset of n points with random (x, y) coordinates (each uniformly drawn from [0, 1]). It also creates a binary types array where exactly n \cdot rho of the points are positive. (The function adjusts the initially random draws to guarantee the exact number of positives.)

    Purpose: To allow testing and validation of the auditing procedures on controlled synthetic data.
    '''
    
    points = []
    types = np.random.binomial(size=n, n=1, p=rho)
    
    # guarantee n * rho positives
    n_pos = int(n * rho)
    while np.sum(types) != n_pos:
        idx = np.random.randint(0,n)
        if np.sum(types) > n_pos and types[idx]==1:
            types[idx] = 0
        elif np.sum(types) < n_pos and types[idx]==0:
            types[idx] = 1

    for i in range(n):
        x = random.random()
        y = random.random()
        points.append((x,y))
        
    return points, types




######## draw map functions (used to plot maps with Folium) ######

def show_grid_region(df, grid_info, types, region):

    lon_min = grid_info['lon_min']
    lon_max = grid_info['lon_max']
    lat_min = grid_info['lat_min']
    lat_max = grid_info['lat_max']

    lon_n = grid_info['lon_n']
    lat_n = grid_info['lat_n']


    i, j = region['grid_loc']

    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, prefer_canvas = True, tiles='cartodbpositron')

    # pos_group = folium.FeatureGroup("Positive")
    # neg_group = folium.FeatureGroup("Negative")

    lon_start = lon_min + (i/lon_n)*(lon_max - lon_min)
    lon_end = lon_min + ((i+1)/lon_n)*(lon_max - lon_min)

    lat_start = lat_min + (j/lat_n)*(lat_max - lat_min)
    lat_end = lat_min + ((j+1)/lat_n)*(lat_max - lat_min)

    n, p, rho = get_simple_stats(region['points'], types)

    folium.Rectangle([(lat_start,lon_start), (lat_end,lon_end)], tooltip=f'{n=}, {p=}, rho={rho:.2f}').add_to( mapit )


    for point in region['points']:
        if types[point] == 1:
            # pos_group.add_child(folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ))
            folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
        else:
            # neg_group.add_child(folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ))
            folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )

    # mapit.add_child(pos_group)
    # mapit.add_child(neg_group)
    mapit.fit_bounds([ [lat_start, lon_start],[lat_end, lon_end] ])
    return mapit



def show_grid_regions(df, grid_info, types, regions):

    lon_min = grid_info['lon_min']
    lon_max = grid_info['lon_max']
    lat_min = grid_info['lat_min']
    lat_max = grid_info['lat_max']

    lon_n = grid_info['lon_n']
    lat_n = grid_info['lat_n']


    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, prefer_canvas = True, tiles='cartodbpositron')

    for region in regions:

        i, j = region['grid_loc']


        lon_start = lon_min + (i/lon_n)*(lon_max - lon_min)
        lon_end = lon_min + ((i+1)/lon_n)*(lon_max - lon_min)

        lat_start = lat_min + (j/lat_n)*(lat_max - lat_min)
        lat_end = lat_min + ((j+1)/lat_n)*(lat_max - lat_min)

        n, p, rho = get_simple_stats(region['points'], types)

        folium.Rectangle([(lat_start,lon_start), (lat_end,lon_end)], tooltip=f'{n=}, {p=}, ρ={rho:.2f}').add_to( mapit )

        for point in region['points']:
            if types[point] == 1:
                folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
            else:
                folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )

    mapit.fit_bounds([(lat_min, lon_min), (lat_max, lon_max)])
    return mapit



def show_circular_region(df, types, region):

    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, prefer_canvas = True, tiles='cartodbpositron')

    r = region['radius'] * 111320 ## roughly convert diff in lat/lon to meters

    n, p, rho = get_simple_stats(region['points'], types)

    folium.Circle(location=id2loc(df, region['center']), color='#0000FF', fill_color='#0000FF', fill=True, opacity=0.4, fill_opacity=0.4, radius=r, tooltip=f'{n=}, {p=}, rho={rho:.2f}' ).add_to( mapit )
    
    for point in region['points']:
        if types[point] == 1:
            folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
        else:
            folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
    
    return mapit


def show_circular_regions(df, types, regions):
    
    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, prefer_canvas = True, tiles='cartodbpositron')

    for region in regions:
        n, p, rho = get_simple_stats(region['points'], types)
        
        # r = region['radius'] * 111320 ## roughly convert diff in lat/lon to meters
        # folium.Circle(location=id2loc(df, region['center']), color='#0000FF', fill_color='#0000FF', fill=True, opacity=0.4, fill_opacity=0.4, radius=r, tooltip=f'{n=}, {p=}, rho={rho:.2f}' ).add_to( mapit )
        
        r = region['radius']
        c = id2loc(df, region['center'])
        folium.Rectangle([(c[0]-r, c[1]-r), (c[0]+r, c[1]+r)], tooltip=f'{n=}, {p=}, ρ={rho:.2f}').add_to( mapit )

        for point in region['points']:
            if types[point] == 1:
                folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
            else:
                folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
        
    return mapit

