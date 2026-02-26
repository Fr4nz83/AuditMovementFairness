import pandas as pd
import numpy as np
from itertools import combinations


class CandidateGeneration() :

    ### CLASS CONSTRUCTOR ###

    def __init__(self, path_users_cells_mapping : str) :

        # Read the mapping users<->cells.
        self.map_users_cells = pd.DataFrame(pd.read_pickle(path_users_cells_mapping))
        # print(f"DEBUG: self.map_users_cells: {self.map_users_cells}")



    ### PROTECTED METHODS ###    

    def _create_augmented_grid(self) -> pd.DataFrame :
        '''
        Compute some aggregations at cell-level within the mapping 'users <-> cells', effectively creating an augmented version
        of the grid. 
        '''

        # Compute some aggregations at cell-level, effectively creating an augmented version of the grid. 
        stats_config = {'list_users' : pd.NamedAgg(column='uid', aggfunc=set),
                        #'num_users' : pd.NamedAgg(column='uid', aggfunc='nunique')
                        }
        aug_grid = (self.map_users_cells.reset_index()
                                        .groupby('cell_id')
                                        .agg(**stats_config))


        # Sort the dataframe by cell IDs -- this effectively sort the dataframe's index.
        aug_grid.sort_values(by='cell_id', ascending=True, inplace=True)
        # print(f"Augmented grid df: {aug_grid}")

        return aug_grid
    

    def _gen_candidates_level(self, res_intersections : pd.DataFrame, cnt_threshold : int = 1) -> pd.DataFrame :
        
        # Find out the cardinality of the combinations of cells contained in res_intersections
        dim_itemset = res_intersections.index.nlevels
        # print(f"dim_itemset: {dim_itemset}")


        intersections = {}
        # Base case: the combinations of cells in 'res_intersections' are actually single cells.
        if dim_itemset == 1 :
            # print("Managing the case in which we are generating candidates from single cells.")
            
            pairs = zip(res_intersections.index, res_intersections['list_users'])
            for (cell_id, list_users), (other_cell_id, other_list_users) in combinations(pairs, 2):
                
                # Compute the set intersection, and its cardinality.
                intersection = list_users & other_list_users

                # Add to the dictionary only the cell pairs that have at least 'threshold' users in common.
                # The threshold should be calculated according to the statistical power we want to have in the hypotesis tests.
                if (len(intersection) >= cnt_threshold) : intersections[(cell_id, other_cell_id)] = intersection


        # Case in which the combinations of cells in 'res_intersections' have size 'dim_itemset' > 1.
        else :
            # print("Managing the case in which we are generating candidates from combinations of cells.")

            base_lvls = list(range(dim_itemset - 1)) if dim_itemset > 2 else 0
            print(f"base_lvls: {base_lvls}")

            for keys, sub in res_intersections.loc[:, 'list_users'].groupby(level=base_lvls, sort=False) :
                # If the multi-index has size 2, then the first "n-1" levels is just one level, and thus isn't a tuple.
                if (dim_itemset == 2) : keys = (keys,)

                # Drop the first 'n-1' levels in the multi-index, leaving only the last one.
                s = sub.droplevel(base_lvls)
                # print(f"Series associated with the group {keys}: {s}")
                

                # Generate all the possible tuples of length 'n+1' by keeping fixed the first "n-1" keys and 
                # consider all the possible combinations of length 2 that can be generated in the last level of the
                # multi-index.
                pairs = zip(s.index, s.values)
                for (cell_id, list_users), (other_cell_id, other_list_users) in combinations(pairs, 2):
                    
                    # Compute the set intersection, and its cardinality.
                    intersection = list_users & other_list_users

                    # Add to the dictionary only the cell pairs that have at least 'threshold' users in common.
                    # The threshold should be calculated according to the statistical power we want to have in the hypotesis tests.
                    if len(intersection) >= cnt_threshold : intersections[(*keys, cell_id, other_cell_id)] = intersection


        # Finally, output the results in the appropriate format. 
        return pd.Series(data = intersections, name='list_users').to_frame()



    ### PUBLIC METHODS ###
    
    def candidate_generation(self, cnt_threshold : int) -> list[pd.DataFrame] :

        # 1 - Create an augmented grid: this is the candidate generation starting point.
        augmented_grid = self._create_augmented_grid()
        # print(f"Augmented grid df: {augmented_grid}")

        
        ### CANDIDATE FILTERING AND GENERATION ###

        # 2 - Set up the initial state of 'res_intersections', i.e., the dataframe that will be used to
        # progressively generate the subsets of cells candidated to be statistically checked. It initially
        # contains candidates of size '1 cell'.
        res_intersections = augmented_grid.copy(deep = True)
        res_intersections['num_users'] = res_intersections['list_users'].apply(len)
        res_intersections = res_intersections[res_intersections['num_users'] >= cnt_threshold] # Eliminate the cells with < 'cnt_threshold' users.
        del res_intersections['num_users']
        # print(f"DEBUG: Initial state of res_intersection df: {res_intersections}")

        # 3 - Check and generation loop.
        list_candidates_test = []
        l = 1
        while True :
            # 1 - Add the candidates from the previous level...
            list_candidates_test.append(res_intersections)

            # 2 - Generate the set of candidates for the current level...
            print(f"Generating candidates of size '{l+1} cells' from viable candidates of size '{l} cells'...")
            res_intersections = self._gen_candidates_level(res_intersections, cnt_threshold)
            # print(f"DEBUG: Candidates generated: {type(res_intersections)} {res_intersections}")
            print(f"Number of candidates with at least {cnt_threshold} associated objects found: {len(res_intersections)}")

            # 3 - Check if we haven't generate more candidates: if so, exit the loop.
            if len(res_intersections) == 0 : 
                print("No more viable candidates generated; exiting the candidate generation loop!")
                break

            l += 1
        print(f"Total number of candidates of length up to {l} cells generated: {sum(len(df) for df in list_candidates_test)}")


        # Return a single dataframe containing the candidates computed at the various levels of the generation process.
        list_candidates_test = pd.concat(list_candidates_test)
        
        # Turn the sets into the 'list_users' column into numpy arrays, so to further speed up the use of np.take()
        # during the Monte Carlo computations in the subsequent steps of our approach.
        # Also, enforce the use of unsigned 32-bit integers to reduce the memory footprint.
        list_candidates_test['list_users'] = list_candidates_test['list_users'].apply(lambda s: np.array(list(s), dtype=np.uint32))

        return list_candidates_test