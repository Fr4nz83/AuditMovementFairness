import pandas as pd
import numpy as np
from itertools import combinations


class CandidateGenerationClassification() :
# TODO: one day we can actually create two different subclasses, one dealing with the classification case and the other with regression.
#       We cover just the classification case for now.

    ### CLASS CONSTRUCTOR ###

    def __init__(self, path_users_labels : str, path_users_cells_mapping : str) :
        
        self.users_labels = pd.read_parquet(path_users_labels)
        # print(self.users_labels)

        # Read the mapping users<->cells. Then, add the user labels to the dataframe.
        self.map_users_cells = pd.DataFrame(pd.read_pickle(path_users_cells_mapping))
        self.map_users_cells['label'] = self.users_labels['label']
        # print(self.map_users_cells)


        # Assuming that we are dealing with the classification case, and that the labels follow a Bernoullian distribution, 
        # determine the global positive and negative rates on the users labels.
        self.global_positive_rate = self.users_labels['label'].mean()
        # print(f"Global positive rate: {self.global_positive_rate} - global negative rate: {1 - self.global_positive_rate}")

        # Remap IDs of cells and users to contiguous ranges.
        # TODO: ensure that the cell remapping is undone when we output the subsets of cells to be statistically tested
        #       for movement fairness.
        (self.remapped_indices_cells,
        self.remapped_indices_users,
        self.remap_users_cells,
        self.remap_user_labels) = self._remapping_indices()





    ### PROTECTED METHODS ###

    def _remapping_indices(self) :
        '''
        This method internally remaps the IDs of the cells and users to continuous ranges, in order to speedup
        subsequent operations.
        '''

        # Produce a more compact indexing for the cell IDs: these IDs can have gaps in them, so reindex their IDs
        # to prepare more efficient set intersections over the user IDs they refer to.
        array_cell_ids = np.sort(self.map_users_cells['cell_id'].unique())
        remapping_indices_cells = pd.Series(index = array_cell_ids, data = range(len(array_cell_ids)))
        # print(remapping_indices_cells)

        # Produce a more compact indexing for the user IDs. Same reason as above.
        array_user_ids = np.sort(self.map_users_cells.index.unique())
        remapping_indices_users = pd.Series(index = array_user_ids, data = range(len(array_user_ids)))
        # print(remapping_indices_users)


        # Reindex the mapping 'user <-> cells', ensuring that cell and user IDs fall in a continuous range.
        remap_users_cells = self.map_users_cells.copy(deep=True)
        remap_users_cells.index = remap_users_cells.index.map(remapping_indices_users) # Reindex user IDs
        remap_users_cells['cell_id'] = remap_users_cells['cell_id'].map(remapping_indices_cells) # Reindex cell IDs.
        # print(f"Remapped 'users <-> cells' df: {remap_users_cells}")

        # Reindex the mapping 'user <-> label', ensuring that user IDs fall in a continuous range.
        remap_user_labels = remap_users_cells.groupby('uid')['label'].first()
        # print(f"Remapped 'user <-> labels' df: {remap_user_labels}")


        return remapping_indices_cells, remapping_indices_users, remap_users_cells, remap_user_labels
    

    def _create_augmented_grid(self) :
        '''
        Compute some aggregations at cell-level within the mapping 'users <-> cells', effectively creating an augmented version
        of the grid. 
        '''

        # Compute some aggregations at cell-level, effectively creating an augmented version of the grid. 
        stats_config = {'list_users' : pd.NamedAgg(column='uid', aggfunc=set),
                        #'num_users' : pd.NamedAgg(column='uid', aggfunc='nunique'),
                        #'positive_rate' : pd.NamedAgg(column='label', aggfunc='mean')
                        }
        aug_grid = (self.remap_users_cells.reset_index()
                                          .groupby('cell_id')
                                          .agg(**stats_config))


        # Sort the dataframe by cell IDs -- this effectively sort the dataframe's index.
        aug_grid.sort_values(by='cell_id', ascending=True, inplace=True)
        # print(f"Augmented grid df: {aug_grid}")

        return aug_grid
    

    def _check_candidates_stat(self, res_intersections : pd.DataFrame, eps_threshold : float) -> list :
        '''
        This method takes in input a dataframe containing a subset of cells generated during the candidate generation step,
        and for each candidate if it needs to undergo statistical hypotesis testing. The method assumes that the dataframe
        is structured as follows: each row represents a subset of cells, referenced in the index by a tuple containing the IDs
        of the cells, and each value in the column "list_users" represents a set of user IDs associated with that subset of cells.
        '''
        
        # 1 - turn each user ID in a list element into a row.
        tmp = (res_intersections.loc[:, 'list_users']
               .explode(ignore_index=False)
               .rename('uid')
               .to_frame())
        # print(tmp)

        # 2 - For every user ID, find the associated predicted label.
        tmp['labels'] = tmp['uid'].map(self.remap_user_labels)
        # print(tmp)

        # 3- For every combination of cells found in tmp's index, compute the local positive rate according to the labels of the user IDs
        #    associated with that combination.
        res_intersections['positive_rate'] = tmp.groupby(level=list(range(tmp.index.nlevels)))['labels'].mean()
        del tmp
        # print(res_intersections)

        # 4 - Finally, determine which combinations of cells have a local positive rate that differs more than an
        # "eps_diff" threshold from the global one.
        combs_cells_tocheck = res_intersections[abs(res_intersections['positive_rate'] - self.global_positive_rate) > eps_threshold]
        print(f"Combinations of cells to test: {combs_cells_tocheck}")


        # 5 - Return the combinations in the form of list.
        return combs_cells_tocheck.index.to_list()
    

    def _gen_candidates_level(self, res_intersections : pd.DataFrame, cnt_threshold : int) :
        
        # Find out the cardinality of the combinations of cells contained in res_intersections
        dim_itemset = res_intersections.index.nlevels
        print(f"dim_itemset: {dim_itemset}")


        intersections = {}

        # Base case: the combinations of cells in 'res_intersections' are actually single cells.
        if dim_itemset == 1 :
            print("Managing the case in which we are generating candidates from single cells.")
            
            pairs = zip(res_intersections.index, res_intersections['list_users'])
            for (cell_id, list_users), (other_cell_id, other_list_users) in combinations(pairs, 2):
                
                # Compute the set intersection, and its cardinality.
                intersection = list_users & other_list_users

                # Add to the dictionary only the cell pairs that have at least 'threshold' users in common.
                # The threshold should be calculated according to the statistical power we want to have in the hypotesis tests.
                if (len(intersection) > cnt_threshold) : intersections[(cell_id, other_cell_id)] = intersection


        # Case in which the combinations of cells in 'res_intersections' have size 'dim_itemset' > 1.
        else :
            print("Managing the case in which we are generating candidates from combinations of cells.")

            base_lvls = list(range(dim_itemset - 1)) if dim_itemset > 2 else 0
            print(f"base_lvls: {base_lvls}")

            for keys, sub in res_intersections.loc[:, 'list_users'].groupby(level=base_lvls, sort=False) :
                # If the multi-index has size 2, then the first "n-1" levels is just one level, and thus isn't a tuple.
                if (dim_itemset == 2) : keys = (keys,)

                # Drop the first 'n-1' levels in the multi-index, leaving only the last one.
                s = sub.droplevel(base_lvls)
                print(f"Series associated with the group {keys}: {s}")
                

                # Generate all the possible tuples of length 'n+1' by keeping fixed the first "n-1" keys and 
                # consider all the possible combinations of length 2 that can be generated in the last level of the
                # multi-index.
                #
                # TODO: try to optimize with the 'pairs' trick, instead of accessing the series 's' with 'a' and 'b'.
                for a, b in combinations(s.index, 2):
                    # Compute the set intersection, and its cardinality.
                    intersection = s[a] & s[b]
                    cnt = len(intersection)

                    # Add to the dictionary only the cell pairs that have at least 'threshold' users in common.
                    # The threshold should be calculated according to the statistical power we want to have in the hypotesis tests.
                    if len(intersection) > cnt_threshold : intersections[(*keys, a, b)] = intersection


        # Finally, output the results in the appropriate format. 
        return pd.Series(data = intersections, name='list_users').to_frame()



    ### PUBLIC METHODS ###

    def get_global_positive_rate(self) -> float :
        return self.global_positive_rate
    
    def get_global_negative_rate(self) -> float :
        return 1 - self.global_positive_rate
    
    def candidate_generation(self, cnt_threshold : int, eps_threshold : float) -> list[tuple] :

        # Create an augmented grid: this will be the starting point of the generation of subset of cells
        # to be statistically tested for movement fairness.
        augmented_grid = self._create_augmented_grid()
        # print(f"Augmented grid df: {augmented_grid}")

        
        ### Candidate generation and selection phase ###

        # 1 - Set up 'res_intersections', i.e., the dataframe that will be used to progressively generate the candidates.
        res_intersections = augmented_grid.copy(deep = True)
        res_intersections['num_users'] = res_intersections['list_users'].apply(len)
        print(f"Initial state of res_intersection df: {res_intersections}")

        # 1.1 - Eliminate the cells that have less than 'cnt_threshold' users.
        res_intersections = res_intersections[res_intersections['num_users'] >= cnt_threshold]

        
        list_candidates_test = []
        level = 1
        # TODO: BEGIN do-while ...
        while True :
            # 1 - Check which candidates need to undergo statistical test...
            list_level_candidates = self._check_candidates_stat(res_intersections, eps_threshold)
            list_candidates_test.extend(list_level_candidates)
            print(f"Number of candidates of size {l} added to the list: {len(list_level_candidates)}")

            # 2 - Generate the set of candidates for the next level...
            print(f"Generating candidates of size {l+1}...")
            res_intersections = self._gen_candidates_level(res_intersections, cnt_threshold)
            print(res_intersections)

            # 3 - Check if we haven't generate more candidates: if so, exit the loop.
            if len(res_intersections) == 0 : 
                print("Exiting the candidate generation loop!")
                break

            l += 1