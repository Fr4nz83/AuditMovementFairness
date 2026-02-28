import numpy as np
import pickle

from joblib import Parallel, delayed
from scipy.special import xlog1py, xlogy
from tqdm import tqdm


class BernoulliSpatialScan:
    """
    Encapsulates the Monte Carlo hypothesis test used in notebook 7.
    """

    ### PROTECTED METHODS ###

    def _batch_max_likelihood_ratio(self, labels_objects: np.ndarray, 
                                    flat_ids: np.ndarray, indptr: np.ndarray, lenghts: np.ndarray,
                                    tot_sum_labels: int,
                                    logL0_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        

        # Gather labels for all ids, then sum per candidate via segmented reduction.
        flat_vals = labels_objects[flat_ids]
        inside_sum = np.add.reduceat(flat_vals, indptr[:-1]).astype(np.float32, copy=False)


        # Vectorized computation: for each candidate subset of cells, compute the positive rate of the objects
        # associated with the subset vs the positive rate of the other objects.
        # NOTE: we use np.divide with the `where` parameter to avoid divisions by zero.
        p, n = inside_sum, lenghts
        P, N = tot_sum_labels, labels_objects.size
        inside_positive_rate  = np.divide(p, n, out=np.zeros_like(p, dtype=np.float32), where=(n > 0))
        outside_positive_rate = np.divide(P - p, N - n, out=np.zeros_like(p, dtype=np.float32), where=((N - n) > 0))
        

        # Potentially numpy-unsafe computation of the log-likelihood under the alternative hypotesis.
        #logL1 = (p * np.log(inside_positive_rate)
        #         + (n - p) * np.log1p(-inside_positive_rate)
        #         + (P - p) * np.log(outside_positive_rate)
        #         + (N - n - (P - p)) * np.log1p(-outside_positive_rate))
        
        
        # Safe alternative computation of the log-L1 via scipy functions. 
        # NOTE: the log-likelihood is -inf when the positive rate is 0 or 1, which can happen when p==0 or p==n for the inside positive rate, 
        # or when P-p==0 or N-n-(P-p)==0 for the outside positive rate. This is not a problem per se, since we are interested in the likelihood
        # ratio, and if the likelihood under the alternative hypotesis is -inf, then the likelihood ratio will be 0, which is what we expect in
        # these cases.
        # valid = (n > 0) & (n < N) # optional: mask degenerate windows (n==0 or n==N)
        # logL1 = np.full_like(inside_positive_rate, -np.inf, dtype=np.float32)
        logL1 = ( xlogy(p, inside_positive_rate) + 
                xlog1py((n - p), -inside_positive_rate) +
                xlogy((P - p), outside_positive_rate) +
                xlog1py((N - n - (P - p)), -outside_positive_rate) )

        # Vectorized computation of the log-likelihood ratio of the candidates
        # logLR = logL1 - logL0_max
        logL1 -= logL0_max
        maxLogLR = float(np.nanmax(logL1))

        return inside_positive_rate, outside_positive_rate, logL1, maxLogLR


    def _reject_H0(self, vec_max_LR : np.ndarray, max_LR_dataset : float) :
        '''
        This function determines if the null H_0 must be rejected or not, according to the test statistic's empirical
        distribution in 'vec_max_LR' and the test statistic's value in 'max_LR_dataset' computed from the real labels. 
        '''
        
        # Determine where the max LR computed with the original labels fall in the empirical test statistic's distribution.
        rank = np.count_nonzero(vec_max_LR >= max_LR_dataset)

        # Monte Carlo p-value of the observed test statistic's value derived from the ranked empirical test statistic's distribution 
        # (right tail), with +1 correction to include max_LR_dataset itself.
        p_value = (rank + 1) / (self.num_simulations + 1)

        # Based on the distribution and the real data we have, decide if we have to reject H_0.
        reject_H0 = p_value <= self.alpha

        # DEBUG: a few prints...
        print(f"Position in sorted MC sample: {rank}/{self.num_simulations} (extreme if below position {int(self.num_simulations * self.alpha)})")
        print(f"Monte Carlo p-value: {p_value:.6f}")
        print(f"Decision: {'Reject H0' if reject_H0 else 'Do NOT reject H0'}")

        return reject_H0


    ### PUBLIC CLASS CONSTRUCTOR ###

    def __init__(self,
                 num_simulations: int, alpha: float,
                 flat_ids : np.ndarray, indptr : np.ndarray, lengths : np.ndarray) :
        
        # Set some parameters for the Bernoulli-based spatial scan statistic
        self.num_simulations = int(num_simulations)
        self.alpha = float(alpha)

        ### Store the references to the data structures holding the flattened objects ID lists associated with the candidates ###
        self.flat_ids, self.indptr, self.lengths = flat_ids, indptr, lengths
        

        # Perform some sanity checks.
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.indptr.ndim != 1 or self.lengths.ndim != 1:
            raise ValueError("indptr and lengths must be 1D arrays")
        if self.indptr.size != self.lengths.size + 1:
            raise ValueError("indptr must have size len(lengths) + 1")
        if not np.all(self.lengths == np.diff(self.indptr)):
            raise ValueError("lengths/indptr mismatch")
        if not np.all(self.lengths > 0):
            raise ValueError("All candidates must have at least one object")


    def sequential_simulations(self, labels : np.ndarray) -> tuple[bool, np.ndarray, np.float32] :
        ### SEQUENTIAL VERSION ###

        # 1 - Compute the L_0 likelihood, which models the likelihood of observing the labels in the data under the the assumption that the 
        # null hypotesis H_0 is true, i.e., there is a single global distribution that governs the labels. L_0 is constant across 
        # permutations, since it depends only on the total number of positive and negative labels in the dataset, which does not change
        # when shuffling the original labels.
        P = labels.sum(dtype=np.uint32) # Constant across permutations
        N = labels.size
        rho = P / N
        logL0_max = P * np.log(rho) + (N - P) * np.log1p(-rho)


        # 2 - Compute the approximated distribution of the test statistic using shuffled labels.
        vec_max_LR = np.empty(self.num_simulations, dtype=np.float32)
        for i in tqdm(range(self.num_simulations)):    
            # Shuffle the original labels assigned to the objects. This represents the null hypotesis H_0, according to which
            # there is a single global distribution that governs the labels, i.e., there is not one or more sets of geographical regions
            # in which the associated objects have an average positive rate that is significantly different than that of the other objects. 
            rng = np.random.default_rng(i)
            shuffled_labels = rng.permutation(labels)

            # For the objects associated with each subset of cells, compute their positive rate vs that of the other objects.
            _, _, _, vec_max_LR[i] = self._batch_max_likelihood_ratio(shuffled_labels,
                                                                      self.flat_ids, self.indptr, self.lengths,
                                                                      P, logL0_max)
            

        # 3 - Compute the max log likelihood ratio from the candidates when considering the original labels.
        _, _, _, max_LR_dataset = self._batch_max_likelihood_ratio(labels,
                                                                   self.flat_ids, self.indptr, self.lengths,
                                                                   P, logL0_max)



        # 4 - Determine if we have to reject H_0 or not.
        reject = self._reject_H0(vec_max_LR, max_LR_dataset)


        return reject, vec_max_LR, max_LR_dataset


    def parallel_simulations(self, labels : np.ndarray) :

        ### JOBLIB HELPER FUNCTION ###
        def one_simulation(i, labels, flat_ids, indptr, lenghts, P, logL0_max):
            rng = np.random.default_rng(i)
            shuffled_labels = rng.permutation(labels)
            return self._batch_max_likelihood_ratio(shuffled_labels, flat_ids, indptr, lenghts, P, logL0_max)[3]


        ### JOBLIB MAIN CODE ###
        # 1 - Compute the L_0 likelihood, which models the likelihood of observing the labels in the data under the the assumption that the 
        # null hypotesis H_0 is true, i.e., there is a single global distribution that governs the labels. L_0 is constant across 
        # permutations, since it depends only on the total number of positive and negative labels in the dataset, which does not change
        # when shuffling the original labels.
        P = labels.sum(dtype=np.uint32) # Overall number of objects with positive labels.
        N = labels.size                 # Overall number of objects.
        rho = P / N
        logL0_max = P * np.log(rho) + (N - P) * np.log1p(-rho)

        # 2 - Compute the simulations' max log-LRs in parallel.
        vec_max_LR = Parallel(n_jobs=-1,
                              backend="loky",
                              verbose=10,
                              #max_nbytes="1M",   # threshold that triggers auto-memmapping
                              mmap_mode="r")(delayed(one_simulation)(i, labels, self.flat_ids, self.indptr, self.lengths, P, logL0_max) for i in range(self.num_simulations))
        vec_max_LR = np.asarray(vec_max_LR, dtype=np.float32)
                

        # 3 - Compute the max log likelihood ratio from the candidates when considering the original labels.
        _, _, _, max_LR_dataset = self._batch_max_likelihood_ratio(labels,
                                                                   self.flat_ids, self.indptr, self.lengths,
                                                                   P, logL0_max)



        # 4 - Determine if we have to reject H_0 or not.
        reject = self._reject_H0(vec_max_LR, max_LR_dataset)


        return reject, vec_max_LR, max_LR_dataset
    


    ### PUBLIC STATIC METHODS

    @staticmethod
    def load_flattened_candidates(path_dict_candidates : str) :
        '''
        Helper function that loads the flattened candidates (plus aux information); it also
        performs some sanity checks.
        '''
        
        with open(path_dict_candidates, "rb") as f:
            data = pickle.load(f)

        ### Load the dictory containing the flattened objects ID lists associated with the candidates ###
        flat_ids, indptr, lenghts = data['flat_ids'], data['start_pos'], data['lengths']
        del data

        # Ensure the big arrays are contiguous (helps memmap efficiency)
        flat_ids = np.ascontiguousarray(flat_ids)
        indptr   = np.ascontiguousarray(indptr)
        lenghts  = np.ascontiguousarray(lenghts)


        # SANITY CHECK: check that there is no candidate with 0 associated objects. 
        # It shouldn't happen, but we do a quick check.
        assert np.all(lenghts == np.diff(indptr)), "lenghts/indptr mismatch (or empty segments present)"
        assert np.all(lenghts > 0), "Candidates with zero associated objects detected, should not happen!"

        return flat_ids, indptr, lenghts