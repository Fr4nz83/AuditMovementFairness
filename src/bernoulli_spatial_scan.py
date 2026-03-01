import numpy as np
import pickle
import math

from tqdm import tqdm
import numba as nb


### NUMBA FUNCTIONS ###

@nb.njit(cache=True, nogil=True)
def max_loglr_streaming(is_simulation : bool,
                        labels : np.ndarray, 
                        flat_ids : np.ndarray, indptr : np.ndarray, lengths : np.ndarray, 
                        P : np.uint32, logL0_max : np.float32):
    """
    'Numbized' computation of the log-likelihood ratios of candidate subsets of cells, associated with some set of grids.
    For each candidate subset, the function computes:
    
    - (inside) positive rate of the objects associated with the candidate;
    - (outside) positive rate of the other objects;
    - candidate log-likelihood ratio (`logL1 - logL0_max`)
    
    The function ultimately returns:
    
    - the maximum log-likelihood ratio across candidates.
    - the inside positive rates of the candidates (if not in simulation mode)
    - the outside positive rates of the candidates (if not in simulation mode)
    - the log likelihood ratios of the candidates (if not in simulation mode)

    NOTE: in simulation mode we avoid allocating and updating some arrays because this achieves to
          further speed up the computations.

    Parameters
    ----------
    is_simulation : bool
        Boolean flag indicating if this function must operate in simulation mode or not.
    labels : np.ndarray
        Binary labels (0/1) for all objects.
    flat_ids : np.ndarray
        Flattened object indices for all candidates.
    indptr : np.ndarray
        Candidate start offsets in `flat_ids`.
    lenghts : np.ndarray
        Number of objects per candidate.
    P : int
        Total number of positive labels in `labels`.
    logL0_max : float
        Log-likelihood under H0 (global Bernoulli model). Doesn't change when permuting the labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        `(inside_positive_rate, outside_positive_rate, logLR_per_candidate, max_logLR)`.
    """

    N = labels.size
    max_lr = -np.inf
    m = lengths.size
    lr_candidates = np.empty(m if not is_simulation else 0)
    in_rate_candidates = np.empty_like(lr_candidates)
    out_rate_candidates = np.empty_like(lr_candidates)
    for i in range(m):
        start = indptr[i]
        end   = indptr[i + 1]

        # segmented sum without allocating flat_vals
        p = 0
        for k in range(start, end):
            p += labels[flat_ids[k]]

        n = lengths[i]
        if (n <= 0) or (n >= N) : continue  # degenerate candidate (no object or contains all objects)

        # Positive rate for the objects associated with the candidate, and for the objects that are not.
        inside_rate  = p / n
        outside_rate = (P - p) / (N - n)

        # compute logL1 safely (xlogy/xlog1py equivalents)
        logL1 = p * math.log(inside_rate) if p > 0 else 0.0
        logL1 += (n - p) * math.log1p(-inside_rate) if n - p > 0 else 0.0
        #
        Pout = P - p
        Nout = N - n
        logL1 += Pout * math.log(outside_rate) if Pout > 0 else 0.0
        #
        neg_out = Nout - Pout
        logL1 += neg_out * math.log1p(-outside_rate) if neg_out > 0 else 0.0

        # Compute the log-likelihood ratio.
        lr = logL1 - logL0_max
        
        # Add the ratio to the vector of log-LRs of the candidates.
        if (not is_simulation) : 
            lr_candidates[i] = lr
            in_rate_candidates[i], out_rate_candidates[i] = inside_rate, outside_rate

        # Update the candidate with the largest log-LR found so far.
        if lr > max_lr:
            max_lr = lr

    return in_rate_candidates, out_rate_candidates, lr_candidates, max_lr


@nb.njit(cache=True, nogil=True, parallel=True)
def compute_simulations(num_sims : np.uint32, labels : np.ndarray, 
                        flat_ids : np.ndarray, indptr : np.ndarray, lengths : np.ndarray,
                        P : np.uint32, logL0_max : np.float32):
    '''
    This numba function is in charge of computing a given number of simulations in parallel.
    To this end, it wraps the numba function 'max_loglr_streaming' and collects the results it produces.

    Parameters
    ----------
    num_sims : np.uint32
        Number of simulations to be performed.
    labels : np.ndarray
        Binary labels (0/1) for all objects.
    flat_ids : np.ndarray
        Flattened object indices for all candidates.
    indptr : np.ndarray
        Candidate start offsets in `flat_ids`.
    lenghts : np.ndarray
        Number of objects per candidate.
    P : np.uint32
        Total number of positive labels in `labels`.
    logL0_max : np.float32
        Log-likelihood under H0 (global Bernoulli model). Doesn't change when permuting the labels.

    Returns
    -------
    max_logLR : float
        The maximum log-likelihood ratio found during the simulations.
    '''
    
    vec_max_LR = np.empty(num_sims, dtype=np.float32)
    for s in nb.prange(num_sims) :
        # Shuffle the labels.
        np.random.seed(s)
        shuffled_labels = labels.copy()
        np.random.shuffle(shuffled_labels)

        # Compute the max log-LR distribution expected under the assumption that H_0 is true.
        vec_max_LR[s] = max_loglr_streaming(True,
                                            shuffled_labels, 
                                            flat_ids, indptr, lengths,
                                            P, logL0_max)[-1]

    return vec_max_LR



class BernoulliSpatialScan:
    """
    Bernoulli-based spatial scan statistic for binary labels.

    This class evaluates the candidate subset of cells of a given set of grids and computes, for each candidate,
    the log-likelihood ratio between:
    - H0: the hypothesis that exists just one label distribution over all objects
    - H1: the hypothesis that exists at least one candidate in which we have a label distribution that is different
          than the distribution over the labels of the other objects.

    The test statistic is thus the maximum log-likelihood ratio.
    """

    ### PROTECTED METHODS ###

    def _reject_H0(self, vec_max_LR : np.ndarray, max_LR_dataset : float) :
        """
        Decide whether to reject H0 from the Monte Carlo null distribution according to
        the test statistic's approximated distribution under H_0, the test statistic's
        observed with the 'original' labels, and the chosen significance level 'alpha'.

        Computes the right-tail Monte Carlo p-value using +1 correction:
        `(rank + 1) / (num_simulations + 1)`, where `rank` counts simulated
        max log-LR greater than or equal to the one computed with the 'original' labels.

        Parameters
        ----------
        vec_max_LR : np.ndarray
            Simulated maximum log-likelihood ratios under shuffled labels, i.e.,
            the test statistic's approx distribution under the assumpyion that H_0 is true.

        max_LR_dataset : float
            Observed maximum log-likelihood ratio from original labels.

        Returns
        -------
        bool
            `True` if `p_value <= alpha`, otherwise `False`.
        """
        
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
        """
        Initialize scan-test configuration and candidate indexing structures.

        Parameters
        ----------
        num_simulations : int
            Number of label permutations used to estimate the null distribution.
        alpha : float
            Significance level in `(0, 1)`.
        flat_ids : np.ndarray
            Flattened object ID lists for all candidates.
        indptr : np.ndarray
            Candidate boundaries in `flat_ids`.
        lengths : np.ndarray
            Number of objects per candidate.

        Raises
        ------
        ValueError
            If simulation parameters or candidate-index arrays are invalid.
        """
        
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


    def sequential_simulations(self, labels : np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, np.float32] :
        """
        Run the Bernoulli-based spatial scan statistic with the simulations computed sequentially.

        Steps:
        1. Compute `logL0_max` (global Bernoulli distribution under the null model).
        2. Conduct `num_simulations` with shuffled-label datasets and collect simulated
        maxima of candidate log-likelihood ratios to compute the approx test statistic's distribution under H_0.
        The simulations are computed sequentially.
        3. Compute candidate log-likelihood ratios with the original labels.
        4. Decide whether to reject H0 based on where the max log-LR with the original labels falls within the
           approx test statistic's distribution.

        Parameters
        ----------
        labels : np.ndarray
            Binary labels (0/1) for all objects.

        Returns
        -------
        tuple[bool, np.ndarray, np.ndarray, np.float32]
            `(reject_H0, simulated_max_logLR, observed_logLR_per_candidate, observed_max_logLR)`.
        """

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
            vec_max_LR[i] = max_loglr_streaming(True, shuffled_labels,
                                                self.flat_ids, self.indptr, self.lengths,
                                                P, logL0_max)[-1]
            

        # 3 - Compute the max log likelihood ratio from the candidates when considering the original labels.
        _, _, dist_lr_dataset, max_LR_dataset = max_loglr_streaming(False,labels,
                                                                    self.flat_ids, self.indptr, self.lengths,
                                                                    P, logL0_max)



        # 4 - Determine if we have to reject H_0 or not.
        reject = self._reject_H0(vec_max_LR, max_LR_dataset)


        return reject, vec_max_LR, dist_lr_dataset, max_LR_dataset


    def parallel_simulations(self, labels : np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, np.float32] :

        """
        Run the Bernoulli-based spatial scan statistic with the simulations computed in parallel.

        Steps:
        1. Compute `logL0_max` (global Bernoulli distribution under the null model).
        2. Generate `num_simulations` with shuffled-label datasets and collect simulated
        maxima of candidate log-likelihood ratios to compute the approx test statistic's distribution under H_0.
        The simulations are computed in parallel.
        3. Compute candidate log-likelihood ratios with the original labels.
        4. Decide whether to reject H0 based on where the max log-LR with the original labels falls within the
           approx test statistic's distribution.

        Parameters
        ----------
        labels : np.ndarray
            Binary labels (0/1) for all objects.

        Returns
        -------
        tuple[bool, np.ndarray, np.ndarray, np.float32]
            `(reject_H0, simulated_max_logLR, observed_logLR_per_candidate, observed_max_logLR)`.
        """

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
        vec_max_LR = compute_simulations(self.num_simulations, labels, 
                                         self.flat_ids, self.indptr, self.lengths, 
                                         P, logL0_max)
                

        # 3 - Compute the max log likelihood ratio from the candidates when considering the original labels.
        _, _, dist_lr_dataset, max_LR_dataset = max_loglr_streaming(False, labels,
                                                                    self.flat_ids, self.indptr, self.lengths,
                                                                    P, logL0_max)



        # 4 - Determine if we have to reject H_0 or not.
        reject = self._reject_H0(vec_max_LR, max_LR_dataset)


        return reject, vec_max_LR, dist_lr_dataset, max_LR_dataset
    


    ### PUBLIC STATIC METHODS

    @staticmethod
    def load_flattened_candidates(path_dict_candidates : str) :
        """
        Load flattened candidate representation from a pickle file.

        The pickle is expected to contain a dictionary with the following keys:
        - `flat_ids`: candidates' flattened object ID lists
        - `start_pos`: candidate pointer array
        - `lengths`: candidate sizes (number of objects associated with the candidate)

        Parameters
        ----------
        path_dict_candidates : str
            Path to the pickle containing flattened candidate data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            `(flat_ids, indptr, lengths)`.
        """
        
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