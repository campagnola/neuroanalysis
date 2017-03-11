import numpy as np

def binomial_ci(p, n, alpha=0.05 ):
    """
    Two-sided confidence interval for a binomial test.
   
    Given n successes occurring out of p trials, what is the likely range of
    binomial distribution probabilities that could lead to this observation?
   
    Parameters
    ----------
    p : int
        The number of trials
    n : int
        The number of successful trials
    alpha : int
        The range of the confidence interval to return. (alpha=0.05 gives a 95% confidence interval)
    Credit: http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    """
    upper_fn = lambda c: scipy.stats.binom.cdf(p, n, c) - alpha
    lower_fn = lambda c: scipy.stats.binom.cdf(p, n, c) - (1.0 - alpha)
    return scipy.optimize.bisect(lower_fn, 0, 1), scipy.optimize.bisect(upper_fn, 0, 1)


def ragged_mean(arrays, method='pad'):
    """Return the mean of a list of arrays, where each array may have 
    different length.
    """
    arrays = arrays[:]
    
    if method == 'pad':
        # Pad all arrays with nan until we have a rectangular array
        max_len = max([len(a) for a in arrays])
        for j,arr in enumerate(arrays):
            if len(arr) < max_len:
                arrays[j] = np.empty(max_len, dtype=arr.dtype)
                arrays[j][:len(arr)] = arr
                arrays[j][len(arr):] = np.nan
    elif method == 'clip':
        min_len = min([len(a) for a in arrays])
        arrays = [arr[:min_len] for arr in arrays]
    else:
        raise ValueError("method must be 'pad' or 'clip'")
            
    # Stack into one array and return the nanmean
    return np.nanmean(np.vstack(arrays), axis=0)
