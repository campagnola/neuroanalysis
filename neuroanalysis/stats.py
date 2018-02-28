import numpy as np
import scipy.optimize
import scipy.stats


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
    alpha : float
        The range of the confidence interval to return. (alpha=0.05 gives a 95% confidence interval)
    
    Credit: http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    """
    if p == n:
        return (np.nan, np.nan)
    upper_fn = lambda c: scipy.stats.binom.cdf(p, n, c) - alpha
    lower_fn = lambda c: scipy.stats.binom.cdf(p, n, c) - (1.0 - alpha)
    return scipy.optimize.bisect(lower_fn, 0, 1), scipy.optimize.bisect(upper_fn, 0, 1)


def binomial_sliding_window(x, success, window, spacing=None, alpha=0.05):
    """Given a set of success/failure events occurring at different positions,
    measure the probability of success versus position using a sliding window.
    Also generate confidence intervals on the probability of a binomial
    distribution.
    
    This function is used primarily to measure the probability of synaptic
    connection as it varies with the distance between cells. 

    Parameters
    ----------
    x : float array
        X values of each observation. For cell-cell connectivity, this is the
        distance between cells.
    success : bool array
        True/False indicating success or failure for each observation
    window : float
        The width of the sliding window within which success probability and
        confidence intervals are computed
    spacing : float
        Distance to advance window for each step.
    alpha : float
        Width of confidence interval (alpha=0.05 gives 95% ci)
    
    Returns
    -------
    xvals : array
        Center x value of each window step
    proportion : array
        Proportion of successful trials in each window step
    lower : array
        Lower binomial confidence interval value at each window step
    upper : array
        Upper binomial confidence interval value at each window step
    """
    if spacing is None:
        spacing = window / 4.0
        
    xvals = np.arange(window / 2.0, 500e-6, spacing)
    upper = []
    lower = []
    prop = []
    ci_xvals = []
    for x1 in xvals:
        minx = x1 - window / 2.0
        maxx = x1 + window / 2.0
        # select points inside this window
        mask = (x >= minx) & (x <= maxx)
        pts_in_window = success[mask]
        # compute stats for window
        n_probed = pts_in_window.shape[0]
        n_conn = pts_in_window.sum()
        if n_probed > 0:
            #prop.append(np.nan)
        #else:
            prop.append(n_conn / n_probed)
            ci = binomial_ci(n_conn, n_probed, alpha=alpha)
            lower.append(ci[0])
            upper.append(ci[1])
            ci_xvals.append(x1)
    
    return ci_xvals, prop, lower, upper


def ragged_mean(arrays, method='clip'):
    """Return the mean of a list of arrays, where each array may have 
    different length.
    
    Parameters
    ----------
    arrays : list
        A list of arrays to be averaged
    method : "clip" | "pad"
        If "clip", then the arrays are truncated to the minimum length.
        If "pad", then the arrays are all padded to the maximum length with NaN.
    """
    assert len(arrays) > 0
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


def weighted_std(values, weights):
    """Return the weighted standard deviation of *values*.

    Source: http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return variance**0.5
