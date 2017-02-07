"""
Baseline detection and detrending tools.


"""
from __future__ import division
import numpy as np
import scipy.stats, scipy.signal

    
def adaptive_detrend(data, x=None, threshold=3.0):
    """Linear detrend where the baseline is estimated excluding outliers."""
    if x is None:
        x = data.xvals(0)
    
    d = data.view(ndarray)
    
    d2 = scipy.signal.detrend(d)
    
    stdev = d2.std()
    mask = abs(d2) < stdev*threshold
    
    lr = scipy.stats.linregress(x[mask], d[mask])
    base = lr[1] + lr[0]*x
    d4 = d - base
    
    return d4
    

def float_mode(data, dx=None):
    """Returns the most common value from a floating-point array by binning
    values together.
    """
    if bins is None:
        bins = max(5, int(len(data)/10.))
    y, x = np.histogram(data, bins=bins)
    ind = np.argmax(y)
    mode = 0.5 * (x[ind] + x[ind+1])
    return mode


def mode_filter(data, window=500, step=None, bins=None):
    """Sliding-window mode filter."""
    d1 = data.view(np.ndarray)
    vals = []
    l2 = int(window/2.)
    if step is None:
        step = l2
    i = 0
    while True:
        if i > len(data)-step:
            break
        vals.append(mode(d1[i:i+window], bins))
        i += step
            
    chunks = [np.linspace(vals[0], vals[0], l2)]
    for i in range(len(vals)-1):
        chunks.append(np.linspace(vals[i], vals[i+1], step))
    remain = len(data) - step*(len(vals)-1) - l2
    chunks.append(np.linspace(vals[-1], vals[-1], remain))
    d2 = np.hstack(chunks)
    
    if (hasattr(data, 'implements') and data.implements('MetaArray')):
        return MetaArray(d2, info=data.infoCopy())
    return d2
    

def histogram_detrend(data, window=500, bins=50, threshold=3.0):
    """Linear detrend. Works by finding the most common value at the beginning and end of a trace, excluding outliers."""
    
    d1 = data.view(np.ndarray)
    d2 = [d1[:window], d1[-window:]]
    v = [0, 0]
    for i in [0, 1]:
        d3 = d2[i]
        stdev = d3.std()
        mask = abs(d3-np.median(d3)) < stdev*threshold
        d4 = d3[mask]
        y, x = np.histogram(d4, bins=bins)
        ind = np.argmax(y)
        v[i] = 0.5 * (x[ind] + x[ind+1])
        
    base = np.linspace(v[0], v[1], len(data))
    d3 = data.view(np.ndarray) - base
    
    return d3
