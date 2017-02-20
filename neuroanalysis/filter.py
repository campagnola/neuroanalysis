import numpy as np
import scipy.stats
from pyqtgraph.flowchart.library.functions import besselFilter, butterworthFilter


def bessel_filter(trace, cutoff, order=1, btype='low', bidir=True):
    """Return a Bessel-filtered copy of a Trace.
    """
    filtered = besselFilter(trace.data, dt=trace.dt, cutoff=cutoff, order=order, btype=btype, bidir=bidir)
    # todo: record information about filtering?
    #filtered.meta['processing'].append({'name': 'bessel_filter', 'cutoff': cutoff, 'order': order, 'btype': btype, 'bidir': bidir})
    return trace.copy(data=filtered)


def butterworth_filter(trace, w_pass, w_stop=None, g_pass=2.0, g_stop=20.0, order=1, btype='low', bidir=True):
    """Return a Butterworth-filtered copy of a Trace.
    """
    filtered = besselFilter(trace.data, dt=trace.dt, wStop=w_stop, wPass=w_pass, gPass=g_pass, gStop=g_stop, order=order, btype=btype, bidir=bidir)
    return trace.copy(data=filtered)


def savgol_filter(trace, window_duration, **kwds):
    """Return a Savitsky-Golay-filtered copy of a Trace.
    """
    from scipy.signal import savgol_filter  # only recently available
    wlen = int(window_duration / trace.dt)
    filtered = savgol_filter(trace.data, window_length=wlen, **kwds)
    return trace.copy(data=filtered)
    

def remove_artifacts(trace, edges, window):
    """Remove selected regions from a trace and fill with a flat line.
    """
    data = trace.data.copy()
    t = trace.time_values
    w = window / trace.dt
    for on, off in edges:
        chunkx = t[on-w:off+w]
        chunky = data[on-w:off+w]
        mask = np.ones(len(chunkx), dtype='bool')
        mask[w:w+(off-on)] = False
        chunkx = chunkx[mask]
        chunky = chunky[mask]
        slope, intercept = scipy.stats.linregress(chunkx, chunky)[:2]
        print on, off, w, slope, intercept
        data[on:off] = slope * t[on:off] + intercept
    return trace.copy(data=data)
        