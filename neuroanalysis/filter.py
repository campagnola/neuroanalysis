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
    filtered = butterworthFilter(trace.data, dt=trace.dt, wStop=w_stop, wPass=w_pass, gPass=g_pass, gStop=g_stop, order=order, btype=btype, bidir=bidir)
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

    Parameters
    ----------
    trace : Trace instance
        Data to be filtered.
    edges : list of (start, stop) tuples
        Specifies the indices of regions in the trace to remove.
    window : float
        Window duration (in seconds) on either side of each removed
        chunk that will be used to determine the values used to fill
        the chunk.

    Returns
    -------
    A copy of *trace* with artifacts removed.

    Notes
    -----
    For each (start, stop) pair in *edges*, three windows are used:

    1. The window in [start:stop] that will have its values replaced
    2. The window immediately before (1), of width determined by the *window* argument
    3. The window immediately after (1), of width determined by the *window* argument

    The values in (1) are replaced by performing a linear regression on the data in
    (2) and (3), then filling (1) with the resulting extrapolated line.
    """
    data = trace.data.copy()
    t = trace.time_values
    w = int(window / trace.dt)
    
    # merge together overlapping regions
    edges.sort()
    merged_edges = [edges[0]]
    for on, off in edges[1:]:
        on1, off1 = merged_edges[-1]
        if on < off1:
            merged_edges[-1] = (min(on, on1), max(off, off1))
        else:
            merged_edges.append((on, off))

    # remove and replace with linregress
    for on, off in merged_edges:
        on = int(on)
        off = int(off)
        chunkx = t[on-w:off+w]
        chunky = data[on-w:off+w]
        mask = np.ones(len(chunkx), dtype='bool')
        mask[w:w+(off-on)] = False
        chunkx = chunkx[mask]
        chunky = chunky[mask]
        slope, intercept = scipy.stats.linregress(chunkx, chunky)[:2]
        data[on:off] = slope * t[on:off] + intercept
    
    return trace.copy(data=data)
        