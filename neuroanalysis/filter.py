from pyqtgraph.flowchart.library.functions import besselFilter, butterworthFilter
from scipy.signal import savgol_filter



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
    wlen = int(window_duration / trace.dt)
    filtered = savgol_filter(trace.data, window_length=wlen, **kwds)
    return trace.copy(data=filtered)
    
