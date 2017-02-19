from __future__ import division, print_function
import numpy as np
from scipy.ndimage import gaussian_filter

from .data import Trace, PatchClampRecording


def detect_evoked_spike(data, pulse_edges, **kwds):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    Parameters
    ==========
    data : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (int, int)
        The start and end indices of the stimulation pulse. 
    """
    trace = data['primary']
    if data.clamp_mode == 'vc':
        return detect_vc_evoked_spike(trace, pulse_edges, **kwds)
    elif data.clamp_mode == 'ic':
        return detect_ic_evoked_spike(trace, pulse_edges, **kwds)
    else:
        raise ValueError("Unsupported clamp mode %s" % trace.clamp_mode)


def detect_ic_evoked_spike(trace, pulse_edges, threshold=-10e-3, duration=3e-3):
    assert trace.data.ndim == 1
    pulse_edges = tuple(map(int, pulse_edges))  # make sure pulse_edges is (int, int)
    
    dt = trace.dt
    w = int(duration / dt)
    pstart, pstop = pulse_edges
    chunk = trace.data[pstart:pstart+w]
    
    peak_ind = np.argmax(chunk)
    peak_val = chunk[peak_ind]
    if peak_val < threshold:
        return None
    
    dvdt = np.diff(chunk[:peak_ind])
    rise_ind = np.argmax(dvdt)
    max_dvdt = dvdt[rise_ind]
    
    return {'peak_index': peak_ind + pstart, 'rise_index': rise_ind + pstart,
            'peak_val': peak_val, 'max_dvdt': max_dvdt}


def detect_vc_evoked_spike(trace, pulse_edges, sigma=20e-6, delay=150e-6, threshold=50e-12):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    This function assumes that a square voltage pulse is used to evoke an unclamped spike
    in a voltage clamp recording, and that the peak of the unclamped spike occurs *during*
    the stimulation pulse.

    Parameters
    ==========
    trace : Trace instance
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (int, int)
        The start and end indices of the stimulation pulse. 
    sigma : float
        Time constant (sec) for a gaussian filter used to smooth the trace before 
        peak detection. default is 20 us.
    delay : float
        Time (sec) after the onset of the stimulation pulse to begin searching for the
        peak of the unclamped spike. This is used to exclude artifacts that can
        appear immediately after the pulse onset. Default is 150 us.
    threshold : float
        Threshold (amps) for detection of the unclamped spike peak (see Notes). Default
        is 50 pA.
        
    Notes
    =====
    
    This is a simple unclamped spike detection algorithm that compares the minimum
    pipette current value during the stimulation pulse to the final value before
    pulse offset. An unclamped spike is detected if the minimum value is less
    than the final value, and the difference is greater than the specified
    threshold. The rising phase of the action potential is determined by searching
    for a local minimum in the derivative of the pipette current immediately before
    the detected peak.    

    This algorithm generally performs well for spike detection, with few false
    positives or negatives. However, the max dv/dt measurement may be invalid for
    very small unclamped spikes, making the true latency of the spike difficult
    to measure.
    """
    if not isinstance(trace, Trace):
        raise TypeError("data must be PatchClampRecording or Trace instance.")
    
    assert trace.ndim == 1
    pulse_edges = tuple(map(int, pulse_edges))  # make sure pulse_edges is (int, int)
    
    dt = trace.dt

    # select just the portion of the trace that contains the pulse, excluding a short
    # delay after the pulse onset
    pstart = pulse_edges[0] + int(delay / dt)
    pstop = pulse_edges[1]

    # find the location of the minimum value during the pulse
    smooth = gaussian_filter(trace.data[pstart:pstop], int(sigma/dt))
    peak_ind = np.argmin(smooth)
    
    # a spike is detected only if the peak is at least 50pA less than the final value before pulse offset
    peak_diff = smooth[-1] - smooth[peak_ind]
    if peak_diff > threshold:
        
        # Walk backward to the point of max dv/dt
        dv = np.diff(smooth)
        rstart = max(0, peak_ind - int(1e-3/dt))  # don't search for rising phase more than 1ms before peak
        rise_ind = np.argmin(dv[rstart:peak_ind])
        max_dvdt = dv[rise_ind] / dt
        rise_ind += pstart + rstart

        return {'peak_index': peak_ind + pstart, 'rise_index': rise_ind,
                'peak_diff': peak_diff, 'max_dvdt': max_dvdt}
    else:
        return None
