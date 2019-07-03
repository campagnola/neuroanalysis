from __future__ import division, print_function
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from .data import Trace, PatchClampRecording
from .filter import bessel_filter
from .baseline import mode_filter, adaptive_detrend


def detect_evoked_spike(data, pulse_edges, **kwds):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    Parameters
    ==========
    data : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times (seconds relative to the timing specified in the recording) of the stimulation pulse. 
    """
    trace = data['primary']
    if data.clamp_mode == 'vc':
        return detect_vc_evoked_spike(trace, pulse_edges, **kwds)
    elif data.clamp_mode == 'ic':
        return detect_ic_evoked_spike(trace, pulse_edges, **kwds)
    else:
        raise ValueError("Unsupported clamp mode %s" % trace.clamp_mode)


def rc_decay(t, tau, Vo): 
    """function describing the deriviative of the voltage.  If there
    is no spike one would expect this to fall off as the RC of the cell. """
    return -(Vo/tau)*np.exp(-t/tau)


def detect_ic_evoked_spike(trace, pulse_edges, dvdt_threshold=0.0013, mse_threshold=80.):

    assert trace.data.ndim == 1
#    trace = bessel_filter(trace, 20e3)
#    trace = trace.resample(sample_rate=20000)
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)
    
    dt = trace.dt
    pstart, pstop = trace.index_at(pulse_edges[0]), trace.index_at(pulse_edges[1])
    # get indicies for a window within the pulse to look in (note these are not index of the pulse)
    pulse_window_start = pstart + int(.0003 / dt)
    pulse_window_end = pstop - int(.00005 / dt)

    # derivatives of entire trace
    dvdt = np.diff(trace.data)
    d2vdt2 = np.diff(dvdt)
    #d2vdt2 = adaptive_detrend(d2vdt2, window=(pulse_window_start, pulse_window_end))

    plt.figure()
    plt.plot(dvdt, 'r')

    # mask out pulse termination artifact
    index_mask = int(np.round(2.e-5/dt))
    dvdt[(pstop-index_mask+1) : (pstop+index_mask+1)] = np.nan


    plt.plot(dvdt, 'b')


    chunk = d2vdt2[pulse_window_start:pulse_window_end]
    max_d2vdt2_idx = np.argmax(chunk) + pulse_window_start
    d2vdt2_threshold = np.std(chunk)*2.

    if d2vdt2[max_d2vdt2_idx] > d2vdt2_threshold:
        #find max dvdt nearby
        max_dvdt_index = np.argmax(dvdt[max_d2vdt2_idx : max_d2vdt2_idx + int(35.e-5 / dt)]) + max_d2vdt2_idx 

        # if d2vdt2 crosses threshold make sure the dvdt crosses it's threshold
        if dvdt[max_dvdt_index] > dvdt_threshold:
            spike_index = max_dvdt_index + 1 #+1 due to translating dvdt time to v time (off by 1)
        else:
            spike_index = None
    else:
        spike_index = None
    

    #if no spike was found in the pulse region check to see if there is a spike in the pulse termination region
    if spike_index is None:
        #note that this is using the dvdt with the termination artifact in it to locate where it should start 
        chunk = trace.data[pstop : pstop + int(.0005 / dt)]
        
        # #find location of minimum dv/dt
        min_dvdt_idx = np.argmin(np.diff(chunk)) + pstop  #global minimum index
    
        # create a vector to fit
        dvtofit=np.diff(trace.data)[min_dvdt_idx + 1:] #+1 here to get rid of actual artifact, 
        ttofit=trace.time_values[(min_dvdt_idx + 1 + 1):] - trace.time_values[(min_dvdt_idx + 1 + 1)] # setting time to start at zero, note: +1 because time trace of derivative needs to be one shorter

        # do fit and see if it matches
        popt, pcov = curve_fit(rc_decay, ttofit, dvtofit, maxfev=10000)
        fit = rc_decay(ttofit, *popt)
        mse = (np.sum((dvtofit-fit)**2))/len(fit)*1e10 #mean squared error
        plt.figure()
        plt.plot(ttofit, dvtofit, 'r')
        plt.plot(ttofit, fit, 'k--', label=('mse: %f' % (mse)))
        plt.legend()   
        if mse > mse_threshold:
            # note this is looking for the max of the data with the termination artifact removed.
            max_dvdt_idx = np.argmax(dvdt[pulse_window_end : pulse_window_end + int(3./dt)]) + pulse_window_end  #TODO: potential problem if this goes past end of trace
            spike_index = max_dvdt_idx   
        else:
            spike_index =  None

    plt.figure(figsize =  (10, 8))
    voltage= trace.data
    time_ms = trace.time_values*1.e3
    ax1=plt.subplot(111)
    ax1.plot(time_ms, voltage)
    ax2=ax1.twinx()
    ax2.plot(time_ms[1:], dvdt, color='r')
    ax2.plot(time_ms[2:], d2vdt2, color='g')
    ax2.plot([time_ms[pulse_window_start],time_ms[pulse_window_end]], [d2vdt2_threshold, d2vdt2_threshold], '--k', lw=.5)    
    plt.axvspan(time_ms[pulse_window_start],time_ms[pulse_window_end], color='k', alpha=.3 )
    if spike_index:
        plt.axvline(time_ms[spike_index], color='k')
    plt.xlabel('ms')
    plt.title('dt '+str(trace.dt))
    plt.show()



    # find voltage peak within 1 ms after spike initiation
    if spike_index:
        chunk = trace.data[spike_index: (spike_index +  int(.001 / dt))]
        peak_idx =  np.argmax(chunk) + spike_index    
        return {'peak_time': trace.time_at(peak_idx), 'max_dvdt_time': trace.time_at(spike_index),
            'peak_val': trace.data[peak_idx], 'max_dvdt': trace.data[spike_index]}
    else:
        return None
    



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
    if len(smooth) == 0:
        raise ValueError("Invalid pulse indices [%d->%d:%d] for data (%d)" % (pulse_edges[0], pstart, pstop, len(trace.data)))
    peak_ind = np.argmin(smooth)
    if peak_ind == 0 or peak_ind == len(smooth)-1:
        # no local minimum found within pulse edges
        return None
    
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
