from __future__ import division, print_function
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile

from .data import Trace, PatchClampRecording
from .filter import bessel_filter
from .baseline import mode_filter, adaptive_detrend
from .event_detection import threshold_events


def detect_evoked_spike(data, pulse_edges, **kwds):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    This function simply wraps either detect_ic_evoked_spike or detect_vc_evoked_spike, depending on the clamp mode
    used in *data*.

    Parameters
    ==========
    data : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *data*. 
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


def detect_ic_evoked_spike(trace, pulse_edges, dvdt_threshold=0.0013, mse_threshold=40., show=False):
    """
    """
    if show:
        import pyqtgraph as pg
        w = pg.GraphicsLayoutWidget()
        plt1 = w.addPlot()
        plt2 = w.addPlot(row=1, col=0)
        plt2.setXLink(plt1)
        plt3 = w.addPlot(row=2, col=0)
        plt3.setXLink(plt1)
        plt4 = w.addPlot(row=3, col=0)
        plt4.setXLink(plt1)
        plt1.plot(trace.time_values, trace.data)
        w.resize(1000, 900)
        w.show()

    assert trace.data.ndim == 1
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)
    
    # get indicies for a window within the pulse to look in (note these are not index of the pulse)
    pulse_window = (pulse_edges[0] + 300e-6, pulse_edges[1] - 50e-6)

    # third derivative of trace
    diff1 = trace.time_slice(*pulse_edges).diff()
    diff2 = diff1.diff()

    print("==========================")
    print("pulse edges: %s" % repr(pulse_edges))

    # mask out pulse artifacts from diff2 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff2, center=edge + 100e-6, radius=400e-6, power=2)

    diff2 = bessel_filter(diff2, 20e3, order=4, bidir=True)
    diff3 = diff2.diff()

    # mask out pulse artifacts from diff3 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff3, center=edge, radius=400e-6, power=3)

    # lowpass 3rd derivative
    diff3 = bessel_filter(diff3, 10e3, order=8, bidir=True)

    threshold2 = 40e3  # would be better to measure this, but it's tricky in d2..
    threshold3 = scoreatpercentile(np.abs(diff3.time_slice(*pulse_window).data), 70)

    if show:
        plt2.plot(diff1.time_values, diff1.data)
        plt3.plot(diff2.time_values, diff2.data)
        plt4.plot(diff3.time_values, diff3.data)
        plt3.addLine(y=threshold2)
        plt4.addLine(y=threshold3)
    
    events2 = list(threshold_events(diff2 / threshold2, threshold=1.0, adjust_times=False))
    events3 = list(threshold_events(diff3 / threshold3, threshold=1.0, adjust_times=False, omit_ends=False))

    # A spike produces a combination of detectable events in both the second and third derivatives;
    # by combining these events, we increase the accuracy for detection.    
    joined_events = []
    while len(events2) > 0:
        je = []
        event = events2.pop(0)
        
        # spikes begin with a + event in d2
        if event['peak'] < 0:
            continue
        # if diff2.value_at(event['peak_time']) < 0:
        #     continue
        print('------')
        print(event)
        je.append(event)
        
        # # .. and a possible + event in d3:
        # d3_search_start = event['time'] - 300e-6
        # d3_search_stop = event['time'] + 100e-6
        # # discard older or negative d2 events
        # while len(events3) > 0 and (events3[0]['time'] < d3_search_start or events3[0]['peak'] < 0):
        #     events3.pop(0)
        # if len(events3) > 0 and d3_search_start < events3[0]['time'] < d3_search_stop:
        #     print("d3+ :")
        #     print(events3[0])
        #     je.append(events3.pop(0))

        #     # ..and possibly followed by a - event in d3
        #     if len(events3) > 0 and events3[0]['peak'] < 0 and je[-1]['time'] < events3[0]['time'] < je[-1]['time'] + 500e-6:
        #         print("d3- :")
        #         print(events3[0])
        #         je.append(events3.pop(0))

        # else:
        #     print("no d3+ :")
            
            
        joined_events.append(je)

    
    spikes = []
    for je in joined_events:
        total_area = sum([abs(ev['area']) for ev in je])
        onset_time = je[0]['time']
        if total_area < 200e-6:
            continue
        
        # don't double-count spikes within 1 ms
        if len(spikes) > 0 and onset_time < spikes[-1]['onset_time'] + 1e-3:
            continue

        max_dvdt_window = onset_time, pulse_edges[1]-50e-6
        max_dvdt_chunk = diff1.time_slice(*max_dvdt_window)
        max_dvdt_idx = np.argmax(max_dvdt_chunk.data)
        max_dvdt_time = max_dvdt_chunk.time_at(max_dvdt_idx)

        max_dvdt_time, is_edge = max_time(diff1.time_slice(onset_time, pulse_edges[1] - 50e-6))
        if is_edge != 0:
            # can't see max slope
            max_dvdt_time = None
        peak_time, is_edge = max_time(trace.time_slice(onset_time, pulse_edges[1] + 2e-3))
        if is_edge != 0 or pulse_edges[1] < peak_time < pulse_edges[1] + 50e-6:
            # peak is obscured by pulse edge
            peak_time = None
        
        spikes.append({
            'onset_time': onset_time,
            'peak_time': peak_time,
            'max_dvdt_time': max_dvdt_time,
        })

    # if no spike was found in the pulse region check to see if there is a spike in the pulse termination region
    if len(spikes) == 0:
        # note that this is using the dvdt with the termination artifact in it to locate where it should start 
        dv_after_pulse = trace.time_slice(pulse_edges[1] + 100e-6, None).diff()
        
        # create a vector to fit
        dvtofit = dv_after_pulse #.time_slice(min_dvdt_time, None)
        ttofit = dvtofit.time_values  # setting time to start at zero, note: +1 because time trace of derivative needs to be one shorter
        ttofit = ttofit - ttofit[0]

        # do fit and see if it matches
        popt, pcov = curve_fit(rc_decay, ttofit, dvtofit.data, maxfev=10000)
        fit = rc_decay(ttofit, *popt)
        if show:
            plt2.plot(dv_after_pulse.time_values, dv_after_pulse.data)
            plt2.plot(dvtofit.time_values, fit, pen='b')
        mse = ((dvtofit.data - fit)**2).mean()  # mean squared error
        if mse > mse_threshold:
            search_window = 3e-3
            max_dvdt_time, is_edge = max_time(dv_after_pulse.time_slice(pulse_edges[1], pulse_edges[1] + search_window))
            if is_edge != 0:
                max_dvdt_time = None
            peak_time, is_edge = max_time(trace.time_slice(max_dvdt_time or pulse_edges[1] + 100e-6, pulse_edges[1] + search_window))
            if is_edge != 0:
                peak_time = None
            spikes.append({'onset_time': None, 'max_dvdt_time': max_dvdt_time, 'peak_time': peak_time})

    if show:
        for plt in [plt1, plt2, plt3, plt4]:
            for spike in spikes:
                if spike['onset_time'] is not None:
                    plt.addLine(x=spike['onset_time'])
                if spike['max_dvdt_time'] is not None:
                    plt.addLine(x=spike['max_dvdt_time'], pen='b')
                if spike['peak_time'] is not None:
                    plt.addLine(x=spike['peak_time'], pen='g')
        
        while w.isVisible():
            pg.Qt.QtTest.QTest.qWait(1)
    
    if len(spikes) == 0:
        return None
    return spikes[0]



def detect_vc_evoked_spike(trace, pulse_edges, sigma=20e-6, delay=150e-6, threshold=50e-12, show=False):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    This function assumes that a square voltage pulse is used to evoke an unclamped spike
    in a voltage clamp recording, and that the peak of the unclamped spike occurs *during*
    the stimulation pulse.

    Parameters
    ==========
    trace : Trace instance
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *trace*.
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
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)
    
    dt = trace.dt

    # select just the portion of the trace that contains the pulse, excluding a short
    # delay after the pulse onset
    # pstart = pulse_edges[0] + int(delay / dt)
    # pstop = pulse_edges[1]
    pulse = trace.time_slice(pulse_edges[0] + delay, pulse_edges[1])
    if len(pulse) == 0:
        raise ValueError("Invalid pulse edges %r for trace %s" % (pulse_edges, trace))

    # find the location of the minimum value during the pulse
    smooth = pulse.copy(data=gaussian_filter(pulse.data, int(sigma/dt)))
    
    peak_time, is_edge = min_time(smooth)
    if is_edge != 0:
        # no local minimum found within pulse edges
        return None
    
    # a spike is detected only if the peak_time is at least 50pA less than the final value before pulse offset
    peak_diff = smooth.data[-1] - smooth.value_at(peak_time)
    if peak_diff > threshold:
        # Walk backward to the point of max dv/dt
        dv = smooth.diff()
        max_dvdt_time, is_edge = min_time(dv.time_slice(peak_time - 1e-3, peak_time))
        max_dvdt = dv.value_at(max_dvdt_time)

        return {'max_dvdt_time': max_dvdt_time, 'peak_time': peak_time, 'peak_diff': peak_diff, 'max_dvdt': max_dvdt}
    else:
        return None


def apply_cos_mask(trace, center, radius, power):
    """Multiply a region of a trace by a cosine mask to dampen artifacts without generating 
    sharp edges.
    
    The input *trace* is modified in-place.
    """
    chunk = trace.time_slice(center - radius, center + radius)
    w1 = np.pi * (chunk.t0 - center) / radius
    w2 = np.pi * (chunk.t_end - center) / radius
    mask_t = np.pi + np.linspace(w1, w2, len(chunk))
    mask = ((np.cos(mask_t) + 1) * 0.5) ** power
    chunk.data[:] = chunk.data * mask


def max_time(trace):
    """Return the time of the maximum value in the trace, and a value indicating whether the
    time returned coincides with the first value in the trace (-1), the last value in the
    trace (1) or neither (0).
    """
    ind = np.argmax(trace.data)
    if ind == 0:
        is_edge = -1
    elif ind == len(trace) - 1:
        is_edge = 1
    else:
        is_edge = 0
    return trace.time_at(ind), is_edge

def min_time(trace):
    """Return the time of the minimum value in the trace, and a value indicating whether the
    time returned coincides with the first value in the trace (-1), the last value in the
    trace (1) or neither (0).
    """
    ind = np.argmin(trace.data)
    if ind == 0:
        is_edge = -1
    elif ind == len(trace) - 1:
        is_edge = 1
    else:
        is_edge = 0
    return trace.time_at(ind), is_edge
