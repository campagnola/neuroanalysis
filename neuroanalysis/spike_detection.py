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


import pyqtgraph as pg
import pyqtgraph.console
pg.mkQApp()
c = pg.console.ConsoleWidget()
c.show()

def detect_ic_evoked_spike(trace, pulse_edges, dvdt_threshold=0.0013, mse_threshold=80.):
    global c
    c.setStack()

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

    assert trace.data.ndim == 1
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)
    
    # get indicies for a window within the pulse to look in (note these are not index of the pulse)
    pulse_window = (pulse_edges[0] + 300e-6, pulse_edges[1] - 50e-6)

    # third derivative of trace
    diff1 = trace.time_slice(*pulse_edges).diff()
    plt2.plot(diff1.time_values, diff1.data)
    diff2 = diff1.diff()

    # plt.figure()
    # plt.plot(dvdt, 'r')

    # mask out pulse edge artifacts
    # index_mask = int(np.round(2.e-5/dt))
    # diff3[(pstop-index_mask+1) : (pstop+index_mask+1)] = np.nan
    print(pulse_edges)

    # plt3.plot(diff2.time_values, diff2.data, pen=0.4)

    # mask out pulse artifacts from diff2 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff2, center=edge + 100e-6, radius=400e-6, power=2)

    diff2 = bessel_filter(diff2, 20e3, order=4, bidir=True)
    diff3 = diff2.diff()
    
    plt3.plot(diff2.time_values, diff2.data)

    # mask out pulse artifacts from diff3 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff3, center=edge, radius=400e-6, power=3)

    # plt4.plot(diff3.time_values, diff3.data, pen=0.4)

    diff3 = bessel_filter(diff3, 10e3, order=8, bidir=True)
    
    # clip off sharp artifacts at the end of the trace
    # diff2 = diff2.time_slice(diff2.t0, diff2.t_end - 0.5e-3)
    # diff3 = diff3.time_slice(diff3.t0, diff3.t_end - 0.5e-3)

    plt4.plot(diff3.time_values, diff3.data)
    w.show()
    # plt.plot(dvdt, 'b')

    # threshold2 = scoreatpercentile(diff2.time_slice(pulse_edges[1]+3e-3, None).data, 85) * 2
    threshold2 = 40e3  # would be better to measure this, but it's tricky in d2..
    threshold3 = scoreatpercentile(np.abs(diff3.time_slice(*pulse_window).data), 70)
    plt3.addLine(y=threshold2)
    plt4.addLine(y=threshold3)
    
    events2 = list(threshold_events(diff2 / threshold2, threshold=1.0, adjust_times=False))
    events3 = list(threshold_events(diff3 / threshold3, threshold=1.0, adjust_times=False, omit_ends=False))

    # A spike produces a combination of detectable events in both the second and third derivatives;
    # by combining these events, we increase the accuracy for detection.    
    print("==========================")
    joined_events = []
    while len(events3) > 0:
        je = []
        event = events3.pop(0)
        
        # spikes begin with a + event in d3 # that coincides with a positive value in the lowpassed d2
        if event['peak'] < 0:
            continue
        # if diff2.value_at(event['peak_time']) < 0:
        #     continue
        print('------')
        print(event)
        je.append(event)
        
        # .. and a mandatory + event in d2:
        d2_search_start = event['time'] - 100e-6
        d2_search_stop = event['time'] + 300e-6
        # discard older or negative d2 events
        while len(events2) > 0 and (events2[0]['time'] < d2_search_start or events2[0]['peak'] < 0):
            events2.pop(0)
        if len(events2) > 0 and d2_search_start < events2[0]['time'] < d2_search_stop:
            print("d2+ :")
            print(events2[0])
            je.append(events2.pop(0))
        else:
            print("no d2+ :")
            for ev2 in events2:
                print(ev2)
            continue
            
        # ..and possibly followed by a - event in d3
        if len(events3) > 0 and events3[0]['peak'] < 0 and event['time'] < events3[0]['time'] < event['time'] + 500e-6:
            print("d3- :")
            print(events3[0])
            je.append(events3.pop(0))
            
        joined_events.append(je)

    
    spikes = []
    for je in joined_events:
        total_area = sum([abs(ev['area']) for ev in je])
        onset_time = diff3.time_at(je[0]['peak_index'])
        print(onset_time, total_area, len(je))
        if total_area < 300e-6:
            continue
        
        # don't double-count spikes within 1 ms
        if len(spikes) > 0 and onset_time < spikes[-1]['onset_time'] + 1e-3:
            continue

        max_dvdt_window = onset_time, pulse_edges[1]-50e-6
        max_dvdt_chunk = diff1.time_slice(*max_dvdt_window)
        max_dvdt_idx = np.argmax(max_dvdt_chunk.data)
        max_dvdt_time = max_dvdt_chunk.time_at(max_dvdt_idx)
        if max_dvdt_time > max_dvdt_window[1] - 10e-6:
            # can't see max slope
            max_dvdt_time = None
        
        spikes.append({
            'onset_time': onset_time,
            'peak_time': None,
            'max_dvdt_time': max_dvdt_time,
        })

    

    # if d2vdt2[max_d2vdt2_idx] > d2vdt2_threshold:
    #     #find max dvdt nearby
    #     max_dvdt_index = np.argmax(dvdt[max_d2vdt2_idx : max_d2vdt2_idx + int(35.e-5 / dt)]) + max_d2vdt2_idx 

    #     # if d2vdt2 crosses threshold make sure the dvdt crosses it's threshold
    #     if dvdt[max_dvdt_index] > dvdt_threshold:
    #         spike_index = max_dvdt_index + 1 #+1 due to translating dvdt time to v time (off by 1)
    #     else:
    #         spike_index = None
    # else:
    #     spike_index = None
    

    # if no spike was found in the pulse region check to see if there is a spike in the pulse termination region
    if len(spikes) == 0:
        # note that this is using the dvdt with the termination artifact in it to locate where it should start 
        afterpulse = trace.time_slice(pulse_edges[1], None).diff()
        plt2.plot(afterpulse.time_values, afterpulse.data)
        
        # #find location of minimum dv/dt
        chunk = afterpulse.time_slice(afterpulse.t0+50e-6, afterpulse.t0 + 500e-6)
        min_dvdt_time = afterpulse.time_at(np.argmin(chunk.data))
    
        # create a vector to fit
        dvtofit = afterpulse.time_slice(min_dvdt_time + 50e-6, None)  # +50us here to get rid of actual artifact, 
        ttofit = dvtofit.time_values  # setting time to start at zero, note: +1 because time trace of derivative needs to be one shorter
        ttofit = ttofit - ttofit[0]

        # do fit and see if it matches
        popt, pcov = curve_fit(rc_decay, ttofit, dvtofit.data, maxfev=10000)
        fit = rc_decay(ttofit, *popt)
        plt2.plot(dvtofit.time_values, fit, pen='b')
        mse = np.sum((dvtofit.data - fit)**2) / len(fit) * 1e10  # mean squared error
        if mse > mse_threshold:
            # note this is looking for the max of the data with the termination artifact removed.
            max_dvdt_idx = np.argmax(afterpulse.time_slice(pulse_edges[1], pulse_edges[1] + 3e-3).data)
            max_dvdt_time = afterpulse.time_at(max_dvdt_idx)
            spikes.append({'onset_time': None, 'max_dvdt_time': max_dvdt_time})

    # plt.figure(figsize =  (10, 8))
    # voltage = trace.data
    # time_ms = trace.time_values * 1.e3
    # ax1 = plt.subplot(111)
    # ax1.plot(time_ms, voltage)
    # ax2 = ax1.twinx()
    # ax2.plot(time_ms[1:], dvdt, color='r')
    # ax2.plot(time_ms[:len(d2vdt2)], d2vdt2, color='g')
    # ax2.plot([time_ms[pulse_window_start],time_ms[pulse_window_end]], [d2vdt2_threshold, d2vdt2_threshold], '--k', lw=.5)    
    # plt.axvspan(time_ms[pulse_window_start],time_ms[pulse_window_end], color='k', alpha=.3 )
    # if spike_index:
    #     plt.axvline(time_ms[spike_index], color='k')
    # plt.xlabel('ms')
    # plt.title('dt '+str(trace.dt))
    # plt.show()



    # # find voltage peak within 1 ms after spike initiation
    # if spike_index:
    #     chunk = trace.data[spike_index: (spike_index +  int(.001 / dt))]
    #     peak_idx =  np.argmax(chunk) + spike_index    
    #     return {'peak_time': trace.time_at(peak_idx), 'max_dvdt_time': trace.time_at(spike_index),
    #         'peak_val': trace.data[peak_idx], 'max_dvdt': trace.data[spike_index]}
    # else:
    #     return None

    for plt in [plt1, plt2, plt3, plt4]:
        for spike in spikes:
            if spike['onset_time'] is not None:
                plt.addLine(x=spike['onset_time'])
            if spike['max_dvdt_time'] is not None:
                plt.addLine(x=spike['max_dvdt_time'], pen='b')
    
    while w.isVisible():
        pg.Qt.QtTest.QTest.qWait(1)
    
    if len(spikes) == 0:
        return None
    return spikes[0]



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
