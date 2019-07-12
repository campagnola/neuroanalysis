from __future__ import division, print_function
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile

from .data import Trace, PatchClampRecording
from .filter import bessel_filter
from .baseline import mode_filter, adaptive_detrend
from .event_detection import threshold_events


def detect_evoked_spikes(data, pulse_edges, **kwds):
    """Return a list of dicts describing spikes in a patch clamp recording that were evoked by a single stimulus pulse.

    This function simply wraps either detect_ic_evoked_spikes or detect_vc_evoked_spikes, depending on the clamp mode
    used in *data*.

    Parameters
    ==========
    data : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *data*. 

    Returns
    =======
    spikes : list
        Each item in this list is a dictionary containing keys 'onset_time', 'max_dvdt_time', and 'peak_time',
        indicating three possible time points during the spike that could be detected. Any of these values may
        be None to indicate that the timepoint could not be reliably determined. Additional keys may be present,
        such as 'peak' and 'max_dvdt'.
    """
    trace = data['primary']
    if data.clamp_mode == 'vc':
        return detect_vc_evoked_spikes(trace, pulse_edges, **kwds)
    elif data.clamp_mode == 'ic':
        return detect_ic_evoked_spikes(trace, pulse_edges, **kwds)
    else:
        raise ValueError("Unsupported clamp mode %s" % trace.clamp_mode)


def rc_decay(t, tau, Vo): 
    """function describing the deriviative of the voltage.  If there
    is no spike one would expect this to fall off as the RC of the cell. """
    return -(Vo/tau)*np.exp(-t/tau)


def detect_ic_evoked_spikes(trace, pulse_edges, dv2_threshold=40e3, mse_threshold=40., ui=None):
    """
    """
    if ui is not None:
        ui.clear()
        ui.console.setStack()
        ui.plt1.plot(trace.time_values, trace.data)

    assert trace.data.ndim == 1
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)
    
    # calculate derivatives within pulse window
    diff1 = trace.time_slice(*pulse_edges).diff()
    diff2 = diff1.diff()

    # mask out pulse artifacts in diff2 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff2, center=edge + 100e-6, radius=400e-6, power=2)

    # low pass filter the second derivative
    diff2 = bessel_filter(diff2, 10e3, order=4, bidir=True)

    # look for positive bumps in second derivative
    events2 = list(threshold_events(diff2 / dv2_threshold, threshold=1.0, adjust_times=False))

    if ui is not None:
        ui.plt2.plot(diff1.time_values, diff1.data)
        ui.plt3.plot(diff2.time_values, diff2.data)
        ui.plt3.addLine(y=dv2_threshold)
    
    # for each bump in d2, either discard the event or generate spike metrics
    spikes = []
    for ev in events2:
        total_area = ev['area']
        onset_time = ev['time']

        # require dv2 bump to be positive, not tiny
        if total_area < 10e-6:
            continue
        
        # don't double-count spikes within 1 ms
        if len(spikes) > 0 and onset_time < spikes[-1]['onset_time'] + 1e-3:
            continue

        max_dvdt_window = onset_time, pulse_edges[1]-50e-6
        max_dvdt_chunk = diff1.time_slice(*max_dvdt_window)
        max_dvdt_idx = np.argmax(max_dvdt_chunk.data)
        max_dvdt_time = max_dvdt_chunk.time_at(max_dvdt_idx)

        max_dvdt_time, is_edge = max_time(diff1.time_slice(onset_time, pulse_edges[1] - 50e-6))
        max_dvdt = diff1.value_at(max_dvdt_time)
        # require dv/dt to be above a threshold value
        if max_dvdt <= 50:  # mV/ms
            continue
        if is_edge != 0:
            # can't see max slope
            max_dvdt_time = None
            max_dvdt = None
        peak_time, is_edge = max_time(trace.time_slice(onset_time, pulse_edges[1] + 2e-3))
        if is_edge != 0 or pulse_edges[1] < peak_time < pulse_edges[1] + 50e-6:
            # peak is obscured by pulse edge
            peak_time = None
        
        spikes.append({
            'onset_time': onset_time,
            'peak_time': peak_time,
            'max_dvdt_time': max_dvdt_time,
            'peak': None if peak_time is None else trace.value_at(peak_time),
            'max_dvdt': max_dvdt,
        })

    # if no spike was found in the pulse region check to see if there is a spike in the pulse termination region
    if len(spikes) == 0:
        # note that this is using the dvdt with the termination artifact in it to locate where it should start 
        dv_after_pulse = trace.time_slice(pulse_edges[1] + 100e-6, None).diff()
        dv_after_pulse = bessel_filter(dv_after_pulse, 15e3, bidir=True)

        # create a vector to fit
        dvtofit = dv_after_pulse #.time_slice(min_dvdt_time, None)
        ttofit = dvtofit.time_values  # setting time to start at zero, note: +1 because time trace of derivative needs to be one shorter
        ttofit = ttofit - ttofit[0]

        # do fit and see if it matches
        popt, pcov = curve_fit(rc_decay, ttofit, dvtofit.data, maxfev=10000)
        fit = rc_decay(ttofit, *popt)
        if ui is not None:
            ui.plt2.plot(dv_after_pulse.time_values, dv_after_pulse.data)
            ui.plt2.plot(dvtofit.time_values, fit, pen='b')
        mse = ((dvtofit.data - fit)**2).mean()  # mean squared error
        if mse > mse_threshold:
            search_window = 2e-3
            max_dvdt_time, is_edge = max_time(dv_after_pulse.time_slice(pulse_edges[1], pulse_edges[1] + search_window))
            if is_edge != 0:
                max_dvdt_time = None
            peak_time, is_edge = max_time(trace.time_slice(max_dvdt_time or pulse_edges[1] + 100e-6, pulse_edges[1] + search_window))
            if is_edge != 0:
                peak_time = None
            spikes.append({
                'onset_time': None,
                'max_dvdt_time': max_dvdt_time,
                'peak_time': peak_time,
                'peak': None if peak_time is None else trace.value_at(peak_time),
                'max_dvdt': None if max_dvdt_time is None else dv_after_pulse.value_at(max_dvdt_time),
            })

    if ui is not None:
        ui.show_spike_lines(spikes)
        for spike in spikes:
            print(spike)
    
    return spikes


def detect_vc_evoked_spikes(trace, pulse_edges, sigma=20e-6, delay=150e-6, threshold=50e-12, ui=None):
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
        return []
    
    # a spike is detected only if the peak_time is at least 50pA less than the final value before pulse offset
    peak_diff = smooth.data[-1] - smooth.value_at(peak_time)
    if peak_diff > threshold:
        # Walk backward to the point of max dv/dt
        dv = smooth.diff()
        max_dvdt_time, is_edge = min_time(dv.time_slice(peak_time - 1e-3, peak_time))
        max_dvdt = dv.value_at(max_dvdt_time)

        return [{'max_dvdt_time': max_dvdt_time, 'peak_time': peak_time, 'peak_diff': peak_diff, 'max_dvdt': max_dvdt}]
    else:
        return []


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


class SpikeDetectUI(object):
    """Used to display details of spike detection analysis.
    """
    def __init__(self):
        import pyqtgraph as pg
        import pyqtgraph.console

        self.pw = pg.GraphicsLayoutWidget()
        self.plt1 = self.pw.addPlot()
        self.plt2 = self.pw.addPlot(row=1, col=0)
        self.plt2.setXLink(self.plt1)
        self.plt3 = self.pw.addPlot(row=2, col=0)
        self.plt3.setXLink(self.plt1)
        
        self.console = pg.console.ConsoleWidget()
        
        self.widget = pg.QtGui.QSplitter(pg.QtCore.Qt.Vertical)
        self.widget.addWidget(self.pw)        
        self.widget.addWidget(self.console)
        self.widget.resize(1000, 900)
        self.widget.show()
    
    def clear(self):
        self.plt1.clear()
        self.plt2.clear()
        self.plt3.clear()

    def show_spike_lines(self, spikes):
        for plt in [self.plt1, self.plt2, self.plt3]:
            for spike in spikes:
                if spike['onset_time'] is not None:
                    plt.addLine(x=spike['onset_time'])
                if spike['max_dvdt_time'] is not None:
                    plt.addLine(x=spike['max_dvdt_time'], pen='b')
                if spike['peak_time'] is not None:
                    plt.addLine(x=spike['peak_time'], pen='g')
