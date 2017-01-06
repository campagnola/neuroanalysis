from __future__ import division, print_statement
import numpy as np


def detect_evoked_spike(trace, pulse_start, pulse_stop, pulse_amp, search_duration=None):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    Parameters
    ==========
    trace : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_start : float
        The start time (sec) of the pulse intended to evoke a spike.
    pulse_stop : float
        The end time (sec) of the pulse intended to evoke a spike.
    pulse_amp : float
        The amplitude (A for current clamp, V for voltage clamp) of the pulse.
    search_duration : float
        The time duration (sec) following the onset of the stimulus to search for an evoked spike.
        By default, this is set to ``max(0.002, 2*(pulse_stop-pulse_start))``.
    """
    pulse_start = float(pulse_start)
    pulse_stop = float(pulse_stop)
    pulse_amp = float(pulse_amp)
    if search_duration is None:
        search_duration = max(0.002, 2 * (pulse_stop-pulse-start))

    if trace.clamp_mode == 'vc':
        return _detect_evoked_spike_vc(trace, pulse_start, pulse_stop, pulse_amp)
    elif trace.clamp_mode == 'ic':
        return _detect_evoked_spike_ic(trace, pulse_start, pulse_stop, pulse_amp)
    else:
        raise ValueError("Unsupported clamp mode %s" % trace.clamp_mode)


def _detect_evoked_spike_vc(trace, pulse_start, pulse_stop, pulse_amp, search_duration, sigma=20e-6):
    dt = trace.dt
    sigma = sigma / dt
    
    delay = int(150e-6 / dt)  # short window after stimulus should be ignored
    peak_inds = []
    rise_inds = []
    for j in range(8):  # loop over pulses
        # select just the portion of the chunk that contains the pulse
        pstart = on_times[j+1] - start + delay
        pstop = off_times[j+1] - start
        # find the location of the minimum value during the pulse
        smooth = ndi.gaussian_filter(chunk[pstart:pstop], sigma)
        peak_ind = np.argmin(smooth)
        # a spike is detected only if the peak is at least 50pA less than the final value before pulse offset
        margin = 50e-12
        if smooth[peak_ind] < smooth[-1] - margin:
            peak_inds.append(peak_ind + pstart)
            
            # Walk backward to the point of max dv/dt
            dvdt = np.diff(smooth)
            rstart = max(0, peak_ind - int(1e-3/dt))  # don't search for rising phase more than 1ms before peak
            rise_ind = np.argmin(dvdt[rstart:peak_ind]) + pstart + rstart
            rise_inds.append(rise_ind)




def _detect_evoked_spike_ic(trace, pulse_start, pulse_stop, pulse_amp, search_duration):
    raise NotImplementedError()
