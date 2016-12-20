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


def _detect_evoked_spike_vc(trace, pulse_start, pulse_stop, pulse_amp, search_duration):
    baseline = trace.view(time_range=[pulse_start-5e-3, pulse_start], channel='primary')
    noise = baseline.data.std()
    trace = trace.view(time_range=[pulse_start, pulse_start + search_duration], channel='primary')
    dt = 1.0 / trace.sample_rate
    y = trace.data
    y2 = np.diff(np.diff(y))
    events = zero_crossing_events(y, min_length=int(50e-6/dt), min_peak=noise*3)
    mask = (events['index'] < 10e-6/dt) | ((events['index'] > pulse_stop/dt) & (events['index'] < (pulse_stop+10e-6)/dt))
    events = events[~mask]
    ind = np.argmax(events['peak'])
    return {'index': events[ind]['index'] + trace.time_offset, 'peak': events[ind]['peak'], 'noise': noise}

def _detect_evoked_spike_ic(trace, pulse_start, pulse_stop, pulse_amp, search_duration):
    raise NotImplementedError()
