import numpy as np


def square_pulses(trace, baseline=None):
    """Return a list of (start, stop, amp) tuples describing square pulses
    in the stimulus.
    
    A pulse is defined as any contiguous region of the stimulus waveform
    that has a constant value other than the baseline. If no baseline is
    specified, then the first sample in the stimulus is used.
    
    Parameters
    ----------
    trace : Trace instance
        The stimulus command waveform. This data should be noise-free.
    baseline : float | None
        Specifies the value in the command waveform that is considered to be
        "no pulse". If no baseline is specified, then the first sample of
        *trace* is used.
    """
    if baseline is None:
        baseline = trace[0]
    sdiff = np.diff(trace)
    changes = np.argwhere(sdiff != 0)[:, 0] + 1
    pulses = []
    for i, start in enumerate(changes):
        amp = trace[start]
        if amp != baseline:
            stop = changes[i+1] if (i+1 < len(changes)) else len(trace)
            pulses.append((start, stop, amp))
    return pulses
