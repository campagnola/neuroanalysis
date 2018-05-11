import numpy as np
from .util import WeakRef
from .data import Trace


class Stimulus(object):
    """Base metadata class for describing a stimulus (current injection, laser modulation, etc.)

    Stimulus descriptions are built as a hierarchy of Stimulus instances, where each item in
    the hierarchy may have multiple children that describe its sub-components. Stimulus
    subclasses each define a set of metadata fields and an optional eval() method that
    can be used to generate the stimulus.

    Parameters
    ----------
    description : str
        Human-readable description of this stimulus
    start_time : float
        The starting time of this stimulus relative to its parent's start_time.
    items : list | None
        An optional list of child Stimulus instances. 
    parent : Stimulus | None
        An optional parent Stimulus instance.


    Examples
    --------

    1. A waveform with two square pulses::

        pulse1 = SquarePulse(start_time=0.01, duration=0.01, amplitude=-50e-12)
    """
    def __init__(self, description, start_time=0, items=None, parent=None):
        self.description = description
        self._start_time = start_time
        
        self._items = []
        self._parent = WeakRef(None)
        self.parent = parent

        for item in (items or []):
            self.append_item(item)        

    @property
    def type(self):
        """String type of this stimulus.

        The default implementation returns the name of this class.
        """
        return type(self).__name__

    @property
    def parent(self):
        """The parent stimulus object, or None if there is not parent.
        """
        return self._parent()

    @parent.setter
    def parent(self, new_parent):
        old_parent = self.parent
        if old_parent is new_parent:
            return
        if old_parent is not None:
            old_parent.remove_item(self)
        self._parent = WeakRef(new_parent)
        if self not in new_parent.items:
            new_parent.append_item(self)

    @property
    def items(self):
        """Tuple of child items contained within this stimulus.
        """
        return tuple(self._items)

    def append_item(self, item):
        """Append an item to the list of child stimuli.

        The item's parent will be set to this Stimulus.
        """
        self._items.append(item)
        item.parent = self

    def remove_item(self, item):
        """Remove an item from the list of child stimuli.

        The item's parent will be set to None.
        """
        self._items.remove(item)
        item.parent = None

    def insert_item(self, index, item):
        """Insert an item into the list of child stimuli.

        The item's parent will be set to this Stimulus.
        """
        self._items.insert(index, item)
        item.parent = self        

    @property
    def start_time(self):
        """The global starting time of this stimulus relative to 0.
        
        This is computed as the sum of all starting times in the ancestry
        of this item (including this item itself).
        """
        return sum([i.local_start_time for i in self.ancestry])

    @property
    def local_start_time(self):
        """The starting time of this stimulus relative to its parent's start time.
        """
        return self._start_time

    @property
    def ancestry(self):
        """A generator yielding this item, its parent, and all grandparents.
        """
        item = self
        while item is not None:
            yield item
            item = item.parent

    def eval(self, trace=None, t0=0, n_pts=None, dt=None, sample_rate=None, time_values=None):
        """Return the value of this stimulus (a Trace instance) at defined timepoints.
        """
        trace = self._make_eval_trace(trace=trace, t0=t0, n_pts=n_pts, dt=dt, sample_rate=sample_rate, time_values=time_values)
        for item in self.items:
            item.eval(trace=trace)
        return trace

    def _make_eval_trace(self, trace=None, t0=0, n_pts=None, dt=None, sample_rate=None, time_values=None):
        """Helper function used by all Stimulus.eval subclass methods to interpret arguments.
        """
        if trace is not None:
            return trace
        if time_values is not None:
            data = np.zeros(len(time_values))
        else:
            data = np.zeros(n_pts)
        return Trace(data, t0=t0, dt=dt, sample_rate=sample_rate, time_values=time_values)


class SquarePulse(Stimulus):
    """A square pulse stimulus.
    """
    def __init__(self, start_time, duration, amplitude, description="square pulse", parent=None):
        self.duration = duration
        self.amplitude = amplitude
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        trace.time_slice(self.start_time, self.start_time+self.duration).data[:] += self.amplitude
        return trace


class SquarePulseTrain(Stimulus):
    """A train of identical, regularly-spaced square pulses.
    """
    def __init__(self, start_time, n_pulses, pulse_duration, amplitude, interval, description="square pulse train", parent=None):
        self.n_pulses = n_pulses
        self.pulse_duration = pulse_duration
        self.amplitude = amplitude
        self.interval = interval
        self.pulse_times = np.arange(n_pulses) * pulse_interval + start_time
        pulses = []
        for i,t in enumerate(self.pulse_times):
            pulse = SquarePulse(start_time=t, duration=pulse_duration, amplitude=amplitude, parent=self)
            pulse.pulse_number = i
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, items=pulses)
        

class Ramp(Stimulus):
    """A linear ramp.
    """
    def __init__(self, start_time, duration, slope, initial_value=0, description="ramp", parent=None):
        self.duration = duration
        self.slope = slope
        self.initial_value = initial_value
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        region = trace.time_slice(self.start_time, self.start_time+self.duration).data
        region += np.arange(len(region)) * self.slope + self.initial_value
        return trace


def find_square_pulses(trace, baseline=None):
    """Return a list of SquarePulse instances describing square pulses found
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
    time_vals = trace.time_values
    if baseline is None:
        baseline = trace[0]
    sdiff = np.diff(trace)
    changes = np.argwhere(sdiff != 0)[:, 0] + 1
    pulses = []
    for i, start in enumerate(changes):
        amp = trace[start] - baseline
        if amp != 0:
            stop = changes[i+1] if (i+1 < len(changes)) else len(trace)
            t_start = time_values[start]
            duration = stop - start * trace.dt
            pulses.append(SquarePulse(t_start, duration, amp))
            pulses[-1].pulse_number = i
    return pulses
