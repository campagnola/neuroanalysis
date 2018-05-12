from collections import OrderedDict
import numpy as np
from .util import WeakRef
from .data import Trace


def load_stimulus(state):
    """Re-create a Stimulus structure from a previously saved state.

    States are generated using Stimulus.save().
    """
    return Stimulus.load(state)


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

    A waveform with two square pulses::

        stimulus = Stimulus(items=[
            SquarePulse(start_time=0.01, duration=0.01, amplitude=-50e-12),
            SquarePulse(start_time=0.2, duration=0.5, amplitude=200e-12),
        ], units='A')

    A waveform with a square pulse followed by a pulse train::

        stimulus = Stimulus(items=[
            SquarePulse(start_time=0.01, duration=0.01, amplitude=-50e-12),
            SquarePulseTrain(start_time=0.2, n_pulses=8, pulse_duration=0.002, amplitude=1.6e-9, interval=0.02),
        ], units='A')

    """
    _subclasses = {}

    _attributes = ['description', 'start_time', 'units']

    def __init__(self, description, start_time=0, units=None, items=None, parent=None):
        self.description = description
        self._start_time = start_time
        self.units = units
        
        self._items = []
        self._parent = WeakRef(None)  
        self.parent = parent

        for item in (items or []):
            self.append_item(item)        

    @property
    def type(self):
        """String type of this stimulus.
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
        self._parent = WeakRef(new_parent)
        if old_parent is not None:
            try:
                old_parent.remove_item(self)
            except ValueError:
                pass  # already removed
        if new_parent is not None and self not in new_parent.items:
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
    def global_start_time(self):
        """The global starting time of this stimulus.
        
        This is computed as the sum of all ``start_time``s in the ancestry
        of this item (including this item itself).
        """
        return sum([i.start_time for i in self.ancestry])

    @property
    def start_time(self):
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
        return Trace(data, t0=t0, dt=dt, sample_rate=sample_rate, time_values=time_values, units=self.units)

    def __repr__(self):
        return '<{class_name} "{desc}" 0x{id:x}>'.format(class_name=type(self).__name__, desc=self.description, id=id(self))

    def __eq__(self, other):
        if self.type != other.type:
            return False
        if len(self.items) != len(other.items):
            return False
        for name in self._attributes:
            if getattr(self, name) != getattr(other, name):
                return False
        for i in range(len(self.items)):
            if self.items[i] != other.items[i]:
                return False
        return True

    def save(self):
        state = OrderedDict([
            ('type', self.type),
            ('args', OrderedDict([('start_time', self.start_time)])),
        ])
        for name in self._attributes:
            state['args'][name] = getattr(self, name)
        state['items'] = [item.save() for item in self.items]
        return state

    @classmethod
    def load(cls, state):
        item_type = state['type']
        item_class = cls.get_stimulus_class(item_type)
        child_items = [cls.load(item_state) for item_state in state['items']]
        if len(child_items) > 0:
            return item_class(items=child_items, **state['args'])
        else:
            return item_class(**state['args'])

    @classmethod
    def get_stimulus_class(cls, name):
        if name not in cls._subclasses:
            cls._subclasses = {sub.__name__:sub for sub in cls.__subclasses__()}
            cls._subclasses[cls.__name__] = cls
        if name not in cls._subclasses:
            raise KeyError('Unknown stimulus class "%s"' % name)
        return cls._subclasses[name]


class SquarePulse(Stimulus):
    """A square pulse stimulus.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    duration : float
        The duration in seconds of the pulse.
    amplitude : float
        The amplitude of the pulse.
    description : str
        Optional string describing the stimulus. The default value is 'square pulse train'.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['duration', 'amplitude']

    def __init__(self, start_time, duration, amplitude, description="square pulse", units=None, parent=None):
        self.duration = duration
        self.amplitude = amplitude
        Stimulus.__init__(self, description=description, start_time=start_time, units=units, parent=parent)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        trace.time_slice(start, start+self.duration).data[:] += self.amplitude
        return trace


class SquarePulseTrain(Stimulus):
    """A train of identical, regularly-spaced square pulses.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    n_pulses : int
        The number of pulses in the train.
    pulse_duration : float
        The duration in seconds of a single pulse.
    amplitude : float
        The amplitude of a single pulse.
    interval : float
        The time in seconds between the onset of adjacent pulses.
    description : str
        Optional string describing the stimulus. The default value is 'square pulse train'.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['n_pulses', 'pulse_duration', 'amplitude', 'interval']

    def __init__(self, start_time, n_pulses, pulse_duration, amplitude, interval, description="square pulse train", units=None, parent=None):
        self.n_pulses = n_pulses
        self.pulse_duration = pulse_duration
        self.amplitude = amplitude
        self.interval = interval
        Stimulus.__init__(self, description=description, start_time=start_time, units=units, parent=parent)

        pulse_times = np.arange(n_pulses) * interval
        for i,t in enumerate(pulse_times):
            pulse = SquarePulse(start_time=t, duration=pulse_duration, amplitude=amplitude, parent=self, units=units)
            pulse.pulse_number = i

    @property
    def global_pulse_times(self):
        """A list of the global start times of all pulses in the train.
        """
        return [t + self.global_start_time for t in self.pulse_times]

    @property
    def pulse_times(self):
        """A list of the start times of all pulses in the train.
        """
        return [item.start_time for item in self.items]

    def save(self):
        state = Stimulus.save(self)
        state['items'] = []  # don't save auto-generated items
        return state

        

class Ramp(Stimulus):
    """A linear ramp.
    """
    _attributes = Stimulus._attributes + ['duration', 'slope', 'initial_value']

    def __init__(self, start_time, duration, slope, initial_value=0, description="ramp", units=None, parent=None):
        self.duration = duration
        self.slope = slope
        self.initial_value = initial_value
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

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
    if baseline is None:
        baseline = trace.data[0]
    sdiff = np.diff(trace.data)
    changes = np.argwhere(sdiff != 0)[:, 0] + 1
    pulses = []
    for i, start in enumerate(changes):
        amp = trace.data[start] - baseline
        if amp != 0:
            stop = changes[i+1] if (i+1 < len(changes)) else len(trace)
            t_start = trace.time_at(start)
            duration = (stop - start) * trace.dt
            pulses.append(SquarePulse(start_time=t_start, duration=duration, amplitude=amp, units=trace.units))
            pulses[-1].pulse_number = i
    return pulses
