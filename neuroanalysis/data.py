"""
Data Abstraction Layer
----------------------

These classes present a high-level API for handling data from various types of neurophysiology experiments.
They do not implement functionality for reading any particular type of data file, though. Rather, they
provide an abstraction layer that separates analysis and visualization from the idiosyncratic
details of any particular data acquisition system.

Each data acquisition system can provide a set of subclasses to adapt its data formats to this API
(as long as the data are reasonably similar), and analysis tools should rely only on this API to ensure
that they will not need to be rewritten in order to support data collected under different acquisition
systems.

This abstraction layer also helps to enforce good coding practice by separating data representation,
analysis, and visualization.
"""
import numpy as np
import pandas
from . import util
from collections import OrderedDict



class Container(object):
    """Generic hierarchical container. 
    
    This class is the basis for most other classes in the DAL.
    """
    def __init__(self):
        self._meta = OrderedDict()
        
    @property
    def parent(self):
        return None
    
    @property
    def children(self):
        return []

    @property
    def key(self):
        """Key that uniquely identifies this object among its siblings.
        """
        return None
    
    @property
    def meta(self):
        return self._meta

    @property
    def all_children(self):
        allch = [self]
        for ch in self.children:
            allch.extend(ch.all_children)
        return allch

    @property
    def all_meta(self):
        allmeta = OrderedDict()
        for obj in self.path:
            m = obj.meta
            allmeta.update(m)
        return allmeta
        
    @property
    def path(self):
        obj = self
        path = []
        while obj is not None:
            path.append(obj)
            obj = obj.parent
        return path[::-1]


class Experiment(Container):
    """A generic container for RecordingSequence and SyncRecording instances that
    were acquired together.
    
    The boundaries between one experiment and the next are sometimes ambiguous, but
    in general we group multiple recordings into an experiment if they are likely to
    be analyzed together. Likewise, recordings that have no causal relationship
    to each other probably belong in different Experiment containers. For example,
    a series of recordings made on the same cell almost certainly belong in the same
    Experiment, whereas recordings made from different pieces of tissue probably
    belong in different Experiments.
    """
    def __init__(self, data=None, meta=None):
        Container.__init__(self)
        self._data = data
        if meta is not None:
            self._meta.update(OrderedDict(meta))
    
    @property
    def contents(self):
        """A list of data objects (Trace, Recording, SyncRecording, RecordingSequence)
        directly contained in this experiment.
        
        Grandchild objects are not included in this list.
        """
        return self._data[:]

    def find(self, type):
        return [c for c in self.all_children if isinstance(c, type)]

    @property
    def all_traces(self):
        return self.find(Trace)
    
    @property
    def all_recordings(self):
        return self.find(Recording)

    @property
    def all_sync_recordings(self):
        return self.find(SyncRecording)

    def meta_table(self, objs):
        # collect all metadata
        meta = []
        for i,o in enumerate(objs):
            meta.append(o.all_meta)
            
        # generate a list of common fields (in the correct order)
        fields = set(meta[0].keys())
        for m in meta[1:]:
            fields &= set(m.keys())
        order = list(meta[0].keys())
        for k in order[:]:
            if k not in fields:
                order.remove(k)
        
        # transpose
        tr = OrderedDict()
        for k in order:
            tr[k] = [m[k] for m in meta]
        
        # create a table
        return pandas.DataFrame(tr)

    @property
    def trace_table(self):
        return self.meta_table(self.all_traces)

    @property
    def parent(self):
        """None
        
        This is a convenience property used for traversing the object hierarchy.
        """
        return None
    
    @property
    def children(self):
        """Alias for self.contents
        
        This is a convenience property used for traversing the object hierarchy.
        """
        return self.contents
    
    

class RecordingSequence(Container):
    
    #  Acquisition?
    #  RecordingSet?
    
    #  Do we need both SyncRecordingSequence and RecordingSequence ?
    #  Can multiple RecordingSequence instances refer to the same underlying sequence?
    #    - Let's say no--otherwise we have to worry about unique identification, comparison, etc.
    #    - Add ___View classes that slice/dice any way we like.
    
    """Representation of a sequence of data acquisitions.

    For example, this could be a single type of acquisition that was repeated ten times,
    or a series of ten acquisitions that varies one parameter across ten values.
    Usually the recordings in a sequence all use the same set of devices.

    Sequences may be multi-dimensional and may vary more than one parameter.
    
    Items in a sequence are usually SyncRecording instances, but may also be
    nested RecordingSequence instances.
    """
    @property
    def type(self):
        """An arbitrary string representing the type of acquisition.
        """
        pass

    @property
    def shape(self):
        """The array-shape of the sequence.
        """

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, item):
        """Return one item (a SyncRecording instance) from the sequence.
        """

    def sequence_params(self):
        """Return a structure that describes the parameters that are varied across each
        axis of the sequence.

        For example, a two-dimensional sequence might return the following:

            [
                [param1, param2],   # two parameters that vary across the first sequence axis
                [],  # no parameters vary across the second axis (just repetitions)
                [param3],  # one parameter that varies across all recordings, regardless of its position along any axis
            ]

        Each parameter must be a key in the metadata for a single recording.
        """

    @property
    def parent(self):
        """None
        
        This is a convenience property used for traversing the object hierarchy.
        """
        return None
    
    @property
    def children(self):
        """Alias for self.contents
        
        This is a convenience property used for traversing the object hierarchy.
        """
        return self.contents


class IVCurve(RecordingSequence):
    """A sequence of recordings on a single patch-clamp amplifier that vary the amplitude
    of a current or voltage pulse, typically used to characterize intrinsic membrane properties
    of the cell.
    """


class SyncRecording(Container):
    """Representation of multiple synchronized recordings.

    This is typically the result of recording from multiple devices at the same time
    (for example, two patch-clamp amplifiers and a camera).
    """
    def __init__(self, recordings=None, parent=None):
        self._parent = parent
        self._recordings = recordings if recordings is not None else OrderedDict()
        Container.__init__(self)

    @property
    def type(self):
        """An arbitrary string representing the type of acquisition.
        """
        pass

    @property
    def devices(self):
        """A list of the names of devices in this recording.
        """
        return self._recordings.keys()

    def __getitem__(self, item):
        """Return a recording given its device name.
        """
        return self._recordings[item]

    @property
    def recordings(self):
        """A list of the recordings in this syncrecording.
        """
        return self._recordings.values()

    def data(self):
        return np.concatenate([self[dev].data()[None, :] for dev in self.devices], axis=0)

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self.recordings


class MultipatchProbe(SyncRecording):
    """A synchronous recording between multiple patch electrodes in which spiking (and possibly
    synaptic events) are evoked in one or more cells.
    """


device_tree = {
    'patch clamp amplifier': {
        'MultiClamp 700': [
            'MultiClamp 700A',
            'MultiClamp 700B',
        ],
    },
}


class Recording(Container):
    """Representation of a single continuous data acquisition from a single device,
    possibly with multiple channels of data (for example, a recording from a single
    patch-clamp headstage with input and output channels, or ).
    
    Each channel is described by a single Trace instance. Channels are often 
    recorded with the same timebase, but this is not strictly required.
    """
    def __init__(self, channels=None, start_time=None, device_type=None, device_id=None, sync_recording=None):
        Container.__init__(self)
        self._meta = OrderedDict([
            ('start_time', start_time),
            ('device_type', device_type),
            ('device_id', device_id),
        ])
        
        if channels is None:
            channels = OrderedDict()
        else:
            channels = OrderedDict(channels)
            for k,v in channels.items():
                assert isinstance(v, Trace)
        self._channels = channels

        self._sync_recording = util.WeakRef(sync_recording)
        
    @property
    def device_type(self):
        """A string representing the type of device that generated this recording.

        Strings should be described in the global ``device_tree``.        
        """
        return self._meta['device_type']

    @property
    def channels(self):
        """A list of channels included in this recording.
        """
        return self._channels.keys()

    @property
    def start_time(self):
        """The starting time (unix epoch) of this recording.
        """
        return self._meta['start_time']

    @property
    def device_id(self):
        return self._meta['device_id']
    
    @property
    def sync_recording(self):
        return self._sync_recording()

    def __getitem__(self, chan):
        return self._channels[chan]

    def data(self):
        return np.concatenate([self[ch].data[:,None] for ch in self.channels], axis=1)

    @property
    def parent(self):
        return self.sync_recording
    
    @property
    def children(self):
        return list(self._channels.values())


class PatchClampRecording(Recording):
    """Recording from a patch-clamp amplifier.

    * Current- or voltage-clamp mode
    * Minimum one recorded channel, possibly more
    * May include stimulus waveform
    * Metadata about amplifier state (filtering, gain, bridge balance, compensation, etc)
    """
    def __init__(self, *args, **kwds):
        meta = OrderedDict()
        for k in ['clamp_mode', 'patch_mode', 'holding_potential', 'holding_current']:
            meta[k] = kwds.pop(k, None)
        Recording.__init__(self, *args, **kwds)
        self._meta.update(meta)
        
    @property
    def clamp_mode(self):
        """The mode of the patch clamp amplifier: 'vc', 'ic', or 'i0'.
        """
        return self._meta['clamp_mode']

    @property
    def patch_mode(self):
        """The state of the membrane patch. E.g. 'whole cell', 'cell attached', 'loose seal', 'bath', 'inside out', 'outside out'
        """
        return self._meta['patch_mode']

    @property
    def holding_potential(self):
        """The holding potential if the recording is voltage-clamp, or the
        resting membrane potential if the recording is current-clamp.
        """
        if self.clamp_mode == 'vc':
            return self._meta['holding_potential']
        else:
            return self._baseline_value()

    @property
    def holding_current(self):
        """The steady-state pipette current applied during this recording.
        """
        if self.clamp_mode == 'ic':
            return self._meta['holding_current']
        else:
            return self._baseline_value()

    def _baseline_value(self):
        """Return median value of first 10 ms from primary channel data.
        """
        t = self['primary']
        return np.median(t.data[:int(10e-3/t.dt)])

    @property
    def nearest_test_pulse(self):
        """The test pulse that was acquired nearest to this recording.
        """

    def __repr__(self):
        mode = self.clamp_mode
        if mode == 'vc':
            extra = "mode=VC holding=%d" % int(np.round(self.holding_potential))
        elif mode == 'ic':
            extra = "mode=IC holding=%d" % int(np.round(self.holding_current))

        return "<%s %s>" % (self.__class__.__name__, extra)


class Trace(Container):
    """A homogeneous time series data set. 
    
    This is a representation of a single stream of data recorded over time. The
    data must be representable as a single N-dimensional array where the first
    array axis is time. 
    
    Examples:
    
    * A membrane potential recording from a single current-clamp headstage
    * A video stream from a camera
    * A digital trigger waveform
    
    Traces may specify units, a starting time, and either a sample period or an
    array of time values.
    """
    def __init__(self, data=None, dt=None, start_time=None, time_values=None, units=None, channel_id=None, recording=None, **meta):
        Container.__init__(self)
        self._data = data
        self._meta = OrderedDict([
            ('start_time', start_time),
            ('dt', dt),
            ('units', units),
            ('channel_id', channel_id),
        ])
        self._meta.update(meta)
        self._time_values = time_values
        self._recording = util.WeakRef(recording)
        
    @property
    def data(self):
        return self._data
        
    @property
    def start_time(self):
        return self._meta['start_time']
    
    @property
    def sample_rate(self):
        if self._meta['dt'] is None:
            raise TypeError("Trace sample rate was not specified.")
        return 1.0 / self._meta['dt']

    @property
    def dt(self):
        if self._meta['dt'] is None:
            # assume regular sampling
            t = self.time_values
            self._meta['dt'] = t[1] - t[0]
        return self._meta['dt']
    
    @property
    def units(self):
        return self._meta['units']

    @property
    def time_values(self):
        if self._time_values is None:
            if self._meta['dt'] is None:
                raise TypeError("No time values or sample rate were specified for this Trace.")
            self._time_values = np.arange(len(self.data)) * self.dt
        return self._time_values

    @property
    def shape(self):
        return self.data.shape
    
    def __len__(self):
        return self.shape[0]
    
    @property
    def duration(self):
        return self.shape[0] * self.dt

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def channel_id(self):
        return self._meta['channel_id']
    
    @property
    def recording(self):
        return self._recording()

    def copy(self, data=None, time_values=None, **kwds):
        if data is None:
            data = self.data.copy()
        
        if time_values is None:
            tval = self._time_values
            if tval is not None:
                tval = tval.copy()
        else:
            tval = time_values
        
        meta = self._meta.copy()
        meta.update(kwds)
        
        return Trace(data, time_values=tval, recording=self.recording, **meta)

    @property
    def parent(self):
        return self.recording

    def __getitem__(self, item):
        if isinstance(item, slice):
            return TraceView(self, item)
        else:
            raise TypeError("Invalid Trace slice: %r" % item)

    def downsample(self, n=None, f=None):
        """Return a downsampled copy of this trace.
        
        Parameters
        ----------
        n : int
            (optional) number of samples to average
        f : float
            (optional) desired target sample rate
        """
        if None not in (f, n):
            raise TypeError("Must specify either n or f (not both).")
        if n is None:
            if f is None:
                raise TypeError("Must specify either n or f.")
            n = self.sample_rate // f
        data = util.downsample(self.data, n, axis=0)
        tvals = self._time_values
        if tvals is not None:
            tvals = tvals[::n]
        dt = self._meta['dt']
        if dt is not None:
            dt = dt * n
        
        return self.copy(data=data, time_values=tvals, dt=dt)
        
    

class TraceView(Trace):
    def __init__(self, trace, sl):
        self._parent_trace = trace
        self._view_slice = sl
        data = trace.data[self._view_slice]
        tvals = trace.time_values[self._view_slice]
        meta = {k:trace.meta[k] for k in ['dt', 'start_time', 'units', 'channel_id']}
        Trace.__init__(self, data, time_values=tvals, recording=trace.recording, **meta)
        

# TODO: this class should not be a subclass of PatchClampRecording
# Instead, it should have a PatchClampRecording instance as an attribute.
class PatchClampTestPulse(PatchClampRecording):    
    @property
    def access_resistance(self):
        """The access resistance at the time of this recording.

        This value may be calculated from a test pulse found within the recording,
        or from data collected shortly before/after this recording, or it may
        be None if the value is not known.
        """
        return None
        
    @property
    def input_resistance(self):
        """The input resistance of the cell at the time of this recording.

        This value may be calculated from a test pulse found within the recording,
        or from data collected shortly before/after this recording, or it may
        be None if the value is not known.
        """
        return None
    
    @property
    def capacitance(self):
        """The capacitance of the cell at the time of this recording.

        This value may be calculated from a test pulse found within the recording,
        or from data collected shortly before/after this recording, or it may
        be None if the value is not known.
        """
        return None


class DAQRecording(Recording):
    """Input from / output to multiple channels on a data acquisition device.

    * Multiple channels of synchronized data (for each time point, one sample exists on every channel)
    * Metadata about DAQ state (sample rate, resolution, limits, filtering, etc)
    """


class ImageRecording(Recording):
    """Recording of one or more frames from a 2D imaging device (camera, 2p, confocal, etc).
    """


class VolumeRecording(Recording):
    """Recording of one or more frames from a 3D imaging device (MRI, lightfield, tomography, etc).
    """




