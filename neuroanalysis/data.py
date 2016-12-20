"""
Data Abstraction Layer
----------------------

These classes present a high-level API for handling data from various types of neurophysiology experiments.
They do not implement functionality for reading any particular type of data file, though. Rather, they
provide a high-level abstraction layer that separates analysis and visualization from the idiosyncratic
details of any particular data acquisition system.

Each data acquisition system can provide a set of subclasses to adapt its data formats to this API
(as long as the data are reasonably similar), and analysis tools should rely only on this API to ensure
that they will not need to be rewritten in order to support data collected under different acquisition
systems.

This abstraction layer also helps to enforce good coding practice by separating data representation,
analysis, and visualization.
"""

class Experiment(object):
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
    @property
    def recordings(self):
        """A list of SyncRecording instances in this Experiment sorted by starting time.
        """
        
    @property
    def sequences(self):
        """A list of RecordingSequence instances in this Experiment sorted by starting gime.
        """


class RecordingSequence(object):
    
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

    def get_device(self, device):
        """Return a RecordingSequence with only a single device selected from
        each recording.
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


class IVCurve(RecordingSequence):
    """A sequence of recordings on a single patch-clamp amplifier that vary the amplitude
    of a current or voltage pulse, typically used to characterize intrinsic membrane properties
    of the cell.
    """


class SyncRecording(object):
    """Representation of multiple synchronized recordings.

    This is typically the result of recording from multiple devices at the same time
    (for example, two patch-clamp amplifiers and a camera).
    """
    def __init__(self):
        pass

    @property
    def type(self):
        """An arbitrary string representing the type of acquisition.
        """
        pass

    @property
    def devices(self):
        """A list of the names of devices in this recording.
        """
        pass

    def __getitem__(self):
        """Return a recording given its device name.
        """

    @property
    def recordings(self):
        """A list of the recordings in this recording.
        """

    @property
    def meta(self):
        """A dictionary describing arbitrary metadata for this recording.
        """


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


class Recording(object):
    """Representation of a single continuous data acquisition from a single device,
    possibly with multiple channels of data.
    """
    def __init__(self):
        pass

    @property
    def device_type(self):
        """A string representing the type of device that generated this recording.

        Strings should be described in the global ``device_tree``.        
        """

    @property
    def channels(self):
        """A list of channels included in this recording.
        """

    @property
    def start_time(self):
        """The starting time (unix epoch) of this recording.
        """

    @property
    def sample_rate(self):
        """The sample rate (samples/sec) of the recording.
        """


class PatchClampRecording(Recording):
    """Recording from a patch-clamp amplifier.

    * Current- or voltage-clamp mode
    * Minimum one recorded channel, possibly more
    * May include stimulus waveform
    * Metadata about amplifier state (filtering, gain, bridge balance, compensation, etc)
    """
    @property
    def clamp_mode(self):
        """The mode of the patch clamp amplifier: 'vc', 'ic', or 'i0'.
        """

    @property
    def holding_potential(self):
        """The holding potential if the recording is voltage-clamp, or the
        resting membrane potential if the recording is current-clamp.
        """

    @property
    def holding_current(self):
        """The steady-state pipette current applied during this recording.
        """

    @property
    def nearest_test_pulse(self):
        """The test pulse that was acquired nearest to this recording.
        """



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




