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


class RecordingSequence(object):
    """Representation of a sequence of data acquisitions.

    For example, this could be a single type of acquisition that was repeated ten times,
    or a series of ten acquisitions that varies one parameter across ten values.

    Sequences may be multi-dimensional and may vary more than one parameter.
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
        """A list of the recordings in this sweep.
        """
        pass

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

    def channels(self):
        """Return a list of channels included in this device sweep.
        """
        pass


class PatchClampRecording(Recording):
    """Recording from a patch-clamp amplifier.

    * Current- or voltage-clamp mode
    * Minimum one recorded channel, possibly more
    * May include stimulus waveform
    * Metadata about amplifier state (filtering, gain, bridge balance, compensation, etc)
    """
    pass


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




