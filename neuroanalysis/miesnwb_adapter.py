"""
Adapter that coerces MiesNwb API to match the data abstraction layer API.

"""

from .data import Experiment, SyncRecording, RecordingSequence
from .miesnwb import MiesNwb


class MiesNwbExperiment(Experiment):
    """Experiment loaded from MIES NWB file.
    """
    def __init__(self, filename):
        self.nwb = MiesNwb(filename)
        self._recordings = None
        self._sequences = None
        
    @property
    def recordings(self):
        if self._recordings is None:
            self._recordings = [MiesSyncRecording(sweep) for sweep in self.nwb.sweeps()]
        return self._recordings
    
    @property
    def sequences(self):
        if self._sequences is None:
            self._sequences = [MiesRecordingSequence(group) for group in self.nwb.sweep_groups()]
        return self._sequences
    

class MiesSyncRecording(SyncRecording):
    def __init__(self, sweep):
        self._sweep = sweep
    
    
    
class MiesRecordingSequence(RecordingSequence):
    def __init__(self, group):
        self._group = group

    
    