from data import PatchClampRecording


class PatchClampTestPulse(PatchClampRecording):
    """A PatchClampRecording that contains a subthreshold, square pulse stimulus.
    """
    def __init__(self, rec, indices):
        self._parent_recording = rec
        self._indices = indices
        start, stop = indices
        
        pri = rec['primary']
        cmd = rec['command']
        
        # find pulse
        cdata = cmd.data
        mask = cdata == cdata[0]
        inds = np.argwhere(mask[:-1] != mask[1:])[:,0]
        
        if len(inds) != 2:
            raise ValueError("Could not find square pulse in command waveform.")
        self._pulse_inds = inds
        amp = cdata[inds[0]] - cdata[0]
        
        PatchClampRecording.__init__(self,
            device_type=rec.device_type, 
            device_id=rec.device_id,
            start_time=rec.start_time,
            channels={'primary': pri[start:stop], 'command': cmd[start:stop]}
        )
        self._meta.update({'stim_name': 'test_pulse', 'pulse_amplitude': amp})
        for k in ['clamp_mode', 'holding_potential', 'holding_current']:
            self._meta[k] = rec._meta[k]
            
        self._analysis = None
        
    @property
    def access_resistance(self):
        """The access resistance measured from this test pulse.
        
        Includes the bridge balance resistance if the recording was made in
        current clamp mode.
        """
        if self._analysis is None:
            self._analyze()
        return self.analysis['access_resistance']
        
    @property
    def input_resistance(self):
        """The input resistance measured from this test pulse.
        """
        if self._analysis is None:
            self._analyze()
        return self.analysis['input_resistance']
    
    @property
    def capacitance(self):
        """The capacitance of the cell measured from this test pulse.
        """
        if self._analysis is None:
            self._analyze()
        return self.analysis['capacitance']

    @property
    def parent(self):
        """The recording in which this test pulse is embedded.
        """
        return self._parent_recording

    def _analyze(self):
        pass