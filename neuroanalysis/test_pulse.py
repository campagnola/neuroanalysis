import numpy as np

from data import PatchClampRecording, Trace
from fitting import Exp


class PatchClampTestPulse(PatchClampRecording):
    """A PatchClampRecording that contains a subthreshold, square pulse stimulus.
    """
    def __init__(self, rec, indices=None):
        self._parent_recording = rec
        
        if indices is None:
            indices = (0, len(rec['primary']))
        self._indices = indices
        start, stop = indices
        
        pri = rec['primary'][start:stop]
        cmd = rec['command'][start:stop]
        
        # find pulse
        cdata = cmd.data
        mask = cdata == cdata[0]
        inds = np.argwhere(mask[:-1] != mask[1:])[:,0]
        
        if len(inds) != 2:
            raise ValueError("Could not find square pulse in command waveform.")
        amp = cdata[inds[0] + 1] - cdata[0]
        
        PatchClampRecording.__init__(self,
            device_type=rec.device_type, 
            device_id=rec.device_id,
            start_time=rec.start_time,
            channels={'primary': pri, 'command': cmd}
        )
        self._meta.update({'stim_name': 'test_pulse', 'pulse_amplitude': amp,
                           'pulse_edges': inds})
        for k in ['clamp_mode', 'holding_potential', 'holding_current', 'bridge_balance',
                  'lpf_cutoff', 'pipette_offset']:
            self._meta[k] = rec._meta[k]
            
        self._analysis = None
        
    @property
    def access_resistance(self):
        """The access resistance measured from this test pulse.
        
        Includes the bridge balance resistance if the recording was made in
        current clamp mode.
        """
        return self.analysis['access_resistance']
        
    @property
    def input_resistance(self):
        """The input resistance measured from this test pulse.
        """
        return self.analysis['input_resistance']
    
    @property
    def capacitance(self):
        """The capacitance of the cell measured from this test pulse.
        """
        return self.analysis['capacitance']

    @property
    def baseline_potential(self):
        """The potential of the cell membrane measured (or clamped) before
        the onset of the test pulse.
        """
        return self.analysis['baseline_potential']
 
    @property
    def baseline_current(self):
        """The pipette current measured (or clamped) before the onset of the
        test pulse.
        """
        return self.analysis['baseline_current']
 
    @property
    def analysis(self):
        if self._analysis is None:
            self._analyze()
        return self._analysis
 
    @property
    def parent(self):
        """The recording in which this test pulse is embedded.
        """
        return self._parent_recording

    def _analyze(self):
        # adapted from ACQ4
        
        meta = self.meta
        pulse_start, pulse_stop = meta['pulse_edges']
        pulse_amp = meta['pulse_amplitude']
        clamp_mode = self.clamp_mode
        
        data = self['primary']
        dt = data.dt
        
        # Extract specific time segments
        nudge = int(50e-6 / dt)
        base = data[:pulse_start-nudge]
        pulse = data[pulse_start+nudge:pulse_stop-nudge]
        pulse_end = pulse[int(len(pulse)*2./3.):]  # last 1/3 of pulse response 
        end = data[pulse_stop+nudge:]
        
        # Exponential fit

        # predictions
        base_median = np.median(base.data)
        access_r = 10e6
        input_r = 200e6
        if clamp_mode == 'vc':
            ari = pulse_amp / access_r
            iri = pulse_amp / input_r
            params = {
                'xoffset': (pulse.t0, 'fixed'),
                'yoffset': base_median + iri, 
                'amp': ari - iri, 
                'tau': (1e-3, 0.1e-3, 50e-3),
            }
        else:
            bridge = meta['bridge_balance']
            arv = pulse_amp * (access_r - bridge)
            irv = pulse_amp * input_r
            params = {
                'xoffset': pulse.t0, 
                'yoffset': base_median+arv+irv, 
                'amp': -irv, 
                'tau': (10e-3, 1e-3, 50e-3),
            }
            
        fit_kws = {'xtol': 1e-3, 'maxfev': 200, 'nan_policy': 'omit'}
        model = Exp()
        result = model.fit(pulse.data, x=pulse.time_values, fit_kws=fit_kws, params=params)
        fit = result.best_values
        err = model.nrmse(result)
        
        self._fit_trace = Trace(result.eval(), time_values=pulse.time_values)
        
        ### fit again using shorter data
        ### this should help to avoid fitting against h-currents
        #tau4 = fit1[0][2]*10
        #t0 = pulse.xvals('Time')[0]
        #shortPulse = pulse['Time': t0:t0+tau4]
        #if shortPulse.shape[0] > 10:  ## but only if we can get enough samples from this
            #tVals2 = shortPulse.xvals('Time')-params['delayTime']
            #fit1 = scipy.optimize.leastsq(
                #lambda v, t, y: y - expFn(v, t), pred1, 
                #args=(tVals2, shortPulse['primary'].view(np.ndarray) - baseMean),
                #maxfev=200, full_output=1)

        ## Handle analysis differently depenting on clamp mode
        if clamp_mode == 'vc':
            base_v = self['command'].data[0] + self.meta['holding_potential']
            base_i = base_median
            
            input_step = fit['yoffset'] - base_i
            access_step = fit['amp'] + input_step
            
            if pulse_amp >= 0:
                input_step = max(1e-16, input_step)
                access_step = max(1e-16, access_step)
            else:
                input_step = min(-1e-16, input_step)
                access_step = min(-1e-16, access_step)
            
            access_r = pulse_amp / access_step
            input_r = pulse_amp / input_step

            # (under) estimate capacitance by measuring
            # charge transfer
            q = np.sum(pulse.data - base_i) * dt
            cap = q / pulse_amp
        
        else:
            base_v = base_median
            base_i = self['command'].data[0] + self.meta['holding_current']
            y0 = result.eval(x=pulse.t0)
            
            if pulse_amp >= 0:
                v_step = max(1e-5, fit['yoffset'] - y0)
            else:
                v_step = min(-1e-5, fit['yoffset'] - y0)
                
            if pulse_amp == 0:
                pulse_amp = 1e-14
                
            input_r = (v_step / pulse_amp)
            access_r = ((y0 - base_median) / pulse_amp) + bridge
            cap = fit['tau'] / input_r

        self._analysis = {
            'input_resistance': input_r,
            'access_resistance': access_r,
            'capacitance': cap,
            'baseline_potential': base_v,
            'baseline_current': base_i,
        }
        self._fit_result = result
        raise Exception()
    
    def plot(self):
        self.analysis
        import pyqtgraph as pg
        plt = pg.plot()
        plt.plot(self['primary'].time_values, self['primary'].data)
        plt.plot(self._fit_trace.time_values, self._fit_trace.data, pen='b')
