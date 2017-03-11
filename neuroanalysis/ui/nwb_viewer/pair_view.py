import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from collections import OrderedDict
from ..plot_grid import PlotGrid
from ..filter import SignalFilter, ArtifactRemover
from ..baseline import BaselineRemover
from ...data import Trace
from ...spike_detection import detect_evoked_spike
from ... import fitting
from ...baseline import float_mode
from ...stats import ragged_mean


class PairView(QtGui.QWidget):
    """For analyzing pre/post-synaptic pairs.
    """
    def __init__(self, parent=None):
        self.sweeps = []
        self.channels = []

        QtGui.QWidget.__init__(self, parent)

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.vsplit = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(self.vsplit, 0, 0)
        
        self.pre_plot = pg.PlotWidget()
        self.post_plot = pg.PlotWidget()
        self.post_plot.setXLink(self.pre_plot)
        self.vsplit.addWidget(self.pre_plot)
        self.vsplit.addWidget(self.post_plot)
        
        self.response_plots = PlotGrid()
        self.vsplit.addWidget(self.response_plots)
        
        self.event_table = pg.TableWidget()
        self.vsplit.addWidget(self.event_table)

        self.artifact_remover = ArtifactRemover(user_width=True)
        self.baseline_remover = BaselineRemover()
        self.filter = SignalFilter()
        
        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'pre', 'type': 'list', 'values': []},
            {'name': 'post', 'type': 'list', 'values': []},
            self.artifact_remover.params,
            self.baseline_remover.params,
            self.filter.params,
            {'name': 'time constant', 'type': 'float', 'suffix': 's', 'siPrefix': True, 'value': 10e-3, 'dec': True, 'minStep': 100e-6}
            
        ])
        self.params.sigTreeStateChanged.connect(self._update_plots)

    def data_selected(self, sweeps, channels):
        self.sweeps = sweeps
        self.channels = channels
        
        self.params.child('pre').setLimits(channels)
        self.params.child('post').setLimits(channels)
        
        self._update_plots()

    def _update_plots(self):
        sweeps = self.sweeps
        
        # clear all plots
        self.pre_plot.clear()
        self.post_plot.clear()

        pre = self.params['pre']
        post = self.params['post']
        
        # If there are no selected sweeps or channels have not been set, return
        if len(sweeps) == 0 or pre == post or pre not in self.channels or post not in self.channels:
            return

        pre_mode = sweeps[0][pre].clamp_mode
        post_mode = sweeps[0][post].clamp_mode
        for ch, mode, plot in [(pre, pre_mode, self.pre_plot), (post, post_mode, self.post_plot)]:
            units = 'A' if mode == 'vc' else 'V'
            plot.setLabels(left=("Channel %d" % ch, units), bottom=("Time", 's'))
        
        # Iterate over selected channels of all sweeps, plotting traces one at a time
        # Collect information about pulses and spikes
        pulses = []
        spikes = []
        post_traces = []
        for i,sweep in enumerate(sweeps):
            pre_trace = sweep[pre]['primary']
            post_trace = sweep[post]['primary']
            
            # Detect pulse times
            stim = sweep[pre]['command'].data
            sdiff = np.diff(stim)
            on_times = np.argwhere(sdiff > 0)[1:, 0]  # 1: skips test pulse
            off_times = np.argwhere(sdiff < 0)[1:, 0]
            pulses.append(on_times)

            # filter data
            post_filt = self.artifact_remover.process(post_trace, list(on_times) + list(off_times))
            post_filt = self.baseline_remover.process(post_filt)
            post_filt = self.filter.process(post_filt)
            post_traces.append(post_filt)
            
            # plot raw data
            color = pg.intColor(i, hues=len(sweeps)*1.3, sat=128)
            color.setAlpha(128)
            for trace, plot in [(pre_trace, self.pre_plot), (post_filt, self.post_plot)]:
                plot.plot(trace.time_values, trace.data, pen=color, antialias=False)

            # detect spike times
            spike_inds = []
            spike_info = []
            for on, off in zip(on_times, off_times):
                spike = detect_evoked_spike(sweep[pre], [on, off])
                spike_info.append(spike)
                if spike is None:
                    spike_inds.append(None)
                else:
                    spike_inds.append(spike['rise_index'])
            spikes.append(spike_info)
                    
            dt = pre_trace.dt
            vticks = pg.VTickGroup([x * dt for x in spike_inds if x is not None], yrange=[0.0, 0.2], pen=color)
            self.pre_plot.addItem(vticks)

        # Iterate over spikes, plotting average response
        all_responses = []
        avg_responses = []
        fits = []
        fit = None
        
        npulses = max(map(len, pulses))
        self.response_plots.clear()
        self.response_plots.set_shape(1, npulses+1) # 1 extra for global average
        self.response_plots.setYLink(self.response_plots[0,0])
        for i in range(1, npulses+1):
            self.response_plots[0,i].hideAxis('left')
        units = 'A' if post_mode == 'vc' else 'V'
        self.response_plots[0, 0].setLabels(left=("Averaged events (Channel %d)" % post, units))
        
        fit_pen = {'color':(30, 30, 255), 'width':2, 'dash': [1, 1]}
        for i in range(npulses):
            # get the chunk of each sweep between spikes
            responses = []
            all_responses.append(responses)
            for j, sweep in enumerate(sweeps):
                # get the current spike
                if i >= len(spikes[j]):
                    continue
                spike = spikes[j][i]
                if spike is None:
                    continue
                
                # find next spike
                next_spike = None
                for sp in spikes[j][i+1:]:
                    if sp is not None:
                        next_spike = sp
                        break
                    
                # determine time range for response
                max_len = int(40e-3 / dt)  # don't take more than 50ms for any response
                start = spike['rise_index']
                if next_spike is not None:
                    stop = min(start + max_len, next_spike['rise_index'])
                else:
                    stop = start + max_len
                    
                # collect data from this trace
                trace = post_traces[j]
                d = trace.data[start:stop].copy()
                responses.append(d)
                
            # extend all responses to the same length and take nanmean
            avg = ragged_mean(responses, method='clip')
            avg -= float_mode(avg[:int(1e-3/dt)])
            avg_responses.append(avg)
            
            # plot average response for this pulse
            start = np.median([sp[i]['rise_index'] for sp in spikes]) * dt
            t = np.arange(len(avg)) * dt
            self.response_plots[0,i].plot(t+start, avg, pen='w', antialias=True)

            # fit!
            fit = self.fit_psp(avg, t, dt, post_mode)
            fits.append(fit)
            
            self.response_plots[0,i].plot(t+start, fit.eval(), pen=fit_pen, antialias=True)
            
        # display global average
        global_avg = ragged_mean(avg_responses, method='clip')
        t = np.arange(len(global_avg)) * dt
        self.response_plots[0,-1].plot(t, global_avg, pen='w', antialias=True)
        global_fit = self.fit_psp(global_avg, t, dt, post_mode)
        self.response_plots[0,-1].plot(t, global_fit.eval(), pen=fit_pen, antialias=True)
            
        # display fit parameters in table
        events = []
        for i,f in enumerate(fits):
            vals = OrderedDict({'id': i})
            vals.update(OrderedDict([(k,f.best_values[k]) for k in f.params.keys()]))
            events.append(vals)
        self.event_table.setData(events)

    def fit_psp(self, data, t, dt, clamp_mode):
        mode = float_mode(data[:int(1e-3/dt)])
        sign = -1 if data.mean() - mode < 0 else 1
        params = OrderedDict([
            ('xoffset', (2e-3, 1e-3, 5e-3)),
            ('yoffset', data[0]),
            ('amp', sign * 10e-12),
            #('k', (2e-3, 50e-6, 10e-3)),
            ('rise_time', (2e-3, 50e-6, 10e-3)),
            ('decay_tau', (4e-3, 500e-6, 50e-3)),
            ('rise_power', (2.0, 'fixed')),
        ])
        if clamp_mode == 'ic':
            params['amp'] = sign * 10e-3
            #params['k'] = (5e-3, 50e-6, 20e-3)
            params['rise_time'] = (5e-3, 50e-6, 20e-3)
            params['decay_tau'] = (15e-3, 500e-6, 150e-3)
        
        fit_kws = {'xtol': 1e-3, 'maxfev': 100}
        
        psp = fitting.Psp()
        return psp.fit(data, x=t, fit_kws=fit_kws, **params)
