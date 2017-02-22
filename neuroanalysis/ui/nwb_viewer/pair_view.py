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
                plot.plot(trace.time_values, trace.data, pen=color, antialias=True)

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
        for i in range(len(pulses[0])):
            # get the chunk of each sweep between spikes
            responses = []
            all_responses.append(responses)
            for j, sweep in enumerate(sweeps):
                # get the current spike
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
                t = trace.time_values[start:stop]
                #if fit is not None:
                    #d -= fit.eval(x=t)
                    #d += fit.params['yoffset']
                responses.append(d)
                
            # extend all responses to the same length and take nanmean
            max_len = max([len(r) for r in responses])
            for j,resp in enumerate(responses):
                if len(resp) < max_len:
                    responses[j] = np.empty(max_len, dtype=resp.dtype)
                    responses[j][:len(resp)] = resp
                    responses[j][len(resp):] = np.nan
            avg = np.nanmean(np.vstack(responses), axis=0)
            avg_responses.append(avg)
            
            # plot average response for this pulse
            start = np.median([sp[i]['rise_index'] for sp in spikes]) * dt
            t = np.arange(len(avg)) * dt
            self.post_plot.plot(t + start, avg, pen='w', antialias=True)

            # fit!
            mode = float_mode(avg[:int(1e-3/dt)])
            sign = -1 if avg.mean() - mode < 0 else 1
            params = OrderedDict([
                ('xoffset', (2e-3, 1e-3, 5e-3)),
                ('yoffset', avg[0]),
                ('amp', sign * 10e-12),
                #('k', (2e-3, 50e-6, 10e-3)),
                ('rise_time', (2e-3, 50e-6, 10e-3)),
                ('decay_tau', (4e-3, 500e-6, 50e-3)),
                ('rise_power', (2.0, 'fixed')),
            ])
            if post_mode == 'ic':
                params['amp'] = sign * 10e-3
                #params['k'] = (5e-3, 50e-6, 20e-3)
                params['rise_time'] = (5e-3, 50e-6, 20e-3)
                params['decay_tau'] = (15e-3, 500e-6, 150e-3)
            
            fit_kws = {'xtol': 1e-3, 'maxfev': 100}
            
            psp = fitting.Psp()
            fit = psp.fit(avg, x=t, fit_kws=fit_kws, **params)
            print(fit.best_values)
            fits.append(fit)
            
            pen = {'color':(30, 30, 255), 'width':2, 'dash': [1, 1]}
            self.post_plot.plot(t+start, fit.eval(), pen=pen, antialias=True)
            
        # display fit parameters in table
        events = []
        for i,f in enumerate(fits):
            vals = OrderedDict({'id': i})
            vals.update(OrderedDict([(k,v) for k,v in f.best_values.items()]))
            vals['decay_tau'] = psp.decay_tau(**vals)
            vals['peak_time'] = vals['xoffset'] + vals['rise_time']
            events.append(vals)
        self.event_table.setData(events)
