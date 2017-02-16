import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from ..plot_grid import PlotGrid
from ..filter import SignalFilter
from ...data import Trace
from ...spike_detection import detect_evoked_spike


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

        self.filter = SignalFilter()
        
        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'pre', 'type': 'list', 'values': []},
            {'name': 'post', 'type': 'list', 'values': []},
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
        
        # Iterate over selected channels of all sweeps, plotting traces one at a time
        for i,sweep in enumerate(sweeps):
            pre_trace = sweep.traces()[pre]
            post_trace = sweep.traces()[post]
            
            color = pg.mkColor((i, len(sweeps)*1.3))
            
            for trace, plot in [(pre_trace, self.pre_plot), (post_trace, self.post_plot)]:
                dt = trace.sample_rate / 1000.
                t = Trace(trace.data()[:,0], dt=dt)
                f = self.filter.process(t)
                plot.plot(f.time_values, f.data, pen=color)
                plot.setLabels(left="Channel %d" % trace.headstage_id, bottom=("Time", 's'))

            # Detect pulse times
            stim = pre_trace.data()[:,1]
            sdiff = np.diff(stim)
            on_times = np.argwhere(sdiff > 0)[:, 1:]  # 1: skips test pulse
            off_times = np.argwhere(sdiff < 0)[:, 1:]

            # detect spike times
            spike_inds = []
            for on, off in zip(on_times, off_times):
                spike = detect_evoked_spike(trace, [on, off])
                if spike is None:
                    spike_inds.append(None)
                else:
                    spike_inds.append(spike['rise_ind'])
                    
            vticks = pg.VTickGroup([x for x in spike_inds if x is not None], pen=color)
            self.pre_plot.addItem(vticks)

