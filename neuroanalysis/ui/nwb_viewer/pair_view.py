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
            pre_trace = sweep[pre]['primary']
            post_trace = sweep[post]['primary']
            
            color = pg.mkColor((i, len(sweeps)*1.3))
            
            post_filt = self.filter.process(post_trace)
            
            for trace, plot in [(pre_trace, self.pre_plot), (post_filt, self.post_plot)]:
                plot.plot(trace.time_values, trace.data, pen=color)
                plot.setLabels(left="Channel %d" % trace.recording.device_id, bottom=("Time", 's'))

            # Detect pulse times
            stim = sweep[pre]['command'].data
            sdiff = np.diff(stim)
            on_times = np.argwhere(sdiff > 0)[1:, 0]  # 1: skips test pulse
            off_times = np.argwhere(sdiff < 0)[1:, 0]

            # detect spike times
            spike_inds = []
            for on, off in zip(on_times, off_times):
                spike = detect_evoked_spike(sweep[pre], [on, off])
                if spike is None:
                    spike_inds.append(None)
                else:
                    spike_inds.append(spike['rise_index'])
                    
            dt = pre_trace.dt
            vticks = pg.VTickGroup([x * dt for x in spike_inds if x is not None], yrange=[0.0, 0.2], pen=color)
            self.pre_plot.addItem(vticks)

