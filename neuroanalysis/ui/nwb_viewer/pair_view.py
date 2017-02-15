import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from ..plot_grid import PlotGrid
from ..filter import SignalFilter
from ...data import Trace


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
        for sweep in sweeps:
            pre_trace = sweep.traces()[pre]
            post_trace = sweep.traces()[post]
            
            for trace, plot in [(pre_trace, self.pre_plot), (post_trace, self.post_plot)]:
                dt = trace.sample_rate / 1000.
                t = Trace(trace.data()[:,0], dt=dt)
                f = self.filter.process(t)
                plot.plot(f.time_values, f.data)
                plot.setLabels(left="Channel %d" % trace.headstage_id, bottom=("Time", 's'))
