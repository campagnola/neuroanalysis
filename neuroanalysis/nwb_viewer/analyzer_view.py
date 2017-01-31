import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from .plotgrid import PlotGrid


class AnalyzerView(QtGui.QWidget):
    """A sweep analyzer of unspecified function.
    """
    def __init__(self, parent=None):
        self.sweeps = []

        QtGui.QWidget.__init__(self, parent)

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plots = PlotGrid(parent=self)
        self.layout.addWidget(self.plots, 0, 0)

        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            # Spin boxes
            #{'name': 'value1', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 1},
            #{'name': 'value2', 'type': 'float', 'value': 10e-3, suffix='V', dec=True, minStep=1e-6},
            
            # Check box
            #{'name': 'average', 'type': 'bool', 'value': False},
            
            # Dropdown list
            #{'name': 'average', 'type': 'list', 'values': ["option 1", "option 2", "option 3"]},
            
        ])
        self.params.sigTreeStateChanged.connect(self._update_plots)

    def show_sweeps(self, sweeps):
        self.sweeps = sweeps
        if len(sweeps) == 0:
            self.plots.clear()
        else:
            self._update_plots()

    def _update_plots(self):
        sweeps = self.sweeps
        
        # clear all plots
        self.plots.clear()
        
        # If there are no selected sweeps, return now
        if len(sweeps) == 0:
            return
        
        # Resize the plot grid based on the number of channels in the first sweep
        n_channels = len(sweeps[0].channels())
        self.plots.set_shape(n_channels, 1)
        
        # Set some plot options to improve performance
        self.plots.setClipToView(True)
        self.plots.setDownsampling(True, True, 'peak')

        # Link all x axes
        self.plots.setXLink(self.plots[0, 0])
        
        # Iterate overall channels of all sweeps, plotting traces one at a time
        for sweep in sweeps:
            data = sweep.data()
            for i in range(n_channels):
                self.plots[i,0].plot(data[i, :, 0], antialias=True)
