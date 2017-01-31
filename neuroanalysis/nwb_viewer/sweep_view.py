import numpy as np
from scipy.ndimage import gaussian_filter
import pyqtgraph as pg
import pyqtgraph.reload
from pyqtgraph.Qt import QtGui, QtCore
from .plotgrid import PlotGrid
from ..miesnwb import MiesNwb, SweepGroup


class SweepView(QtGui.QWidget):
    def __init__(self, parent=None):
        self.sweeps = []
        self.chans = None

        QtGui.QWidget.__init__(self, parent)

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plots = PlotGrid(parent=self)
        self.layout.addWidget(self.plots, 0, 0)

        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'lowpass', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 1},
            {'name': 'average', 'type': 'bool', 'value': False},
        ])
        self.params.sigTreeStateChanged.connect(self._update_plots)

    def data_selected(self, sweeps, channels):
        self.sweeps = sweeps
        self.channels = channels
        self._update_plots()

    def _update_plots(self):
        sweeps = self.sweeps
        chans = self.channels
        
        self.plots.clear()
        if len(sweeps) == 0 or len(chans) == 0:
            return
        
        # collect data
        data = MiesNwb.pack_sweep_data(sweeps)  # returns (sweeps, channels, samples, 2)
        data, stim = data[...,0], data[...,1]  # unpack stim and recordings
        dt = sweeps[0].traces().values()[0].sample_rate
        t = np.arange(data.shape[2]) * dt

        # mask for selected channels
        mask = np.array([ch in chans for ch in sweeps[0].channels()])
        data = data[:, mask]
        chans = np.array(sweeps[0].channels())[mask]

        # setup plot grid
        self.plots.set_shape(len(chans), 1)
        self.plots.setClipToView(True)
        self.plots.setDownsampling(True, True, 'peak')

        # filter data
        data = self.filter(data)

        # plot all selected data
        for i in range(data.shape[0]):
            alpha = 100 if self.params['average'] else 200
            for j in range(data.shape[1]):
                plt = self.plots[j, 0]
                plt.plot(t, data[i, j], pen=(255, 255, 255, alpha), antialias=True)

        # plot average
        if self.params['average']:
            for j in range(data.shape[1]):
                plt = self.plots[j, 0]
                plt.plot(t, data[:, j].mean(axis=0), pen=(0, 255, 0), shadowPen={'color': (0, 0, 0), 'width': 2}, antialias=True)

        # set axis labels / units
        for j in range(data.shape[1]):
            sw = sweeps[0]
            ch = chans[j]
            tr = sw.traces()[ch]
            units = 'A' if tr.meta()['Clamp Mode'] == 0 else 'V'
            self.plots[j, 0].setLabels(left=("Channel %d" % tr.headstage_id, units))

        # link x axes together
        for j in range(1, data.shape[1]):
            self.plots[j, 0].setXLink(self.plots[0, 0])

    def filter(self, data):
        lp = self.params['lowpass']
        if lp > 0:
            data = gaussian_filter(data, (0, 0, lp))
        return data


