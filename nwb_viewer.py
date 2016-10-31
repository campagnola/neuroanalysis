import numpy as np
from scipy.ndimage import gaussian_filter
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from neuroanalysis.miesnwb import MiesNwb


class MiesNwbViewer(QtGui.QWidget):
    def __init__(self, nwb):
        QtGui.QWidget.__init__(self)
        self.nwb = nwb 
        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'lowpass', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 0.1},
            
        ])
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.hsplit = QtGui.QSplitter()
        self.hsplit.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.hsplit, 0, 0)
        
        self.ptree = pg.parametertree.ParameterTree()
        self.ptree.setParameters(self.params, showTop=False)
        self.hsplit.addWidget(self.ptree)
        
        self.plots = PlotGrid()
        self.hsplit.addWidget(self.plots)

        self.resize(1000, 800)
        self.hsplit.setSizes([150, 850])

    def show_group(self, i):
        g = self.nwb.sweep_groups()[i]
        data = g.data()
        data, stim = data[...,0], data[...,1]  # unpack stim and recordings
        dt = g.sweeps[0].traces()[0].meta()['Minimum Sampling interval']

        # get pulse times for each channel
        stim = stim[0]
        diff = stim[:,1:] - stim[:,:-1]
        # note: the [1:] here skips the test pulse
        on_times = [np.argwhere(diff[i] > 0)[1:] for i in range(diff.shape[0])]
        off_times = [np.argwhere(diff[i] < 0)[1:] for i in range(diff.shape[0])]

        # remove capacitive artifacts
        for i in range(stim.shape[0]):
            for j in range(stim.shape[0]):
                if i == j:
                    continue
                for k in range(len(on_times[i])):
                    on = on_times[i][k]
                    off = off_times[i][k]
                    r = 10  # flatten 10 samples following each transient
                    data[:,j,on:on+r] = data[:,j,on:on+1]
                    data[:,j,off:off+r] = data[:,j,off:off+1]

        data = data.mean(axis=0)
        data = gaussian_filter(data, (0, 15))

        self.plots.set_shape(data.shape[0], data.shape[0])
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                plt = self.plots[i, j]
                start = on_times[j][0] - 1000
                stop = on_times[j][2] + 1000  # only look at the first 3 spikes

                seg = data[i, start:stop]
                seg -= seg[:1000].mean()

                if i == j:
                    color = (80, 80, 80)
                else:
                    qe = 30 * np.clip(seg, 0, 1e20).mean() / seg[:1000].std()
                    qi = 30 * np.clip(-seg, 0, 1e20).mean() / seg[:1000].std()
                    g = 100
                    r = np.clip(g + max(qi, 0), 0, 255)
                    b = np.clip(g + max(qe, 0), 0, 255)
                    color = (r, g, b)

                plt.plot(np.arange(len(seg))*dt, seg, clear=True, pen={'color': color, 'width': 2}, antialias=True)
                # plt.setAutoPan(y=True)
                # plt.setMouseEnabled(y=False)
                if i > 0 or j > 0:
                    plt.setXLink(self.plots[0, 0])
                # if i == 0 and j > 1:
                #     plt.setYLink(self.plots[i, 1])
                # elif i > 0 and j > 0 and j != i:
                #     plt.setYLink(self.plots[i, 0])
                if j > 0:
                    plt.setYLink(self.plots[i, 0])

                if i < data.shape[0] - 1:
                    plt.getAxis('bottom').setVisible(False)
                if j > 0:
                    plt.getAxis('left').setVisible(False)

                plt.setYRange(-14, 14)

        
class PlotGrid(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        
        self.rows = 0
        self.cols = 0
        self.plots = []
        
        self.layout = QtGui.QGridLayout()
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.grid = pg.GraphicsLayoutWidget()
        self.grid.ci.layout.setSpacing(0)
        self.layout.addWidget(self.grid)
        
    def __getitem__(self, item):
        return self.plots[item[0]][item[1]]
        
    def set_shape(self, rows, cols):
        assert rows * cols < 400
        if rows == self.rows and cols == self.cols:
            return
        self.remove_plots()
        
        for i in range(rows):
            row = []
            for j in range(cols):
                p = self.grid.addPlot(i, j)
                row.append(p)
            self.plots.append(row)

        self.rows = rows
        self.cols = cols
        
    def remove_plots(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self[i, j].hide()
                self[i, j].close()
                self[i, j].setScene(None)
        self.plots = []

        
if __name__ == '__main__':
    import sys
    from pprint import pprint
    
    filename = sys.argv[1]
    nwb = MiesNwb(filename)
    sweeps = nwb.sweeps()
    traces = sweeps[0].traces()
    # pprint(traces[0].meta())
    groups = nwb.sweep_groups()
    for i,g in enumerate(groups):
        print "--------", i, g
        print g.describe()

    d = groups[7].data()
    print d.shape

    app = pg.mkQApp()
    w = MiesNwbViewer(nwb)
    w.show()
    w.show_group(7)
