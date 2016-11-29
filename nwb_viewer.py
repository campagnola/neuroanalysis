import numpy as np
from scipy.ndimage import gaussian_filter
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from neuroanalysis.miesnwb import MiesNwb, SweepGroup


class MiesNwbExplorer(QtGui.QWidget):
    """Widget for listing and selecting recordings in a MIES-generated NWB file.
    """
    selection_changed = QtCore.Signal(object)

    def __init__(self, nwb):
        QtGui.QWidget.__init__(self)

        self._nwb = None

        self.layout = QtGui.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # self.menu = QtGui.QMenuBar()
        # self.layout.addWidget(self.menu, 0, 0)

        # self.show_menu = QtGui.QMenu("Show")
        # self.menu.addMenu(self.show_menu)
        # self.groupby_menu = QtGui.QMenu("Group by")
        # self.menu.addMenu(self.groupby_menu)

        self.sweep_tree = QtGui.QTreeWidget()
        self.sweep_tree.setColumnCount(3)
        self.sweep_tree.setHeaderLabels(['Stim Name', 'Clamp Mode', 'Holding'])
        self.sweep_tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.layout.addWidget(self.sweep_tree, 1, 0)

        self.meta_tree = pg.DataTreeWidget()
        self.layout.addWidget(self.meta_tree, 2, 0)

        self.setNwb(nwb)

        self.sweep_tree.itemSelectionChanged.connect(self._selection_changed)

    def setNwb(self, nwb):
        self._nwb = nwb
        self.update_sweep_tree()

    def update_sweep_tree(self):
        for group in self._nwb.sweep_groups():
            meta = [group.sweeps[0].traces()[ch].meta() for ch in group.sweeps[0].channels()]

            mode = ['V' if m.get('Clamp Mode', '') == 0 else 'I' for m in meta]
            holding = [m.get('V-Clamp Holding Level', '') for m in meta]
            holding = ' '.join(['%0.1f'%h if h is not None else '__._' for h in holding])
            mode = ' '.join(mode)

            gitem = QtGui.QTreeWidgetItem([meta[0]['stim_name'], str(mode), str(holding)])
            gitem.data = group
            self.sweep_tree.addTopLevelItem(gitem)
            for sweep in group.sweeps:
                item = QtGui.QTreeWidgetItem([str(sweep.sweep_id)])
                item.data = sweep
                gitem.addChild(item)

        self.sweep_tree.header().resizeSections(QtGui.QHeaderView.ResizeToContents)

    def selection(self):
        """Return a list of selected groups and/or sweeps. 
        """
        items = self.sweep_tree.selectedItems()
        selection = []
        for item in items:
            if item.parent() in items:
                continue
            selection.append(item.data)
        return selection

    def _selection_changed(self):
        sel = self.selection()
        if len(sel) == 1:
            self.meta_tree.setData(sel[0].meta())
        else:
            self.meta_tree.clear()
        self.selection_changed.emit(sel)


class MiesNwbViewer(QtGui.QWidget):
    def __init__(self, nwb):
        QtGui.QWidget.__init__(self)
        self.nwb = nwb 
        
        self.layout = QtGui.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.hsplit = QtGui.QSplitter()
        self.hsplit.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.hsplit, 0, 0)
        
        self.vsplit = QtGui.QSplitter()
        self.vsplit.setOrientation(QtCore.Qt.Vertical)
        self.hsplit.addWidget(self.vsplit)

        self.explorer = MiesNwbExplorer(self.nwb)
        self.explorer.selection_changed.connect(self.data_selection_changed)
        self.vsplit.addWidget(self.explorer)

        self.ptree = pg.parametertree.ParameterTree()
        self.vsplit.addWidget(self.ptree)
        
        self.tabs = QtGui.QTabWidget()
        self.hsplit.addWidget(self.tabs)

        self.views = [
            ('Sweep', SweepView(self)),
            ('Matrix', MultipatchMatrixView(self)),
        ]

        for name, view in self.views:
            self.tabs.addTab(view, name)

        self.resize(1000, 800)
        self.hsplit.setSizes([150, 850])

        self.tab_changed()
        self.tabs.currentChanged.connect(self.tab_changed)

    def data_selection_changed(self, selection):
        sweeps = self.selected_sweeps(selection)
        self.tabs.currentWidget().show_sweeps(sweeps)

    def tab_changed(self):
        w = self.tabs.currentWidget()
        self.ptree.setParameters(w.params, showTop=False)
        sweeps = self.selected_sweeps()
        w.show_sweeps(sweeps)

    def selected_sweeps(self, selection=None):
        if selection is None:
            selection = self.explorer.selection()
        sweeps = []
        for item in selection:
            if isinstance(item, SweepGroup):
                sweeps.extend(item.sweeps)
            else:
                sweeps.append(item)
        return sweeps


class SweepView(QtGui.QWidget):
    def __init__(self, parent=None):
        self.sweeps = []

        QtGui.QWidget.__init__(self, parent)

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plots = PlotGrid(parent=self)
        self.layout.addWidget(self.plots, 0, 0)

        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'lowpass', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 0.1},
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
        data = MiesNwb.pack_sweep_data(sweeps)
        data, stim = data[...,0], data[...,1]  # unpack stim and recordings
        dt = sweeps[0].traces()[0].meta()['Minimum Sampling interval']
        t = np.arange(data.shape[2]) * dt

        self.plots.clear()
        self.plots.set_shape(data.shape[1], 1)
        self.plots.setClipToView(True)
        self.plots.setDownsampling(True, True, 'peak')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt = self.plots[j, 0]
                # a = 255. * 0.8 / data.shape[0]
                a = 200
                plt.plot(t, self.filter(data[i, j]), pen=(255, 255, 255, a), antialias=True)

        for j in range(1, data.shape[1]):
            self.plots[j, 0].setXLink(self.plots[0, 0])

    def filter(self, data):
        lp = self.params['lowpass']
        if lp > 0:
            data = gaussian_filter(data, lp)
        return data


class MultipatchMatrixView(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plots = PlotGrid(parent=self)
        self.layout.addWidget(self.plots, 0, 0)

        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'lowpass', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 0.1},
        ])

    def show_group(self, grp):
        self.show_sweeps(grp.sweeps)

    def show_sweeps(self, sweeps):
        if len(sweeps) == 0:
            self.plots.clear()
            return
        data = MiesNwb.pack_sweep_data(sweeps)
        data, stim = data[...,0], data[...,1]  # unpack stim and recordings
        dt = sweeps[0].traces()[0].meta()['Minimum Sampling interval']

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
                    on = on_times[i][k][0]
                    off = off_times[i][k][0]
                    r = 10  # flatten 10 samples following each transient
                    data[:,j,on:on+r] = data[:,j,on:on+1]
                    data[:,j,off:off+r] = data[:,j,off:off+1]

        data = data.mean(axis=0)
        data = gaussian_filter(data, (0, 15))

        self.plots.set_shape(data.shape[0], data.shape[0])
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                plt = self.plots[i, j]
                start = on_times[j][0][0] - 1000
                stop = on_times[j][2][0] + 1000  # only look at the first 3 spikes

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

                plt.plot(np.arange(len(seg))*dt, seg, clear=True, pen={'color': color, 'width': 1}, antialias=True)
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
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.rows = 0
        self.cols = 0
        self.plots = []
        
        self.layout = QtGui.QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
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

    def _call_on_plots(self, m, *args, **kwds):
        for row in self.plots:
            for plt in row:
                getattr(plt, m)(*args, **kwds)

    # wrap a few methods from PlotItem:
    def clear(self):
        self._call_on_plots('clear')

    def setClipToView(self, clip):
        self._call_on_plots('setClipToView', clip)

    def setDownsampling(self, *args, **kwds):
        self._call_on_plots('setDownsampling', *args, **kwds)



        
if __name__ == '__main__':
    import sys
    from pprint import pprint
    pg.dbg()
    
    filename = sys.argv[1]
    nwb = MiesNwb(filename)
    # sweeps = nwb.sweeps()
    # traces = sweeps[0].traces()
    # # pprint(traces[0].meta())
    # groups = nwb.sweep_groups()
    # for i,g in enumerate(groups):
    #     print "--------", i, g
    #     print g.describe()

    # d = groups[7].data()
    # print d.shape

    app = pg.mkQApp()
    w = MiesNwbViewer(nwb)
    w.show()
    # w.show_group(7)
