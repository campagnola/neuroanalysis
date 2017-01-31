import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


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
                p = self[i, j]
                p.hide()
                p.close()
                p.scene().removeItem(p)
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

    def enableAutoRange(self, *args, **kwds):
        self._call_on_plots('enableAutoRange', *args, **kwds)

