import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from neuroanalysis.data import Trace
from neuroanalysis.ui.event_detection import EventDetector
from neuroanalysis.ui.plot_grid import PlotGrid


pg.mkQApp()

data = np.load("test_data/synaptic_events/events1.npz")
traces = [Trace(data['trace_%02d'%i], dt=1.0/data['sample_rates'][i]) for i in range(13)]

evd = EventDetector()
evd.params['threshold'] = 2e-11

hs = QtGui.QSplitter(QtCore.Qt.Horizontal)
pt = pg.parametertree.ParameterTree()
pt.setParameters(evd.params)
hs.addWidget(pt)

plots = PlotGrid()
plots.set_shape(2, 1)
plots.setXLink(plots[0, 0])
hs.addWidget(plots)

evd.set_plots(plots[0,0], plots[1,0])

def update():
    evd.process(traces[0].time_values, traces[0].data)

evd.parameters_changed.connect(update)
update()

hs.show()
