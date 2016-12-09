import sys
import acq4
from neuroanalysis.nwb_viewer import MiesNwbViewer, MiesNwb
acq4.pyqtgraph.dbg()

m = acq4.Manager.Manager(argv=['-D', '-n', '-m', 'Data Manager'])
dm = m.getModule('Data Manager')
v = MiesNwbViewer()
v.show()
v.setWindowTitle('NWB Viewer')

def load_from_dm():
    v.set_nwb(MiesNwb(m.currentFile.name()))

btn = acq4.pyqtgraph.Qt.QtGui.QPushButton('load from data manager')
v.vsplit.insertWidget(0, btn)
btn.clicked.connect(load_from_dm)


if sys.flags.interactive == 0:
    acq4.pyqtgraph.QtGui.QApplication.exec_()
