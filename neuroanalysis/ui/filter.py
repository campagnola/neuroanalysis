import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.parametertree as pt
from scipy.ndimage import gaussian_filter


class SignalFilter(QtCore.QObject):
    """A user-configurable signal filter.
    
    todo: turn this into a flowchart-based filter
    """
    def __init__(self):
        self.params = pt.Parameter.create(name='filter', type='group', children=[
            {'name': 'sigma', 'type': 'float', 'value': 200e-6, 'step': 1e-5, 'limits': [0, None], 'suffix': 's', 'siPrefix': True},
        ])
        
    def process(self, trace):
        data = gaussian_filter(trace.data, self.params['sigma'] / trace.dt)
        return trace.copy(data=data)
        
