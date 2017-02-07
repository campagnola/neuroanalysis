import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.parametertree as pt
import numpy as np
import scipy.ndimage as ndi

from ..event_detection import zero_crossing_events


class EventDetector(QtCore.QObject):
    """Analyzer to detect transient events from signals such as calcium indicator
    and current clamp recordings.
    
    The basic algorithm is:
    
    1. Lowpass input signal using gaussian filter
    2. Exponential deconvolution to isolate spikes
       (Richardson & Silberberg, J. Neurophysiol 2008)
    3. Threshold detection of events


    Signals
    -------
    parameters_changed(self):
        Emitted whenever a parameter has changed that would affect the output
        of the analyzer.
        
    """
    parameters_changed = QtCore.Signal(object)  # self
    
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.params = pt.Parameter(name='Spike detection', type='group', children=[
            {'name': 'gaussian sigma', 'type': 'float', 'value': 2.0},
            {'name': 'deconv const', 'type': 'float', 'value': 0.04, 'step': 0.01},
            {'name': 'threshold', 'type': 'float', 'value': 0.05, 'step': 0.01},
        ])
        self.sig_plot = None
        self.deconv_plot = None
        
        self.sig_trace = None
        self.vticks = None
        self.deconv_trace = None
        self.threshold_line = None
        
        self.params.sigTreeStateChanged.connect(self._parameters_changed)
        self.params.child('threshold').sigValueChanged.connect(self._threshold_param_changed)
        
    def set_plots(self, plt1=None, plt2=None):
        """Connect this detector to two PlotWidgets where data should be displayed.
        
        The first plot will contain the lowpass-filtered trace and tick marks
        for detected events. The second plot will contain the deconvolved signal
        and a draggable threshold line.
        """
        self.sig_plot = plt1
        if plt1 is not None:
            if self.sig_trace is None:
                self.sig_trace = pg.PlotDataItem()
                self.vticks = pg.VTickGroup(yrange=[0.0, 0.05])
            plt1.addItem(self.sig_trace)
            plt1.addItem(self.vticks)
        
        self.deconv_plot = plt2
        if plt2 is not None:
            if self.deconv_trace is None:
                self.deconv_trace = pg.PlotDataItem()
                self.threshold_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
                self.threshold_line.setValue(self.params['threshold'])
                self.threshold_line.sigPositionChanged.connect(self._threshold_line_moved)
            plt2.addItem(self.deconv_trace)
            plt2.addItem(self.threshold_line)
        
    def process(self, t, y, show=True):
        """Return a table (numpy record array) of events detected in a time series.
        
        Parameters
        ----------
        t : ndarray
            Time values corresponding to sample data.
        y : ndarray
            Signal values to process for events (for example, a single calcium
            signal trace or a single electrode recording).
        show : bool
            If True, then processed data will be displayed in the connected
            plots (see `set_plots()`).
        
        Returns
        -------
        events : numpy record array
            The returned table has several fields:
            
            * index: the index in *data* at which an event began
            * len: the length of the deconvolved event in samples
            * sum: the integral of *data* under the deconvolved event curve
            * peak: the peak value of the deconvolved event
        """
        filtered = ndi.gaussian_filter(y, self.params['gaussian sigma'])
        
        # Exponential deconvolution; see Richardson & Silberberg, J. Neurophysiol 2008
        diff = np.diff(filtered) + self.params['deconv const'] * filtered[:-1]
        
        self.events = zero_crossing_events(diff, min_peak=self.threshold_line.value())
        self.events = self.events[self.events['sum'] > 0]
        self.vticks.setXVals(t[self.events['index']])
        self.vticks.update()

        if show:
            if self.sig_plot is not None:
                self.sig_trace.setData(t[:len(filtered)], filtered)
                self.vticks.setXVals(t[self.events['index']])
                self.vticks.update()  # this should not be needed..
            if self.deconv_plot is not None:
                self.deconv_trace.setData(t[:len(diff)], diff)

        return self.events
        
    def _parameters_changed(self):
        self.parameters_changed.emit(self)
    
    def _threshold_line_moved(self):
        # link line position to threshold parameter
        self.params.child('threshold').setValue(self.threshold_line.value(), blockSignal=self._threshold_param_changed)
        
    def _threshold_param_changed(self):
        # link line position to threshold parameter
        if self.threshold_line is not None:
            self.threshold_line.setValue(self.params['threshold'])
    

