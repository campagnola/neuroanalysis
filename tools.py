from collections import OrderedDict
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
#import pyqtgraph.flowchart
import pyqtgraph.parametertree as pt
import numpy as np
import scipy.ndimage as ndi

import functions as fn



class CellSelector(QtCore.QObject):
    """Select a single cell from a list of cells or from a pre-segmented image.
    
    Signals
    -------
    cell_selection_changed(id)
        Emitted when the selected cell ID has changed.
    """
    cell_selection_changed = QtCore.Signal(object)
    
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.params = pt.Parameter(name='Cell selection', type='group', children=[
            {'name': 'cell id', 'type': 'list', 'value': None, 'values': {'': None}},
        ])
        self.fluor_img = None
        self.roi_img = None
        self.cell_ids = []
        self.imv = None
        self.roi_img_item = None
        self.params.child('cell id').sigValueChanged.connect(self._selection_changed)
        
    def selected_id(self):
        """Return the currently selected cell ID.
        """
        return self.params['cell id']
        
    def set_cell_ids(self, ids):
        """Set the list of available cell IDs.
        
        Parameters
        ----------
        ids : list or ndarray
            Any sequence of integer IDs corresponding to the selectable cells.
        """
        self.cell_ids = ids
        opts = [('', None)] + [(str(i), i) for i in ids]
        self.params.child('cell id').setLimits(OrderedDict(opts))
        
    def set_images(self, fluor_img, roi_img, update_ids=True):
        """Set the images used for visual cell selection.
        
        Parameters
        ----------
        fluor_img : ndarray
            Fluorescence image of cells to display.
        roi_img : ndarray (integer dtype)
            Array containing the cell ID associated with each pixel in the 
            fluorescence image. Pixels with no associated cell should have
            negative value.
        update_ids : bool
            Set the cell ID list from the unique values in *rois*.
        """
        self.fluor_img = fluor_img
        self.roi_img = roi_img
        if update_ids:
            ids = np.unique(roi_img)
            self.set_cell_ids(ids[ids >= 0])
        self._update_images()
        
    def set_imageview(self, imv):
        """Connect this selector to an ImageView instance.
        
        This causes the fluorescence image and selected cell's ROI to be displayed
        in the view, and also allows cells to be selected by clicking in the view.
        
        Parameters
        ----------
        imv : pyqtgraph.ImageView
            The view widget to use for image display.
        """
        self.imv = imv
        self.roi_img_item = pg.ImageItem()
        
        imv.view.addItem(self.roi_img_item)
        lut = np.zeros((256,3), dtype='ubyte')
        lut[:,2] = np.arange(256)
        self.roi_img_item.setLookupTable(lut)
        self.roi_img_item.setZValue(20)
        self.roi_img_item.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        imv.view.scene().sigMouseClicked.connect(self._imview_clicked)
        self._update_images()
        
    def _update_images(self):
        if self.imv is None:
            return
        if self.fluor_img is not None:
            self.imv.setImage(self.fluor_img.T)
            
        cell_id = self.selected_id()
        if cell_id is not None:
            self.roi_img_item.setImage(self.roi_img.T == cell_id)

    def _imview_clicked(self, event):
        pos = self.roi_img_item.mapFromScene(event.pos())
        cell_id = self.roi_img[int(pos.y()), int(pos.x())]
        if cell_id < 0:
            return
        self.params['cell id'] = cell_id
    
    def _selection_changed(self):
        self._update_images()
        self.cell_selection_changed.emit(self.selected_id())


class SpikeDetector(QtCore.QObject):
    """Analyzer to generate spike metrics from a single calcium indicator trace.
    
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
        
        self.events = fn.zeroCrossingEvents(diff, minPeak=self.threshold_line.value())
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
    

class TriggeredAverager(QtCore.QObject):
    parameters_changed = QtCore.Signal(object)  # self

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.params = pt.Parameter(name='Triggered Average', type='group', children=[
            {'name': 'on/off', 'type': 'list', 'values': ['any', 'on', 'off']},
            {'name': 'delay', 'type': 'float', 'value': -0.2, 'suffix': 's', 'siPrefix': True, 'step': 50e-3},
            {'name': 'delay range', 'type': 'float', 'value': 1.0, 'limits': [0,None], 'suffix': 's', 'siPrefix': True, 'step': 50e-3},
            {'name': 'blur STA', 'type': 'float', 'value': 1.0, 'limits': [0,None], 'step': 0.5},
        ])
        self.imgview = None
        self.params.sigTreeStateChanged.connect(self.parameters_changed)
        
    def set_imageview(self, imv):
        self.imgview = imv
        
    def process(self, events, stimuli, stim_index, dt, show=True):
        inds = events['index'] - int(self.params['delay'] / dt)

        if self.params['on/off'] == 'on':
            stimuli = np.clip(stimuli, 127, 255)
        elif self.params['on/off'] == 'off':
            stimuli = np.clip(stimuli, 0, 127)
        
        dr = self.params['delay range']
        blur = self.params['blur STA']
        nframes = int(dr / dt)
        if nframes < 2:
            frames = stim_index[inds]
            mask = frames > 0
            stimuli = stimuli[frames[mask]]
            sta = (stimuli * events['sum'][mask][:,None,None]).mean(axis=0)
            if blur > 0:
                sta = ndi.gaussian_filter(sta, blur)
        else:
            offset = nframes // 2
            sta = np.empty((nframes,) + stimuli.shape[1:], float)
            for i in range(nframes):
                shift_inds = inds - offset + i
                mask = (shift_inds > 0) & (shift_inds < stim_index.shape[0])
                frames = stim_index[shift_inds[mask]]
                mask = frames > 0
                sta[i] = (stimuli[frames[mask]] * events['sum'][mask][:,None,None]).mean(axis=0)
            sta /= sta.mean(axis=1).mean(axis=1)[:,None,None]
            if blur > 0:
                sta = ndi.gaussian_filter(sta, (0, blur, blur))
        
        if self.imgview is not None:
            self.imgview.setImage(sta.transpose(0, 2, 1), xvals=np.arange(-offset, -offset+nframes) * dt)
            self.imgview.setCurrentIndex(sta.shape[0]/2)
        return sta


