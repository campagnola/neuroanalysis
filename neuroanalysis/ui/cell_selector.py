from collections import OrderedDict
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.parametertree as pt
import numpy as np


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


