import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
#import pyqtgraph.flowchart
import pyqtgraph.parametertree as pt
import numpy as np
import scipy.ndimage as ndi

import functions as fn




class SpikeDetector(QtGui.QWidget):
    def __init__(self, boc, expt_id, cell_id):
        self.boc = boc
        self.data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt_id)
        self.lsn_tmp = self.data_set.get_stimulus_template('locally_sparse_noise')
        self.lsn_off = self.lsn_tmp.copy()
        self.lsn_off[self.lsn_off == 255] = 127
        self.lsn_on = self.lsn_tmp.copy()
        self.lsn_on[self.lsn_on == 0] = 127
        
        cells = self.data_set.get_cell_specimen_ids()
        #cells.sort()
        self.cell_mask = (self.data_set.get_roi_mask_array() * np.array(cells)[:,None,None]).max(axis=0)
        self.cell_id = cell_id
        
        # make stimulus frame locations easier to look up
        self.lsn_id = None
        
        QtGui.QWidget.__init__(self)
        self.hs = pg.QtGui.QSplitter()
        self.hs.setOrientation(pg.QtCore.Qt.Horizontal)

        self.vs1 = pg.QtGui.QSplitter()
        self.vs1.setOrientation(pg.QtCore.Qt.Vertical)

        self.params = pt.Parameter(name='params', type='group', children=[
            #{'name': 'container id', 'type': 'list', 'value': cont_id, 'values': list(cont_ids)},
            {'name': 'cell id', 'type': 'list', 'value': cell_id, 'values': sorted(list(cells))},
            {'name': 'gaussian sigma', 'type': 'float', 'value': 2.0},
            {'name': 'deconv const', 'type': 'float', 'value': 0.04},
            {'name': 'on/off', 'type': 'list', 'values': ['on', 'off', 'any']},
            {'name': 'delay', 'type': 'float', 'value': 0.1, 'suffix': 's', 'siPrefix': True, 'step': 50e-3},
            {'name': 'delay range', 'type': 'float', 'value': 0, 'limits': [0,None], 'suffix': 's', 'siPrefix': True, 'step': 50e-3},
        ])

        self.tree = pt.ParameterTree(showHeader=False)
        self.tree.setParameters(self.params, showTop=False)
        self.vs1.addWidget(self.tree)

        self.expt_imv = pg.ImageView()
        self.cell_roi_img = pg.ImageItem()
        self.expt_imv.view.addItem(self.cell_roi_img)
        lut = np.zeros((256,3), dtype='ubyte')
        lut[:,2] = np.arange(256)
        self.cell_roi_img.setLookupTable(lut)
        self.cell_roi_img.setZValue(20)
        self.cell_roi_img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        self.expt_imv.view.scene().sigMouseClicked.connect(self.imageSceneClicked)
        
        self.vs1.addWidget(self.expt_imv)

        self.vs2 = pg.QtGui.QSplitter()
        self.vs2.setOrientation(pg.QtCore.Qt.Vertical)

        self.plt1 = pg.PlotWidget()
        self.trace = self.plt1.plot()
        self.evTicks = pg.VTickGroup(yrange=[0.0, 0.06])
        self.plt1.addItem(self.evTicks)
        self.plt2 = pg.PlotWidget()
        self.ftrace = self.plt2.plot()
        self.plt2.setXLink(self.plt1)

        self.sta_imv = pg.ImageView()

        self.vs2.addWidget(self.plt1)
        self.vs2.addWidget(self.plt2)
        self.vs2.addWidget(self.sta_imv)

        #fc = pg.flowchart.Flowchart(terminals={
            #'dataIn': {'io': 'in'},
            #'dataOut': {'io': 'out'}    
        #})
        #fc.setInput(dataIn=trace[1][0])
        #w.addWidget(fc.widget())

        self.tLine = self.plt2.addLine(y=0.05, movable=True, pen='g')

        self.hs.addWidget(self.vs1)
        self.hs.addWidget(self.vs2)
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.hs)
        
        self.show()

        self.params.sigTreeStateChanged.connect(self.paramsChanged)
        self.tLine.sigPositionChanged.connect(self.updateSpikes)
        #fc.sigStateChanged.connect(fcChanged)
        self.loadCell()

    def paramsChanged(self, root, changes):
        for param, change, val in changes:
            if param is self.params.child('cell id'):
                self.loadCell()
            elif param in (self.params.child('gaussian sigma'), self.params.child('deconv const')):
                self.updateSpikes()
            else:
                self.updateOutput()
                
    def loadCell(self):
        
        self.expt_imv.setImage(self.data_set.get_max_projection().T)
        
        cell_id = self.params['cell id']
        data = self.data_set.get_dff_traces([cell_id])
        
        if self.lsn_id is None:
            lsn_st = self.data_set.get_stimulus_table('locally_sparse_noise')
            self.lsn_id = np.zeros(len(data[0]), dtype='int') - 1
            for i,frame in lsn_st.iterrows():
                self.lsn_id[frame['start']:frame['end']] = frame['frame']
                
        self.cell_roi_img.setImage(self.data_set.get_roi_mask_array([cell_id])[0].T)
        
        self.updateSpikes()

    def updateSpikes(self):
        cell_id = self.params['cell id']
        self.data = self.data_set.get_dff_traces([cell_id])
        #filtered = fc['dataOut'].value()
        #if filtered is None:
            #filtered = trace[1][0]
        t = self.data[0]
        filtered = ndi.gaussian_filter(self.data[1][0], self.params['gaussian sigma'])
        self.trace.setData(t[:len(filtered)], filtered)
        
        # Exponential deconvolution; see Richardson & Silberberg, J. Neurophysiol 2008
        diff = np.diff(filtered) + self.params['deconv const'] * filtered[:-1]
        self.ftrace.setData(t[:len(diff)], diff)
        
        self.events = fn.zeroCrossingEvents(diff, minPeak=self.tLine.value())
        self.events = self.events[self.events['sum'] > 0]
        self.evTicks.setXVals(t[self.events['index']])
        self.evTicks.update()

        self.updateOutput()
        
    def updateOutput(self):
        t = self.data[0]
        dt = t[1] - t[0]
        inds = self.events['index'] - int(self.params['delay'] / dt)

        if self.params['on/off'] == 'on':
            lsn_frames = self.lsn_on
        elif self.params['on/off'] == 'off':
            lsn_frames = self.lsn_off
        else:
            lsn_frames = self.lsn_tmp
        
        dr = self.params['delay range']
        nframes = int(dr / dt)
        if nframes < 2:
            frames = self.lsn_id[inds]
            mask = frames > 0
            lsn_frames = lsn_frames[frames[mask]]
            sta = (lsn_frames * self.events['sum'][mask][:,None,None]).mean(axis=0)
            self.sta_imv.setImage(sta.T)
        else:
            offset = nframes // 2
            sta = np.empty((nframes,) + lsn_frames.shape[1:], float)
            for i in range(nframes):
                shift_inds = inds - offset + i
                mask = (shift_inds > 0) & (shift_inds < self.lsn_id.shape[0])
                frames = self.lsn_id[shift_inds[mask]]
                mask = frames > 0
                sta[i] = (lsn_frames[frames[mask]] * self.events['sum'][mask][:,None,None]).mean(axis=0)
            sta /= sta.mean(axis=1).mean(axis=1)[:,None,None]
            self.sta_imv.setImage(sta.transpose(0, 2, 1), xvals=np.arange(-offset, -offset+nframes) * dt)

    def imageSceneClicked(self, ev):
        pos = self.cell_roi_img.mapFromScene(ev.pos())
        cell_id = self.cell_mask[int(pos.y()), int(pos.x())]
        if cell_id == 0:
            return
        self.params['cell id'] = cell_id
