from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.parametertree


class FitExplorer(QtGui.QWidget):
    def __init__(self, model, fit):
        QtGui.QWidget.__init__(self)
        self.model = model
        self.fit = fit
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        self.ptree = pg.parametertree.ParameterTree()
        self.splitter.addWidget(self.ptree)
        self.plot = pg.PlotWidget()
        self.splitter.addWidget(self.plot)
        
        self.params = pg.parametertree.Parameter.create(name='param_root', type='group',
            children=[
                dict(name='fit', type='action'),
                dict(name='parameters', type='group'),
            ])
        
        for k in fit.params:
            p = pg.parametertree.Parameter.create(name=k, type='float', value=fit.params[k].value)
            self.params.param('parameters').addChild(p)
            
        self.ptree.setParameters(self.params)
        
        self.update_plots()
        
        self.params.param('parameters').sigTreeStateChanged.connect(self.update_plots)
        
    def update_plots(self):
        for k in self.fit.params:
            self.fit.params[k].value = self.params['parameters', k]
            
        self.plot.clear()
        self.plot.plot(self.fit.data)
        self.plot.plot(self.fit.eval(), pen='y')

