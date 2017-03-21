from collections import OrderedDict
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.parametertree


class FitExplorer(QtGui.QWidget):
    def __init__(self, model, data=None, args=None, fit=None):
        QtGui.QWidget.__init__(self)
        self.model = model
        self.fit = None
        self.data = data if data is not None else fit.data
        self.args = args if args is not None else fit.userkws
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        self.ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.splitter.addWidget(self.ptree)
        self.plot = pg.PlotWidget()
        self.splitter.addWidget(self.plot)
        
        self.params = pg.parametertree.Parameter.create(name='param_root', type='group',
            children=[
                dict(name='initial parameters', type='group'),
                dict(name='constraints', type='group'),
                dict(name='fit', type='action'),
                dict(name='fit parameters', type='group'),
            ])
        
        for k in self.model.param_names:
            for pname in ['initial parameters', 'fit parameters']:
                p = pg.parametertree.Parameter.create(name=k, type='float', value=0, dec=True, minStep=1e-12, step=0.2)
                self.params.child(pname).addChild(p)
                
            p = ConstraintParameter(self.params.child('initial parameters', k))
            self.params.child('constraints').addChild(p)
                
            
        if fit is not None:
            self.set_fit(fit)
            
        self.ptree.setParameters(self.params, showTop=False)
        
        self.update_plots()
        
        self.params.sigTreeStateChanged.connect(self.update_plots)
        self.params.child('fit').sigActivated.connect(self.refit)

    def set_fit(self, fit):
        self.fit = fit
        self._fill_init_params(self.params.child('initial parameters'), self.fit)
        self._fill_constraints(self.params.child('constraints'), self.fit)
        self._fill_params(self.params.child('fit parameters'), self.fit)
        
    def update_plots(self):
        self.plot.clear()

        args = self.args.copy()
        self.plot.plot(args['x'], self.data)

        args.update(self.initial_params())
        self.plot.plot(args['x'], self.model.eval(y=self.data, **args), pen='y')

        args.update(self.fit_params())
        self.plot.plot(args['x'], self.model.eval(y=self.data, **args), pen='g')

    def fit_params(self):
        params = OrderedDict()
        for ch in self.params.child('fit parameters').children():
            params[ch.name()] = ch.value()
        return params
        
    def initial_params(self):
        params = OrderedDict()
        for ch in self.params.child('initial parameters').children():
            params[ch.name()] = ch.value()
        return params

    def constraints(self):
        params = OrderedDict()
        for ch in self.params.child('constraints').children():
            params[ch.name()] = ch.constraint()
        return params

    def refit(self):
        args = self.args.copy()
        args.update(self.constraints())
        self.fit = self.model.fit(self.data, **args)
        self._fill_params(self.params.child('fit parameters'), self.fit)
        
    def _fill_params(self, root, fit):
        for k in self.model.param_names:
            root[k] = fit.params[k].value

    def _fill_init_params(self, root, fit):
        for k in self.model.param_names:
            root[k] = fit.params[k].init_value

    def _fill_constraints(self, root, fit):
        for k in self.model.param_names:
            root.child(k).set_from_param(fit.params[k])


class ConstraintParameter(pg.parametertree.parameterTypes.GroupParameter):
    def __init__(self, source_param):
        self.source = source_param
        pg.parametertree.parameterTypes.GroupParameter.__init__(self, name=self.source.name(), children=[
            {'name': 'type', 'type': 'list', 'values': ['unconstrained', 'range', 'fixed']},
            {'name': 'min', 'type': 'float', 'visible': False, 'dec': True, 'minStep': 1e-12},
            {'name': 'max', 'type': 'float', 'visible': False, 'dec': True, 'minStep': 1e-12},
        ])
        
        self.child('type').sigValueChanged.connect(self.type_changed)
        
    def type_changed(self):
        isrange = self['type'] == 'range'
        self.child('min').setOpts(visible=isrange)
        self.child('max').setOpts(visible=isrange)

    def constraint(self):
        sval = self.source.value()
        if self['type'] == 'unconstrained':
            return sval
        elif self['type'] == 'range':
            return (sval, self['min'], self['max'])
        else:
            return (sval, 'fixed')
        
    def set_constraint(self, c):
        if isinstance(c, tuple):
            if c[1] == 'fixed':
                self['type'] = 'fixed'
            else:
                self['type'] = 'range'
                self['min'] = c[1]
                self['max'] = c[2]
        else:
            self['type'] = 'unconstrained'

    def set_from_param(self, param):
        if not param.vary:
            self['type'] = 'fixed'
        else:
            if not np.isfinite(param.min) and not np.isfinite(param.max):
                self['type'] = 'unconstrained'
            else:
                self['type'] = 'range'
                self['min'] = param.min
                self['max'] = param.max
