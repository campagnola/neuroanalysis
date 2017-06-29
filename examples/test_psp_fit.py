import numpy as np
import pyqtgraph as pg
from collections import OrderedDict
from neuroanalysis.fitting import Psp
from neuroanalysis.ui.fitting import FitExplorer

pg.mkQApp()


data = np.loadtxt('psp.csv', delimiter=',', skiprows=1, usecols=[0,1])
x = data[:,0]
y = data[:,1]


psp = Psp()
params = OrderedDict([
    ('xoffset', (2e-3, 5e-4, 5e-3)),
    ('yoffset', 0),
    ('amp', 10e-12),
    ('rise_time', (2e-3, 50e-6, 10e-3)),
    ('decay_tau', (4e-3, 500e-6, 50e-3)),
    ('rise_power', (2.0, 'fixed')),
])
fit = psp.fit(y*1e12, x=x, xtol=1e-3, maxfev=100, **params)


x = FitExplorer(fit=fit)
x.show()
