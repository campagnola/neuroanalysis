"""
Derived from acq4 and cnmodel code originally developed by Luke Campagnola and Paul B. Manis,
Univerity of North Carolina at Chapel Hill.
"""
import lmfit
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class FitModel(lmfit.Model):
    """ Simple extension of lmfit.Model that allows one-line fitting.
    
    Each subclass of lmfit.Model describes a single mathematical function to be fit.
    The major benefit of lmfit is that it allows any of the function parameters to
    be fixed or bounded arbitrarily.

    Example uses:
        
        # single exponential fit
        fit = expfitting.Exp1.fit(data, 
                x=time_vals,
                xoffset=(0, 'fixed'),           # x offset is fixed at 0
                yoffset=(yoff_guess, -120, 0),  # y offset is bounded between -120 and 0
                amp=(amp_guess, 0, 50),         # amp is bounded between 0 and 50
                tau=(tau_guess, 0.1, 50))       # tau is bounded between 0.1 and 50
        
        # plot the fit
        fit_curve = fit.eval()
        plot(time_vals, fit_curve)
        
    
        # double exponential fit with tau ratio constraint
        # note that 'tau_ratio' does not appear in the exp2 model; 
        # we can define new parameters here.
        fit = expfitting.Exp2.fit(data, 
                x=time_vals,
                xoffset=(0, 'fixed'),
                yoffset=(yoff_guess, -120, 0),
                amp1=(amp_guess, 0, 50),
                tau1=(tau_guess, 0.1, 50),
                amp2=(-0.5, -50, 0),
                tau_ratio=(10, 3, 50),          # tau_ratio is bounded between 3 and 50
                tau2='tau1 * tau_ratio'         # tau2 is forced to be tau1 * tau_ratio 
                )
        
    """
    def fit(self, data, interactive=False, **params):
        """ Return a fit of data to this model.
        
        Parameters
        ----------
        data : array
            dependent data to fit
        interactive : bool
            If True, show a GUI used for interactively exploring fit parameters
            
        Extra keyword arguments are passed to make_params() if they are model
        parameter names, or passed directly to Model.fit() for independent
        variable names.
        """
        fit_params = {}
        model_params = {}
        for k,v in params.items():
            if k in self.independent_vars or k in ['weights', 'method', 'scale_covar', 'iter_cb', 'verbose', 'fit_kws']:
                fit_params[k] = v
            else:
                model_params[k] = v
        p = self.make_params(**model_params)
        fit = lmfit.Model.fit(self, data, params=p, **fit_params)
        if interactive:
            self.show_interactive(fit)
        return fit
        
    def make_params(self, **params):
        """
        Make parameters used for fitting with this model.
        
        Keyword arguments are used to generate parameters for the fit. Each 
        parameter may be specified by the following formats:
        
        param=value :
            The initial value of the parameter
        param=(value, 'fixed') :
            Fixed value for the parameter
        param=(value, min, max) :
            Initial value and min, max values, which may be float or None
        param='expression' :
            Expression used to compute parameter value. See:
            http://lmfit.github.io/lmfit-py/constraints.html#constraints-chapter
        """
        p = lmfit.Parameters()
        for k in self.param_names:
            p.add(k)
        
        for param,val in params.items():
            if param not in p:
                p.add(param)
                
            if isinstance(val, str):
                p[param].expr = val
            elif np.isscalar(val):
                p[param].value = val
            elif isinstance(val, tuple):
                if len(val) == 2:
                    assert val[1] == 'fixed'
                    p[param].value = val[0]
                    p[param].vary = False
                elif len(val) == 3:
                    p[param].value = val[0]
                    p[param].min = val[1]
                    p[param].max = val[2]
                else:
                    raise TypeError("Tuple parameter specifications must be (val, 'fixed')"
                                    " or (val, min, max).")
            else:
                raise TypeError("Invalid parameter specification: %r" % val)
            
        # set initial values for parameters with mathematical constraints
        # this is to allow fit.eval(**fit.init_params)
        global_ns = np.__dict__
        for param,val in params.items():
            if isinstance(val, str):
                p[param].value = eval(val, global_ns, p.valuesdict())
        return p

    def show_interactive(self, fit=None):
        """ Show an interactive GUI for exploring fit parameters.
        """
        if not hasattr(self, '_interactive_win'):
            self._interactive_win = FitExplorer(model=self, fit=fit)
        self._interactive_win.show()

        
class Exp(FitModel):
    """Single exponential fitting model.
    
    Parameters are xoffset, yoffset, amp, and tau.
    """
    def __init__(self):
        FitModel.__init__(self, self.exp, independent_vars=['x'])

    @staticmethod
    def exp(x, xoffset, yoffset, amp, tau):
        return yoffset + amp * np.exp(-(x - xoffset)/tau)


class Exp2(FitModel):
    """Double exponential fitting model.
    
    Parameters are xoffset, yoffset, amp, tau1, and tau2.
    
        exp2 = yoffset + amp * (exp(-(x-xoffset) / tau1) - exp(-(x-xoffset) / tau2))

    """
    def __init__(self):
        FitModel.__init__(self, self.exp2, independent_vars=['x'])

    @staticmethod
    def exp2(x, xoffset, yoffset, amp, tau1, tau2):
        xoff = x - xoffset
        out = yoffset + amp * (np.exp(-xoff/tau1) - np.exp(-xoff/tau2))
        out[xoff < 0] = yoffset
        return out


class Gaussian(FitModel):
    """Gaussian fitting model.
    
    Parameters are xoffset, yoffset, sigma, and amp.
    """
    def __init__(self):
        FitModel.__init__(self, self.gaussian, independent_vars=['x'])

    @staticmethod
    def gaussian(x, xoffset, yoffset, sigma, amp):
        return amp * np.exp(-((x-xoffset)**2) / (2 * sigma**2)) + yoffset


class Sigmoid(FitModel):
    """Sigmoid fitting model.
    
    Parameters are xoffset, yoffset, slope, and amp.
    """
    def __init__(self):
        FitModel.__init__(self, self.sigmoid, independent_vars=['x'])

    @staticmethod
    def sigmoid(v, xoffset, yoffset, slope, amp):
        return amp / (1.0 + np.exp(-slope * (x-xoffset))) + yoffset
    

class Psp(FitModel):
    """PSP-like fitting model defined as the product of rising and decaying exponentials.
    
    Parameters are xoffset, yoffset, rise_tau, decay_tau, amp, and rise_power.
    """
    # default guess / bounds:

    # if guess is None:
    #     guess = [
    #         (y.max()-y.min()) * 2,
    #         0, 
    #         x[-1]*0.25,
    #         x[-1]
    #     ]
    
    # if bounds is None:
    #     bounds = [[None,None]] * 4
    #     bounds[1][0] = -2e-3
    #     minTau = (x[1]-x[0]) * 0.5
    #     #bounds[2] = [minTau, None]
    #     #bounds[3] = [minTau, None]

    def __init__(self):
        FitModel.__init__(self, self.psp_func, independent_vars=['x'])

    @staticmethod
    def _psp_inner(x, rise, decay, power):
        out = np.zeros(x.shape, x.dtype)
        mask = x >= 0
        xvals = x[mask]
        out[mask] =  (1.0 - np.exp(-xvals / rise))**power * np.exp(-xvals / decay)
        return out

    @staticmethod
    def _psp_max_time(rise, decay, rise_power=2.0):
        """Return the time from start to peak for a psp with given parameters."""
        return rise * np.log(1 + (decay * rise_power / rise))

    @staticmethod
    def psp_func(x, xoffset, yoffset, rise_tau, decay_tau, amp, rise_power=2.0):
        """Function approximating a PSP shape. 

        Uses absolute value of both taus, so fits may indicate negative tau.
        """
        # first determine scaling factor needed to achieve correct amplitude
        rise_tau = abs(rise_tau)
        decay_tau = abs(decay_tau)
        max_x = Psp._psp_max_time(rise_tau, decay_tau, rise_power)
        max_val = (1.0 - np.exp(-max_x / rise_tau))**rise_power * np.exp(-max_x / decay_tau)

        return (amp / max_val) * Psp._psp_inner(x-xoffset, rise_tau, decay_tau, rise_power)


class Psp2(FitModel):
    """PSP-like fitting model with double-exponential decay.
    
    Shape is computed as the product of a rising exponential and the sum of two decaying exponentials.

    Parameters are xoffset, yoffset, slope, and amp.
    """
    def __init__(self):
        FitModel.__init__(self, self.double_psp_func, independent_vars=['x'])

    @staticmethod
    def double_psp_func(x, xoffset, yoffset, rise_tau, decay_tau1, decay_tau2, amp1, amp2, rise_power=2.0):
        """Function approximating a PSP shape with double exponential decay. 
        """
        x = x-xoffset
        
        out = np.zeros(x.shape, x.dtype)
        mask = x >= 0
        x = x[mask]
        
        rise_exp = (1.0 - np.exp(-x / rise_tau))**rise_power
        decay_exp1 = amp1 * np.exp(-x / decay_tau1)
        decay_exp2 = amp2 * np.exp(-x / decay_tau2)
        out[mask] = riseExp * (decay_exp1 + decay_exp2)

        return out
