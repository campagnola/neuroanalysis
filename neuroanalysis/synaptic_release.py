from collections import OrderedDict
import scipy.optimize as scimin
from scipy.integrate import odeint
import numpy


class ReleaseModel(object):
    """Presynaptic release model based on:
    
        Hennig, M. H. (2013). Theoretical models of synaptic short term plasticity.
        Frontiers in Computational Neuroscience, 7(April), 1-10.
    
    Jung Hoon Lee <jungl@alleninstitute.org>
    """
    def __init__(self):
        self.Dynamics = {'Dep':1, 'Fac':1, 'UR':1, 'SMR':1, 'DSR':1}
        # Dep: depression
        # Fac: facilitation
        # UR: Use-dependent replentishment
        # SMR: Slow modulation of release
        # DSR: Receptor desensitization
        
        self.dict_params = {
            'a_n':0.2, 'Tau_f':100.0, 'a_f':0.01, 'Tau_FDR':100.0,
            'a_FDR':0.1, 'Tau_i':3000.0, 'a_i':-0.1, 'Tau_D':15.0,
            'a_D':0.2, 'Tau_r0':1500.0, 'p0bar':1.0, 'Tau_r':300.0,
        }
        
        self.order = [
            'a_n', 'Tau_r0', 'a_FDR', 'Tau_FDR', 'a_f', 'Tau_f', 'p0bar', 
            'a_i', 'Tau_i', 'a_D', 'Tau_D', 'Tau_r',
        ]

        self.dict_bounds={
            'a_n':(0.0,1.0),'Tau_f':(0.0,300.0), 'a_f':(0.0,1.0),
            'Tau_FDR':(0.0,1000.), 'a_FDR':(0.0,1.0), 'Tau_i':(0.,4000.0),'a_i':(-10.0,10.0),
            'Tau_D':(0.0, 50.0), 'a_D': (.0, 10.0), 'Tau_r0':(100.0, 3000.0), 
            'p0bar':(0.0, 10.0),'Tau_r':(0.0,1000.0),
        }
    
    def f(self, y, t, params):
        n, p, Tau_r, p0, D = y      # unpack current values of y
        a_n, Tau_r0, a_FDR, Tau_FDR, a_f, Tau_f, p0bar, a_i, Tau_i, a_D, Tau_D, blah = params
        derivs = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        if self.Dynamics['Dep']==1:
            if self.Dynamics['UR']==1:
                derivs[0] = (1-n) / Tau_r
            else:
                derivs[0] = (1-n) / blah
        
        if self.Dynamics['Fac']==1:
            if self.Dynamics['SMR']==1:
                derivs[1] = (p0-p) / Tau_f
            else:
                derivs[1] = (1-p) / Tau_f

        if self.Dynamics['UR']==1:
            derivs[2] = (Tau_r0-Tau_r) / Tau_FDR

        if self.Dynamics['SMR']==1:
            derivs[3] = -(p0bar-p0) / Tau_i

        if self.Dynamics['DSR']==1:
            derivs[4] = (1-D) / Tau_D

        return derivs

    def eval(self, spike_times, params, dt=5.0):
        """Evaluate release model, returning an array of state parameters
        for each timepoint in spike_times.
        
        Parameters
        ----------
        spike_times : list
            Times at which presynaptic spikes occur
        params : list
            Model parameters
        dt : float
            Time step (in ms) for integration between evaluated timepoints
        
        Returns
        -------
        output : array
            Array having shape (n_spikes, 7), where the rows are the model
            state parameters for each timepoint, and the columns are (time,
            amplitude (=n*p*D), n, p, tau_r, p0, D).
        """
        # unpack params
        a_n, Tau_r0, a_FDR, Tau_FDR, a_f, Tau_f, p0bar, a_i, Tau_i, a_D, Tau_D, blah = params
        
        # initialize state vector
        y = [
            1.0,    # n: initial value for depression
            1.0,    # p: initial value for facilitation
            Tau_r0, # Tau_r: initial value for desensitization
            p0bar,  # p0
            1.0,    # D
        ]
        
        nspikes = len(spike_times)
        output = numpy.zeros((nspikes, 2 + len(y)))
        output[0, 0] = 0
        output[0, 2:] = y
        
        # Loop over spikes and evaluation timepoints
        # Each spike causes an instantaneous change in state parameters,
        # and then we integrate the ode until the next spike.
        for i in range(1, nspikes):
            
            # Instantaneous changes in state induced by spike
            yp = y[:]
            y01 = 1.0
            if self.Dynamics['Dep']==1:
                y01 *= y[0]
                if self.Dynamics['Fac']==1:
                    y01 *= y[1]
                yp[0] -= a_n * y01
            
            if self.Dynamics['Fac']==1:
                yp[1] += a_f * (1 - y[1])
            
            if self.Dynamics['UR']==1:
                yp[2] -= a_FDR * y[2]
            
            if self.Dynamics['SMR']==1:
                yp[3] -= a_i * y[3]

            if self.Dynamics['DSR']==1:
                yp[4] -= a_D * y01 * y[4]
            
            y = yp[:]
            
            # Integrate until next spike
            time_pts = numpy.arange(spike_times[i-1], spike_times[i], dt) + dt
            psoln = odeint(self.f, y, time_pts, args=(params,))
            
            # Last time point becomes the initial state for the next spike
            y = psoln[-1]
            
            # Record state for this timepoint
            output[i, 0] = time_pts[-1]
            output[i, 2:] = psoln[-1]
        
        # Generate n*p*D column
        output[:, 1] = numpy.product(output[:, [2,3,6]], axis=1)
        
        return output

    def sum_residuals(self, params, spike_sets): # the function we want to minimize
        residuals = []
        for datax, datay in spike_sets:
            datax = numpy.array(datax)
            datay = numpy.array(datay)
            
            # Run model for these spike times   (column 1 contains n*p*D)
            modely = self.eval(datax, params)[:, 1]
            
            # Accumulate residual errors
            residuals.extend(datay - modely)
            
        return (numpy.array(residuals)**2).sum()

    def run_fit(self, spike_sets):
        """Fit the model against data.
        
        The input should be a list of tuples [(x, y), ...], where each tuple
        contains the spike times (x) and PSG amplitudes (y) to fit against.
        
        Returns a dictionary of optimal model parameters.        
        """
        init = [self.dict_params[param] for param in self.order]
        bounds = [self.dict_bounds[param] for param in self.order]
        
        # least square fit with constraints
        self.pwithout = scimin.fmin_slsqp(self.sum_residuals, init, bounds=bounds, args=(spike_sets,))

        return OrderedDict([(self.order[i], p) for i,p in enumerate(self.pwithout)])
