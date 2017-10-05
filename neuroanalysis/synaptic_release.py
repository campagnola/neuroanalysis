from collections import OrderedDict
import scipy.optimize as scimin
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy
import lmfit

initial_params = {
            'Tau_f':100.0, 'Tau_FDR':300.0,
            'a_FDR':0.001, 'Tau_i':1000.0, 'a_i':0.1, 'Tau_D':30.0,
            'a_D':0.1, 'Tau_r0':300.0, 'p0bar':0.1, 'Tau_r':100.0,'p0':0.1
        }
        

bound_params={
            'Tau_f':(0.0,5000.0), 
            'Tau_FDR':(0.0,1000.), 'a_FDR':(0.0,1000.0), 'Tau_i':(0.,5000.0),'a_i':(-10.0,10.0),
            'Tau_D':(1.0, 100.0), 'a_D': (.0, 10.0), 'Tau_r0':(0.0, 3000.0), 
            'p0bar':(0.0, 1.0),'Tau_r':(0.0,5000.0),'p0':(0.,1.0)
        }
param_order = [
            'Tau_r0', 'a_FDR', 'Tau_FDR', 'Tau_f', 'p0bar', 
            'a_i', 'Tau_i', 'a_D', 'Tau_D', 'Tau_r', 'p0'
        ]

dynamics_reverse_order={'0':'Dep','1':'Fac','2':'UR','3':'SMR','4':'DSR'}
ode_variables_reverse_order={'0':'n','1':'p','2':'Tau_r','3':'p0','4':'D'}
dynamics_order={'Dep':0,'Fac':1,'UR':2,'SMR':3,'DSR':4}
ode_variables_order={'n':0,'p':1,'Tau_r':2,'p0':3,'D':4}
dynamics_types = ['Dep', 'Fac', 'UR', 'SMR', 'DSR']


def f(y, t, all_dict,gating):
    # unpack params
    #a_n, Tau_r0, a_FDR, Tau_FDR, a_f, Tau_f, p0bar, a_i, Tau_i, a_D, Tau_D, blah = params
    n, p, Tau_r, p0, D = y      # unpack current values of y
    #print y
    #print n
    #print all_dict,gating
    derivs =numpy.zeros(len(y))
    #print Tau_r0, Tau_r
    initial_y=n
   # print gating
    if gating['Dep']==1:
        if gating['UR']==1:
            derivs[0] = (1-n) /Tau_r 
        else:
            derivs[0] = (1-n) / all_dict['Tau_r']
        
    if gating['Fac']==1:
        if gating['SMR']==1:
            derivs[1] = (p0-p) / all_dict['Tau_f']
        else:
            derivs[1] = (all_dict['p0']-p) / all_dict['Tau_f']
    
    if gating['UR']==1:
        derivs[2] = (all_dict['Tau_r0']-Tau_r) /all_dict['Tau_FDR']

    if gating['SMR']==1:
        derivs[3] = -(all_dict['p0bar']-p0) /all_dict['Tau_i']

    if gating['DSR']==1:
        derivs[4] = (1-D) /all_dict['Tau_D']
    #print initial_y
    return derivs

def feval(spikes, length_array, dynamics,ode_variables, Tau_r0, a_FDR, Tau_FDR, p0, Tau_f, p0bar, a_i, Tau_i, a_D, Tau_D, Tau_r,): 
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
  
    #print spikes
    length=length_array
    spikes_cont=[]
    st=int(0)
    for xin in length:
        spikes_cont.append(spikes[st:st+int(xin)])
        #print datax[st:st+xin]
        st=st+int(xin)
    #print spikes_cont
    
    
    param_dict={}
    
    #param_dict['a_n']=a_n
    param_dict['Tau_r0']=Tau_r0
    param_dict['a_FDR']=a_FDR
    param_dict['Tau_FDR']=Tau_FDR
    #param_dict['a_f']=a_f
    param_dict['Tau_f']=Tau_f
    param_dict['p0bar']=p0bar
    param_dict['a_i']=a_i
    param_dict['Tau_i']=Tau_i
    param_dict['a_D']=a_D
    param_dict['p0']=p0
    param_dict['Tau_D']=Tau_D
    param_dict['Tau_r']=Tau_r
    #print Tau_r

    gating={}
    for di, dn in enumerate(dynamics):
        gating[dynamics_reverse_order[str(di)]]=dn
    #print gating
    ode_vs={}
    for di, dn in enumerate(ode_variables):
        ode_vs[ode_variables_reverse_order[str(di)]]=dn
    
    dt=1.0
    return_v=[]

    for ex in spikes_cont:
        spike_times=numpy.array(ex)
        y=numpy.ones(5)
        y[1]=param_dict['p0']
        p0_initial=y[1]
        if ode_vs['Tau_r']==1:
            y[2]=param_dict['Tau_r0'] # initial value
        if ode_vs['p0']==1:
            y[3]=param_dict['p0bar']  # intial value
		      
        
        nspikes = len(spike_times)
        #print nspikes
        output = numpy.zeros((nspikes, 2 + len(y)))
        yw = numpy.zeros((nspikes, 1))    
        output[0, 0] = 0
        yw[0, 0] = 1 # p should be renormalized to the p0.
        output[0, 2:] = y
        
        # Loop over spikes and evaluation timepoints
        # Each spike causes an instantaneous change in state parameters,
        # and then we integrate the ode until the next spike.
        for i in range(1, nspikes):
            # Instantaneous changes in state induced by spike
            yp = y[:]
            y01 = y[0]*y[1]
           
            if gating['Dep']==1:
                #y01 *= 
                #if gating['Fac']==1:
                #    y01 *= y[1]
                yp[0] -=y01
                
		        
            if gating['Fac']==1:
                yp[1] +=param_dict['p0']*(1-y[1])
		        
            if gating['UR']==1:
                yp[2] -= param_dict['a_FDR'] * y[2]
		        
            if gating['SMR']==1:
                yp[3] -=param_dict['a_i'] * y[3]

            if gating['DSR']==1:
                yp[4] -=param_dict['a_D'] * y01 * y[4]
                
            y = yp[:]
     
            # Integrate until next spike
            time_pts = numpy.arange(spike_times[i-1], spike_times[i], dt) + dt
            #print time_pts
            psoln = odeint(f, y, time_pts, args=(param_dict,gating))
               
            # Last time point becomes the initial state for the next spike
            y = psoln[-1]
     
            
            yw[i,0]=y[0]*y[1]/p0_initial*y[4]
            #print y[1],yw[1],p0_initial
            #print y    
            # Record state for this timepoint
            output[i, 0] = time_pts[-1]
            output[i, 2:] = psoln[-1]  
            #Generate n*p*D column
        output[:, 1] = yw[:,0]
        return_v.extend(output[:, 1])
        #print ex,i,len(output[:, 1])
    #print numpy.array(return_v)    
    return numpy.array(return_v)

class ReleaseModel(object):
    """Presynaptic release model based on Hennig et al. 2013
        The input should be a list of tuples [(x, y), ...], where each tuple
        contains the spike times (x) and PSG amplitudes (y) to fit against.
        
        Returns amplitudes predicted by model and fit results provided by lmfit
        The shape of amplitudes is the same as that of spike_set (the inputs).
    
    Jung Hoon Lee <jungl@alleninstitute.org>
    
    """
    Dynamics = {'Dep':1, 'Fac':1, 'UR':1, 'SMR':1, 'DSR':1}
    ode_variables={'n':1,'p':1,'Tau_r':1,'p0':1,'D':1}
    def __init__(self):
        # Dep: depression
        # Fac: facilitation
        # UR: Use-dependent replentishment
        # SMR: Slow modulation of release
        # DSR: Receptor desensitization
       
        
        self.dict_params =initial_params
        
        self.order =param_order
       
        self.dict_bounds=bound_params
       
        print self.order

    def run_fit(self, spike_sets):
        """Fit the model against data.
        
        The input should be a list of tuples [(x, y), ...], where each tuple
        contains the spike times (x) and PSG amplitudes (y) to fit against.
        
        Returns a dictionary of optimal model parameters.        
        """
        #print spike_sets
        self.data_x=[]
        self.data_y=[]
        self.data_e=[]
        lengths=[]
        for datax, datay in spike_sets:
            self.data_x.extend(datax)
            self.data_y.extend(datay)
            #self.data_e.extend(dataz)
            lengths.append(len(datax))
        print self.Dynamics

        params=lmfit.Parameters()
        self.Sel_gatings=[]
        if self.Dynamics['Dep']==1:
            if self.Dynamics['Fac']==0:
                self.Sel_gatings.append('p0')
            if self.Dynamics['UR']==1:
                self.Sel_gatings.append('Tau_r0')
                self.ode_variables['Tau_r']=1
            else:
                self.ode_variables['n']=1
                self.Sel_gatings.append('Tau_r')
                
        else:
            self.ode_variables['n']=0

        if self.Dynamics['Fac']==1:
            #self.Sel_gatings.append('a_f')            
            self.Sel_gatings.append('Tau_f')
            self.ode_variables['p']=1
            if self.Dynamics['SMR']==1:
                self.Sel_gatings.append('p0bar')
                self.ode_variables['p0']=1
            else:
                self.Sel_gatings.append('p0')
        else:
            self.ode_variables['p']=0

        if self.Dynamics['UR']==1:
            self.Sel_gatings.append('a_FDR')
            self.Sel_gatings.append('Tau_FDR')
           
            if self.Dynamics['Dep']==0:
                self.Sel_gatings.append('Tau_r0')
                self.ode_variables['Tau_r']=1
        else:
            self.ode_variables['Tau_r']=0

        if self.Dynamics['SMR']==1:
            self.Sel_gatings.append('a_i')
            self.Sel_gatings.append('Tau_i')
            if self.Dynamics['Fac']==0:
                self.Sel_gatings.append('p0bar')
                self.ode_variables['p0']=1
        else:
            self.ode_variables['p0']=0

        if self.Dynamics['DSR']==1:
            self.Sel_gatings.append('a_D')
            self.Sel_gatings.append('Tau_D')
            self.ode_variables['D']=1
        else:
            self.ode_variables['D']=0
        dynamics_vec=numpy.zeros(5)
        ode_variables_vec=numpy.zeros(5)
        for dn in dynamics_order:
            dynamics_vec[dynamics_order[dn]]=self.Dynamics[dn]
        
        for dn in ode_variables_order:
            ode_variables_vec[ode_variables_order[dn]]=self.ode_variables[dn]
        self.freeparam=0
        fitmodel=lmfit.Model(feval,independent_vars=['spikes','length_array','dynamics', 'ode_variables'])
        for pm in self.order:
            #print 'pm0',pm
            if pm in self.Sel_gatings:
                print 'variable pm',pm
                self.freeparam=self.freeparam+1
                fitmodel.set_param_hint(pm,vary=True,value=self.dict_params[pm],min=self.dict_bounds[pm][0],max=self.dict_bounds[pm][1])
            else:
                print 'fixed',pm
                fitmodel.set_param_hint(pm,vary=False,value=self.dict_params[pm],min=self.dict_bounds[pm][0],max=self.dict_bounds[pm][1])
        pars=fitmodel.make_params()
        print pars
        result = fitmodel.fit(numpy.array(self.data_y),spikes=numpy.array(self.data_x),length_array=lengths,dynamics=dynamics_vec,ode_variables=ode_variables_vec)
        print(result.fit_report())
        
        self.model_y=result.eval()
        model_ys=[]
        ct=0
        for xi,xin in enumerate(lengths):
            model_ys.append(self.model_y[ct:ct+xin])
            ct=ct+xin
        
        return model_ys,result

    def goodness_of_fit(self):
        data_mean=numpy.mean(self.data_y)
        ss_t=0
        ss_res=0
        p=float(self.freeparam)
        n=float(len(self.data_y))
        for xi,xin in enumerate(self.data_y):
            ss_t=ss_t+(xin-data_mean)**2
            ss_res=ss_res+(xin-self.model_y[xi])**2
        r_square=1-float(ss_res)/float(ss_t)
        adj_r_square=r_square-(1-r_square)*p/(n-p-1)
        R_s=numpy.array([r_square,adj_r_square])
        print R_s

    
