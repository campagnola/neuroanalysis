import numpy as np
from . import Sim, Section, Leak, LGKfast, LGKslow, LGNa, Noise, PatchClamp
from ..data import Trace, PatchClampRecording
from ..units import um, mS, mV, cm, MOhm


class ModelCell(object):
    """A simulated patch-clamped neuron for generating test data.
    """
    def __init__(self, radius=6*um, r_access=10*MOhm):
        self.sim = Sim()
        self._is_settled = False

        # Create a single compartment
        self.soma = Section(name='soma', radius=radius)
        self.sim.add(self.soma)

        # Add channels to the membrane
        self.mechs = {
            'leak': Leak(gbar=0.6*mS/cm**2, erev=-75*mV),
            'lgkfast': LGKfast(gbar=225*mS/cm**2),
            'lgkslow': LGKslow(gbar=0.225*mS/cm**2),
            'lgkna': LGNa(),
            'noise': Noise(),  # adds realism, but slows down the integrator a lot.
        }
        for m in self.mechs.values():
            self.soma.add(m)

        # Add a patch clamp electrode
        self.clamp = PatchClamp(name='electrode', mode='ic', ra=r_access)
        self.soma.add(self.clamp)

    def test(self, command, mode):
        """Send a command (Trace) to the electrode and return a PatchClampRecording
        that contains the result.
        """
        self.clamp.set_mode(mode)
        self.sim.dt = command.dt
        self._is_settled = False
        
        self.settle()
        
        # run simulation
        self.clamp.queue_command(command.data, command.dt)
        result = self.sim.run(len(command))
        
        # collect soma and pipette potentials
        t = result['t']
        vm = result['soma.V']
        response = Trace(vm, time_values=t)
        pip = result['electrode.V'] if mode == 'ic' else result['electrode.I']
        
        # Add in a little electrical recording noise
        enoise = 50e-6 if mode == 'ic' else 5e-12
        pip = pip + np.random.normal(size=len(pip), scale=enoise)
        
        recording = Trace(pip + enoise, time_values=t)
        
        channels = {'command': command, 'primary': recording, 'vsoma': response}
        kwds = {
            'clamp_mode': mode,
            'bridge_balance': 0,
            'lpf_cutoff': None,
            'pipette_offset': 0,
        }
        if mode == 'ic':
            kwds['holding_current'] = self.clamp.holding['ic']
        elif mode == 'vc':
            kwds['holding_potential'] = self.clamp.holding['vc']
            
        return PatchClampRecording(channels=channels, **kwds)
    
    def settle(self, t=1.0):
        """Run the simulation with no input to let it settle into steady state.
        """
        if self._is_settled:
            return
        n = int(t / self.sim.dt)
        self.mechs['noise'].enabled = False
        self.sim.run(n, hmax=1e-3)
        self.mechs['noise'].enabled = True
        self._is_settled = True
        
    def input_resistance(self):
        self.settle()
        return 1.0 / self.soma.conductance(self.sim.last_state)
    
    def capacitance(self):
        return self.soma.cap

    def resting_potential(self):
        self.settle()
        return self.sim.last_state['soma.V']
