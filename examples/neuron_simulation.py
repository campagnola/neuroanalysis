"""
Demo of simple neuron simulator used for testing analysis routines.
"""

import numpy as np
import pyqtgraph as pg

import neuroanalysis.neuronsim as nsim
from neuroanalysis.units import MOhm, ms, us, pA, mS, cm, mV, um


# Initialize simulation container
sim = nsim.Sim()

# Create a single compartment
soma = nsim.Section(name='soma', radius=6*um)
sim.add(soma)

# Add channels to the membrane
mechs = [
    nsim.Leak(gbar=0.3*mS/cm**2, erev=-75*mV),
    nsim.LGKfast(gbar=500*mS/cm**2),
    nsim.LGKslow(),
    nsim.LGNa(),
#    nsim.Noise(),  # adds realism, but slows down the integrator a lot.
]
for m in mechs:
    soma.add(m)

# Add a patch clamp electrode
clamp = nsim.PatchClamp(mode='ic', ra=10*MOhm)
soma.add(clamp)


# Let the model run for a bit so the state can settle
a = sim.run(5000)


# Run an I/V curve protocol
pulse_dur = 100*ms
pulse_amps = np.linspace(-100, 100, 7) * pA

app = pg.mkQApp()
plot = pg.plot(labels={'left': ('soma.V', 'V'), 'bottom': ('time', 's')})

for i,pulse_amp in enumerate(pulse_amps):
    # generate command
    dt = sim.dt
    cmd = np.zeros(int(pulse_dur*2.5 / dt))
    start = int(pulse_dur*0.5 / dt)
    stop = start + int(pulse_dur / dt)
    cmd[start:stop] = pulse_amp
    
    # send command to the clamp
    clamp.queue_command(cmd, dt)

    # record response
    response = sim.run(len(cmd))
    
    # run a bit more to settle
    #sim.run(5000)

    # plot
    t = response['t']
    v = response['soma.V']
    plot.plot(t - t[0], v, pen=(i, len(pulse_amps)*1.5), antialias=True)
    app.processEvents()
