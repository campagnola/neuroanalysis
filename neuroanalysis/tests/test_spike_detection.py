import numpy as np
from neuroanalysis.data import Recording, Trace
from neuroanalysis.neuronsim.model_cell import ModelCell
from neuroanalysis.units import pA, mV, MOhm, pF, us, ms
from neuroanalysis.spike_detection import detect_evoked_spike


def test_spike_detection():
    # Need to fill this function up with many more tests, especially 
    # measuring against real data.
    dt = 10*us
    start = 5*ms
    duration = 2*ms
    pulse_edges = int(start / dt), int((start+duration) / dt)

    resp = create_test_pulse(start=5*ms, pamp=100*pA, pdur=2*ms, mode='ic', dt=dt)
    spike = detect_evoked_spike(resp, pulse_edges)
    assert spike is None
    
    resp = create_test_pulse(start=5*ms, pamp=1000*pA, pdur=2*ms, mode='ic', dt=dt)
    spike = detect_evoked_spike(resp, pulse_edges)
    assert spike is not None


model_cell = ModelCell()

    
def create_test_pulse(start=5*ms, pdur=10*ms, pamp=-10*pA, mode='ic', dt=10*us, r_access=10*MOhm, c_soma=5*pF, noise=5*pA):
    # update patch pipette access resistance
    model_cell.clamp.ra = r_access
    
    # update noise amplitude
    model_cell.mechs['noise'].stdev = noise
    
    # make pulse array
    duration = start + pdur * 3
    pulse = np.zeros(int(duration / dt))
    pstart = int(start / dt)
    pstop = pstart + int(pdur / dt)
    pulse[pstart:pstop] = pamp
    
    # simulate response
    result = model_cell.test(Trace(pulse, dt), mode)

    return result


if __name__ == '__main__':
    import pyqtgraph as pg

    plt = pg.plot(labels={'left': ('Vm', 'V'), 'bottom': ('time', 's')})
    dt = 10*us
    start = 5*ms
    duration = 2*ms
    pulse_edges = int(start / dt), int((start+duration) / dt)

    # Iterate over a series of increasing pulse amplitudes
    for amp in np.arange(50*pA, 500*pA, 50*pA):
        # Simulate pulse response
        resp = create_test_pulse(start=start, pamp=amp, pdur=duration, mode='ic', r_access=100*MOhm)

        # Test spike detection
        spike = detect_evoked_spike(resp, pulse_edges)
        print(spike)
        pen = 'r' if spike is None else 'g'

        # plot in green if a spike was detected
        pri = resp['primary']
        pri.t0 = 0
        plt.plot(pri.time_values, pri.data, pen=pen)

        # redraw after every new test
        pg.QtGui.QApplication.processEvents()
    
