import numpy as np

from neuroanalysis.data import Trace


def test_sample_rate():
    # Make sure sample timing is handled exactly--need to avoid fp error here
    a = np.random.normal(size=100)
    sr = 50000
    dt = 2e-5
    t = np.arange(100) * dt
    
    tr = Trace(a, dt=dt)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    assert np.all(tr.time_values == t)
    
    tr = Trace(a, sample_rate=sr)
    assert tr.dt == dt
    assert tr.sample_rate == sr
    assert np.all(tr.time_values == t)

    tr = Trace(a, time_values=t)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    assert np.all(tr.time_values == t)
    
    