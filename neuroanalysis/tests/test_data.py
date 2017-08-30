from pytest import raises
import numpy as np

from neuroanalysis.data import Trace


def test_trace_timing():
    # Make sure sample timing is handled exactly--need to avoid fp error here
    a = np.random.normal(size=100)
    sr = 50000
    dt = 2e-5
    t = np.arange(100) * dt
    
    # trace with no timing information 
    tr = Trace(a)
    assert not tr.has_timing
    assert not tr.has_time_values
    with raises(TypeError):
        tr.dt
    with raises(TypeError):
        tr.sample_rate
    with raises(TypeError):
        tr.time_values

    # invalid data
    with raises(ValueError):
        Trace(data=np.zeros((10, 10)))

    # invalid timing information
    with raises(TypeError):
        Trace(data=a, dt=dt, time_values=t)
    with raises(TypeError):
        Trace(data=a, sample_rate=sr, time_values=t)
    with raises(TypeError):
        Trace(data=a, dt=dt, t0=0, time_values=t)
    with raises(TypeError):
        Trace(data=a, dt=dt, t0=0, sample_rate=sr)
    with raises(ValueError):
        Trace(data=a, time_values=t[:-1])

    # trace with only dt
    tr = Trace(a, dt=dt)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    assert np.all(tr.time_values == t)
    assert tr.has_timing
    assert not tr.has_time_values
    assert tr.regularly_sampled
    
    # trace with only sample_rate
    tr = Trace(a, sample_rate=sr)
    assert tr.dt == dt
    assert tr.sample_rate == sr
    assert np.all(tr.time_values == t)
    assert tr.has_timing
    assert not tr.has_time_values
    assert tr.regularly_sampled

    # trace with only regularly-sampled time_values
    tr = Trace(a, time_values=t)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    assert np.all(tr.time_values == t)
    assert tr.has_timing
    assert tr.has_time_values
    assert tr.regularly_sampled
    
    # trace with irregularly-sampled time values
    t1 = np.cumsum(np.random.normal(loc=1, scale=0.02, size=a.shape))
    tr = Trace(a, time_values=t1)
    assert tr.dt == t1[1] - t1[0]
    assert np.all(tr.time_values == t1)
    assert tr.has_timing
    assert tr.has_time_values
    assert not tr.regularly_sampled
    
    
    