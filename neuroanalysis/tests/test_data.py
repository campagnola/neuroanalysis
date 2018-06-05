from pytest import raises
import numpy as np

from neuroanalysis.data import Trace


def test_trace_timing():
    # Make sure sample timing is handled exactly--need to avoid fp error here
    a = np.random.normal(size=300)
    sr = 50000
    dt = 2e-5
    t = np.arange(len(a)) * dt
    
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
    with raises(TypeError):
        tr.time_at(0)
    with raises(TypeError):
        tr.index_at(0.1)
    with raises(TypeError):
        tr.value_at(0.1)
        
    view = tr[100:200]
    assert not tr.has_timing
    assert not tr.has_time_values

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
    check_timing(tr, data=a, time_values=t, has_timing=True, has_time_values=False, regularly_sampled=True)

    # test view
    view = tr[100:200]
    assert view.t0 == tr.time_values[100]
    assert view.time_values[0] == view.t0
    assert view.time_values.base is tr.time_values
    assert view.dt == tr.dt
    assert view._meta['sample_rate'] is None
    check_timing(view, data=a[100:200], time_values=t[100:200], has_timing=True, has_time_values=False, regularly_sampled=True)

    view = tr.time_slice(100*dt, 200*dt)
    assert view.t0 == tr.time_values[100]
    assert view.time_values[0] == view.t0
    assert view.dt == tr.dt
    assert view._meta['sample_rate'] is None
    check_timing(view, data=a[100:200], time_values=t[100:200], has_timing=True, has_time_values=False, regularly_sampled=True)
    
    # test nested view
    view2 = view.time_slice(view.t0 + 20*dt, view.t0 + 50*dt)
    assert view2.t0 == view.time_values[20] == tr.time_values[120]
    check_timing(view2, data=a[120:150], time_values=t[120:150], has_timing=True, has_time_values=False, regularly_sampled=True)
    
    # trace with only sample_rate
    tr = Trace(a, sample_rate=sr)
    assert tr.dt == dt
    assert tr.sample_rate == sr
    assert np.all(tr.time_values == t)
    check_timing(tr, data=a, time_values=t, has_timing=True, has_time_values=False, regularly_sampled=True)

    # test view
    view = tr[100:200]
    assert view.t0 == tr.time_values[100]
    assert view.time_values[0] == view.t0
    assert view.sample_rate == tr.sample_rate
    assert view._meta['dt'] is None
    check_timing(view, data=a[100:200], time_values=t[100:200], has_timing=True, has_time_values=False, regularly_sampled=True)
    
    
    # trace with only regularly-sampled time_values
    tr = Trace(a, time_values=t)
    assert tr.dt == dt
    assert np.allclose(tr.sample_rate, sr)
    assert np.all(tr.time_values == t)
    check_timing(tr, data=a, time_values=t, has_timing=True, has_time_values=True, regularly_sampled=True)
    
    # test view
    view = tr[100:200]
    assert view.t0 == tr.time_values[100]
    assert view.time_values[0] == view.t0
    assert view._meta['dt'] is None
    assert view._meta['sample_rate'] is None
    check_timing(view, data=a[100:200], time_values=t[100:200], has_timing=True, has_time_values=True, regularly_sampled=True)


    # trace with irregularly-sampled time values
    t1 = np.cumsum(np.random.normal(loc=1, scale=0.02, size=a.shape))
    tr = Trace(a, time_values=t1)
    assert tr.dt == t1[1] - t1[0]
    assert np.all(tr.time_values == t1)
    check_timing(tr, data=a, time_values=t1, has_timing=True, has_time_values=True, regularly_sampled=False)

    # test view
    view = tr[100:200]
    assert view.t0 == tr.time_values[100]
    assert view.time_values[0] == view.t0
    assert view._meta['dt'] is None
    assert view._meta['sample_rate'] is None
    check_timing(view, data=a[100:200], time_values=t1[100:200], has_timing=True, has_time_values=True, regularly_sampled=False)


def check_timing(tr, data, time_values, has_timing, has_time_values, regularly_sampled):
    assert np.all(tr.time_values == time_values)
    assert tr.has_timing is has_timing
    assert tr.has_time_values is has_time_values
    assert regularly_sampled is regularly_sampled

    # test scalar argument for index_at, time_at, and value_at
    assert tr.value_at(time_values[5]) == tr.data[5]
    assert tr.time_at(5) == time_values[5]
    for t in [time_values[5], time_values[5]*1.0001, time_values[5]/1.0001]:
        index = tr.index_at(t)
        assert isinstance(index, int)
        assert index == 5

    # test array argument for index_at, time_at, and value_at
    t = [time_values[5], time_values[10]]
    indices = tr.index_at(t)
    assert np.all(indices == [5, 10])
    assert indices.dtype == int
    assert np.all(tr.value_at(t) == tr.data[indices])
    assert np.all(tr.time_at([5, 10]) == t)
    indices = np.arange(len(tr))
    assert np.all(tr.value_at(tr.time_at(indices)) == tr.data)
    assert np.all(tr.index_at(tr.time_at(indices)) == indices)
