import numpy as np
import neuroanalysis.stimuli as stimuli


def test_stimuli():
    s1 = stimuli.Stimulus("stimulus 1")
    assert s1.description == "stimulus 1"
    assert s1.parent is None
    assert len(s1.items) == 0

    sp1 = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=10)
    assert sp1.description == 'square pulse'
    assert sp1.parent is None
    assert len(sp1.items) == 0
    assert sp1.start_time == 0.1
    assert sp1.local_start_time == 0.1
    assert sp1.duration == 0.2
    assert sp1.amplitude == 10

    sp1_eval = sp1.eval(n_pts=1000, dt=0.001)
    assert sp1_eval.dt == 0.001
    sp1_data = sp1_eval.data
    assert sp1_data.shape == (1000,)
    assert np.all(sp1_data[:100] == 0)
    assert np.all(sp1_data[100:300] == 10)
    assert np.all(sp1_data[300:] == 0)

