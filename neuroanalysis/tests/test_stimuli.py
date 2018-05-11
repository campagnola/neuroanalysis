from collections import OrderedDict
import numpy as np
import neuroanalysis.stimuli as stimuli


def test_stimulus():
    # test a generic stimulus
    s1 = stimuli.Stimulus("stimulus 1")
    assert s1.description == "stimulus 1"
    assert s1.global_start_time == 0
    assert s1.start_time == 0
    assert s1.type == "Stimulus"
    assert s1.parent is None
    assert len(s1.items) == 0

    # test save/load
    state = s1.save()
    assert state == OrderedDict([
        ('type', 'Stimulus'),
        ('args', OrderedDict([
            ('start_time', 0), 
            ('description', 'stimulus 1'),
            ('units', None),
        ])),
        ('items', []),
    ])
    s2 = stimuli.load_stimulus(state)
    assert isinstance(s2, stimuli.Stimulus)
    assert s2.description == "stimulus 1"
    assert s2.global_start_time == 0
    assert s2.start_time == 0
    assert s2.type == "Stimulus"
    assert s2.parent is None
    assert len(s2.items) == 0


def test_square_pulse():
    # test SquarePulse
    sp1 = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=10)
    assert sp1.type == "SquarePulse"
    assert sp1.description == 'square pulse'
    assert sp1.parent is None
    assert len(sp1.items) == 0
    assert sp1.global_start_time == 0.1
    assert sp1.start_time == 0.1
    assert sp1.duration == 0.2
    assert sp1.amplitude == 10

    # check evaluation of pulse waveform
    sp1_eval = sp1.eval(n_pts=1000, dt=0.001)
    assert sp1_eval.dt == 0.001
    sp1_data = sp1_eval.data
    assert sp1_data.shape == (1000,)
    test_data = np.zeros(1000)
    test_data[100:300] = 10
    assert np.all(sp1_data == test_data)

    # test parent/child logic
    s1 = stimuli.Stimulus("stimulus 1")
    sp1.parent = s1
    assert sp1.parent is s1
    assert s1.items == (sp1,)

    sp1.parent = None
    assert sp1.parent is None
    assert s1.items == ()
    
    s1.append_item(sp1)
    assert sp1.parent is s1
    assert s1.items == (sp1,)
    
    s1.remove_item(sp1)
    assert sp1.parent is None
    assert s1.items == ()

    # add in a second pulse
    sp2 = stimuli.SquarePulse(start_time=0.2, duration=0.2, amplitude=-10, description="square pulse 2", parent=s1, units='A')
    assert sp2.description == "square pulse 2"
    assert sp2.parent is s1
    assert s1.items == (sp2,)
    
    # more parent / child logic testing
    s1.insert_item(1, sp1)
    assert sp1.parent is s1
    assert s1.items == (sp2, sp1)

    s1.remove_item(sp1)
    s1.insert_item(0, sp1)
    assert sp1.parent is s1
    assert s1.items == (sp1, sp2)

    # test waveform eval
    sp2_eval = sp2.eval(n_pts=1000, dt=0.001)
    assert sp2_eval.units == 'A'
    sp2_test_data = np.zeros(1000)
    sp2_test_data[200:400] = -10
    assert np.all(sp2_test_data == sp2_eval.data)

    # test waveform eval with two pulses
    s1_data = s1.eval(n_pts=1000, dt=0.001).data
    test_data[200:400] -= 10
    assert np.all(s1_data == test_data)

    # test save/load
    state = sp1.save()
    assert state == OrderedDict([
        ('type', 'SquarePulse'),
        ('args', OrderedDict([
            ('start_time', 0.1), 
            ('description', 'square pulse'),
            ('units', None),
            ('duration', 0.2),
            ('amplitude', 10),
        ])),
        ('items', []),
    ])
    sp3 = stimuli.load_stimulus(state)
    assert isinstance(sp3, stimuli.SquarePulse)
    assert sp3.type == sp1.type == "SquarePulse"
    assert sp3.description == sp1.description == 'square pulse'
    assert sp3.global_start_time == sp1.global_start_time == 0.1
    assert sp3.start_time == sp1.start_time == 0.1
    assert sp3.duration == sp1.duration == 0.2
    assert sp3.amplitude == sp1.amplitude == 10
    assert sp3.parent is None
    assert len(sp3.items) == 0


def test_pulse_train():
    s1 = stimuli.Stimulus("stimulus 1")
    sp1 = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=10, parent=s1)
    sp2 = stimuli.SquarePulse(start_time=0.2, duration=0.2, amplitude=-10, description="square pulse 2", units='A', parent=s1)

    # test PulseTrain
    pt1 = stimuli.SquarePulseTrain(start_time=0.5, n_pulses=3, pulse_duration=0.02, interval=0.1, amplitude=1.0, parent=s1, units='A')
    assert s1.items == (sp1, sp2, pt1)

    # check sub-pulse start times
    assert pt1.global_start_time == 0.5
    assert pt1.start_time == 0.5
    assert pt1.items[0].global_start_time == 0.5
    assert pt1.items[0].start_time == 0.0
    assert pt1.items[1].global_start_time == 0.6
    assert pt1.items[1].start_time == 0.1
    assert pt1.items[2].global_start_time == 0.7
    assert pt1.items[2].start_time == 0.2
    assert pt1.pulse_times == [0.0, 0.1, 0.2]
    assert pt1.global_pulse_times == [0.5, 0.6, 0.7]

    # test waveform eval with all three items
    s1_data = s1.eval(n_pts=1000, dt=0.001).data
    test_data = np.zeros(1000)
    test_data[100:300] = 10
    test_data[200:400] -= 10
    test_data[500:520] += 1
    test_data[600:620] += 1
    test_data[700:720] += 1
    assert np.all(s1_data == test_data)

    # test save/load
    state = s1.save()
    assert state == OrderedDict([
        ('type', 'Stimulus'),
        ('args', OrderedDict([
            ('start_time', 0),
            ('description', 'stimulus 1'),
            ('units', None),
        ])),
        ('items', [
            OrderedDict([
                ('type', 'SquarePulse'),
                ('args', OrderedDict([
                    ('start_time', 0.1), 
                    ('description', 'square pulse'),
                    ('units', None),
                    ('duration', 0.2),
                    ('amplitude', 10),
                ])),
                ('items', []),
            ]),
            OrderedDict([
                ('type', 'SquarePulse'),
                ('args', OrderedDict([
                    ('start_time', 0.2), 
                    ('description', 'square pulse 2'),
                    ('units', 'A'),
                    ('duration', 0.2),
                    ('amplitude', -10),
                ])),
                ('items', []),
            ]),
            OrderedDict([
                ('type', 'SquarePulseTrain'),
                ('args', OrderedDict([
                    ('start_time', 0.5), 
                    ('description', 'square pulse train'),
                    ('units', 'A'),
                    ('n_pulses', 3),
                    ('pulse_duration', 0.02),
                    ('amplitude', 1.0),
                    ('interval', 0.1),
                ])),
                ('items', []),
            ]),
        ])
    ])
    s2 = stimuli.load_stimulus(state)
    assert isinstance(s2, stimuli.Stimulus)
    assert s2.items[0] == s1.items[0]
    assert s2.items[1] == s1.items[1]
    assert s2.items[2] == s1.items[2]
    assert s2.items[2].items[0] == s1.items[2].items[0]
    assert s2.items[2].items[1] == s1.items[2].items[1]
    assert s2.items[2].items[2] == s1.items[2].items[2]
    