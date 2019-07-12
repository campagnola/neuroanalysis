"""Script used to generate evoked spike test data
"""
import pickle
import numpy as np
from scipy.optimize import curve_fit
from neuroanalysis.spike_detection import detect_evoked_spikes, SpikeDetectUI
from neuroanalysis.data import Trace, TraceList, PatchClampRecording
from multipatch_analysis.database import default_db as db
from multipatch_analysis.data import Analyzer, PulseStimAnalyzer, MultiPatchProbe

import pyqtgraph as pg
pg.dbg()  # for inspecting exception stack


# cells with currently poorly identified spikes 
cell_ids = [
    [1540356446.981, 8, 6, [79]],
    [1497417667.378, 5, 2, [21]],  # no spikes at all
    [1544582617.589, 1, 6, [89, 40, 41, 84]], 
    [1544582617.589, 1, 8, [59]],  #this is a good text bc two fail but the others are sort of sad looking.
    [1491942526.646, 8, 1, [14, 27]],  #this one is giving me issues #this presynaptic cell is sick.  Spiking is ambiguious, very interesting examples
    [1521004040.059, 5, 6, [23, 24]],
    [1534293227.896, 7, 8, [16, 17, 21]], #this one is not quite perfected from the recurve up at the end fix with derivat
##    [1550101654.271, 1, 6, []], # these spike toward the end and are found correctly
    [1516233523.013, 6, 7, [16]],  #very interesting example: a voltage deflection happens very early but cant be seen in dvvdt due to to onset being to early.  Think about if there is a way to fix this.  Maybe and initial pulse window.  
##    [1534297702.068, 7, 2, []],
    [1558647151.979, 4, 3, [20]],
]

ui = SpikeDetectUI()
next_btn = pg.QtGui.QPushButton('next')
ui.widget.addWidget(next_btn)


session = db.session()
def iter_pulses():
    """Generator that yields all selected pulses one at a time.
    """
    global cell_ids, session
    for cell_id in cell_ids:
        expt_id, pre_cell_id, post_cell_id, sweep_ids = cell_id
        print("expt: %0.3f  pair: %d %d" % (expt_id, pre_cell_id, post_cell_id))
        
        # look up experiment from database and load the NWB file
        expt = db.experiment_from_timestamp(expt_id)
        pair = expt.pairs[pre_cell_id, post_cell_id]
        channel = pair.pre_cell.electrode.device_id
        # synapse = pair.synapse
        # synapse_type = pair.connection_strength.synapse_type
        # pulse_responses = pair.pulse_responses
        
        # which sweeps to load from this experiment? (all sweeps if none have been specified)
        if len(sweep_ids) == 0:
            sweeps = expt.data.contents
        else:
            sweeps = [expt.data.contents[swid] for swid in sweep_ids]
        sweeps = expt.data.contents
    
        for sweep in sweeps:
            # Ignore sweep if it doesn't have the requested channel, or the correct stimulus
            try:
                pre_rec = sweep[channel]
            except KeyError:
                continue
            if not isinstance(pre_rec, MultiPatchProbe):
                continue
            print("sweep: %d  channel: %d" % (sweep.key, channel))

            # Get chunks for each stim pulse        
            pulse_stim = PulseStimAnalyzer.get(pre_rec)
            chunks = pulse_stim.pulse_chunks()
            for chunk in chunks:
                yield (expt_id, pre_cell_id, post_cell_id, sweep, channel, chunk)


all_pulses = iter_pulses()
fileno = 1

def load_next():
    global all_pulses, ui, fileno
    (expt_id, pre_cell_id, post_cell_id, sweep, channel, chunk) = all_pulses.next()

    # run spike detection on each chunk
    pulse_edges = chunk.meta['pulse_edges']
    spikes = detect_evoked_spikes(chunk, pulse_edges, ui=ui)
    
    # copy just the necessary parts of recording data for export to file
    export_chunk = PatchClampRecording(channels={k:Trace(chunk[k].data, t0=chunk[k].t0, sample_rate=chunk[k].sample_rate) for k in chunk.channels})
    export_chunk.meta.update(chunk.meta)
    
    # write results out to test file
    info = {
        'expt_id': expt_id,
        'pre_cell_id': pre_cell_id,
        'post_cell_id': post_cell_id,
        'sweep_id': sweep.key,
        'data': export_chunk,
        'pulse_edges': chunk.meta['pulse_edges'],
        'spikes': spikes,
    }
    test_file = '../neuroanalysis/test_data/evoked_spikes/%s_spike_%04d.pkl' % (chunk.clamp_mode, fileno)
    print("write:", test_file)
    pickle.dump(info, open(test_file, 'w'))
    fileno += 1


next_btn.clicked.connect(load_next)
