"""Script used to generate ps fitting test data

Usage:  python -i import_spike_detection.py expt_id cell_id

This will load all PSP/PSCs evoked in the specified cell pair one at a time. 
For each one you can select whether to write the data out to a new test file.
Note that files are saved without results; to generate these, you must run
unit tests with --audit.
"""

import pickle, sys
import numpy as np
from scipy.optimize import curve_fit
from neuroanalysis.fitting.psp import fit_psp, PspFitTestCase
from neuroanalysis.ui.psp_fitting import PspFitUI
from neuroanalysis.data import Trace, TraceList, PatchClampRecording
from multipatch_analysis.database import default_db as db
from multipatch_analysis.data import Analyzer, MultiPatchSyncRecAnalyzer, MultiPatchProbe

import pyqtgraph as pg
pg.dbg()  # for inspecting exception stack


expt_id = float(sys.argv[1])
pre_cell_id = int(sys.argv[2])
post_cell_id = int(sys.argv[3])


ui = PspFitUI()
skip_btn = pg.QtGui.QPushButton('skip')
ui.widget.addWidget(skip_btn)
save_btn = pg.QtGui.QPushButton('save')
ui.widget.addWidget(save_btn)


session = db.session()
def iter_pulses():
    """Generator that yields all selected pulses one at a time.
    """
    # look up experiment from database and load the NWB file
    expt = db.experiment_from_timestamp(expt_id)
    pair = expt.pairs[pre_cell_id, post_cell_id]
    pre_cell = expt.cells[pre_cell_id]
    post_cell = expt.cells[post_cell_id]
    pre_channel = pre_cell.electrode.device_id
    post_channel = post_cell.electrode.device_id
    sweeps = expt.data.contents

    for sweep in sweeps:
        # Ignore sweep if it doesn't have the requested channels, or the correct stimulus
        try:
            pre_rec = sweep[pre_channel]
            post_rec = sweep[post_channel]
        except KeyError:
            continue
        if not isinstance(pre_rec, MultiPatchProbe):
            continue

        print("sweep: %d" % sweep.key)

        # Get chunks for each stim pulse        
        analyzer = MultiPatchSyncRecAnalyzer.get(sweep)
        pulse_responses = analyzer.get_spike_responses(pre_rec, post_rec)
        for pr in pulse_responses:
            yield (expt_id, pre_cell_id, post_cell_id, sweep, pr)


all_pulses = iter_pulses()
last_result = None

def load_next():
    global all_pulses, ui, last_result
    try:
        (expt_id, pre_cell_id, post_cell_id, sweep, pr) = next(all_pulses)
    except StopIteration:
        ui.widget.hide()
        return

    # run psp fit on each chunk
    pulse_edges = chunk.meta['pulse_edges']
    spikes = detect_evoked_spikes(chunk, pulse_edges, ui=ui)
    ui.show_result(spikes)

    # copy just the necessary parts of recording data for export to file
    export_chunk = PatchClampRecording(channels={k:Trace(chunk[k].data, t0=chunk[k].t0, sample_rate=chunk[k].sample_rate) for k in chunk.channels})
    export_chunk.meta.update(chunk.meta)

    # construct test case    
    tc = SpikeDetectTestCase()
    tc._meta = {
        'expt_id': expt_id,
        'cell_id': cell_id,
        'device_id': channel,
        'sweep_id': sweep.key,
    }
    tc._input_args = {
        'data': export_chunk,
        'pulse_edges': chunk.meta['pulse_edges'],
    }
    last_result = tc


def save_and_load_next():
    global last_result

    # write results out to test file
    test_file = 'test_data/evoked_spikes/%s.pkl' % (last_result.name)
    last_result.save_file(test_file)

    load_next()


skip_btn.clicked.connect(load_next)
save_btn.clicked.connect(save_and_load_next)
load_next()
