import os
import numpy as np
import pyqtgraph as pg
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
from tools import STAAnalyzer
import user


drive_path = '/media/luke/Brain2016/'
manifest_path = os.path.join(drive_path,'BrainObservatory','manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_path)

cont_id = 511510670
cont_info = boc.get_ophys_experiments(experiment_container_ids=[cont_id])

# pick a data set with locally sparse noise stimulation
expt_id = cont_info[0]['id']

# pick a starting cell
cells = data_set.get_cell_specimen_ids()
cell_id = cells[0]

pg.mkQApp()

sd = STAAnalyzer(boc, expt_id, cell_id)
