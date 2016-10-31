import sys
from collections import OrderedDict
import numpy as np
import h5py


class MiesNwb(object):
    """Class for accessing data from a MIES-generated NWB file.
    """
    def __init__(self, filename):
        self.filename = filename
        self.hdf = h5py.File(filename, 'r')
        self._sweeps = None
        self._groups = None
        self._notebook = None
        
    def notebook(self):
        """Return compiled data from the lab notebook.

        The format is a dict like {sweep_number: metadata} with one key-value pair per sweep.
        The metadata for each sweep is returned as an array 
        """
        if self._notebook is None:
            # collect all lab notebook entries
            nb_entries = {}
            nb_keys = self.hdf['general']['labnotebook']['ITC1600_Dev_0']['numericalKeys'][0]
            nb = self.hdf['general']['labnotebook']['ITC1600_Dev_0']['numericalValues']
            for i in range(nb.shape[0]):
                rec = nb[i]
                sweep_num = rec[0,0]
                if np.isnan(sweep_num):
                    continue
                sweep_num = int(sweep_num)
                # each sweep gets multiple nb records; for each field we use the last non-nan value in any record
                if sweep_num not in nb_entries:
                    nb_entries[sweep_num]= np.array(rec)
                else:
                    mask = ~np.isnan(rec)
                    nb_entries[sweep_num][mask] = rec[mask]

            for swid, entry in nb_entries.items():
                # last column applies to all channels
                mask = ~np.isnan(entry[:,8:9])
                entry[mask] = entry[:,8:9][mask]

                # convert to list-o-dicts
                meta = []
                for i in range(entry.shape[1]):
                    tm = entry[:, i]
                    meta.append({nb_keys[j]:(None if np.isnan(tm[j]) else tm[j]) for j in range(len(nb_keys))})
                nb_entries[swid] = meta

            self._notebook = nb_entries
        return self._notebook

    def sweeps(self):
        """Return a list of all sweeps in this file.
        """
        if self._sweeps is None:
            sweeps = set()
            for k in self.hdf['acquisition/timeseries'].keys():
                a, b, c = k.split('_')
                sweeps.add(b)
            self._sweeps = [Sweep(self, int(sweep_id)) for sweep_id in sorted(list(sweeps))]
        return self._sweeps
    
    def sweep_groups(self, keys=('stim_name', 'V-Clamp Holding Level')):
        """Return a list of sweep groups--each group contains one or more
        contiguous sweeps with matching metadata.

        The *keys* argument contains the set of metadata keys that are compared
        to determine group boundaries.

        This is used mainly for grouping together sweeps that were repeated or were 
        part of a stim set.
        """
        if self._groups is None:
            current_group = []
            current_meta = None
            groups = [current_group]
            for sweep in self.sweeps():
                # get selected metadata for grouping sweeps
                meta = {}
                for ch,m in sweep.meta().items():
                    meta[ch] = {k:m[k] for k in keys}

                if len(current_group) == 0:
                    current_group.append(sweep)
                    current_meta = meta
                else:
                    if meta == current_meta:
                        current_group.append(sweep)
                    else:
                        current_group = [sweep]
                        current_meta = meta
                        groups.append(current_group)
            self._groups = [SweepGroup(self, grp) for grp in groups]
        return self._groups


class Trace(object):
    """A single stimulus / recording made on a single channel.
    """
    def __init__(self, sweep, sweep_id, ad_chan):
        self.sweep = sweep
        self.nwb = sweep.nwb
        self.trace_id = (sweep_id, ad_chan)
        self.hdf_group = self.nwb.hdf['acquisition/timeseries/data_%05d_AD%d' % self.trace_id]
        self.headstage_id = int(self.hdf_group['electrode_name'].value[0].split('_')[1])
        self._meta = None
        self._da_chan = None

    def data(self):
        """Return an array of shape (N, 2) containing the recording and stimulus
        for this trace.

        The first column [:, 0] contains recorded data and the second column [:, 1]
        contains the stimulus.
        """
        return np.vstack([self.recording(), self.stim()]).T

    def recording(self):
        """Return the recorded data for this trace.
        """
        return self.hdf_group['data']        
        
    def da_chan(self):
        """Return the DA channel ID for this trace.
        """
        if self._da_chan is None:
            hdf = self.nwb.hdf['stimulus/presentation']
            stims = [k for k in hdf.keys() if k.startswith('data_%05d_'%self.trace_id[0])]
            for s in stims:
                elec = hdf[s]['electrode_name'].value[0]
                if elec == 'electrode_%d' % self.headstage_id:
                    self._da_chan = int(s.split('_')[-1][2:])
            if self._da_chan is None:
                raise Exception("Cannot find DA channel for headstage %d" % self.headstage_id)
        return self._da_chan

    def stim(self):
        """Return the stimulus array for this trace.
        """
        return self.nwb.hdf['stimulus/presentation/data_%05d_DA%d/data' % (self.trace_id[0], self.da_chan())]

    def meta(self):
        """Return a dict of metadata for this trace.

        Keys include 'stim_name', 'start_time', and all parameters recorded in the lab notebook.
        """
        if self._meta is None:
            self._meta = {
                'stim_name': self.hdf_group['stimulus_description'].value[0], 
                'start_time': self.hdf_group['starting_time'].value[0], 
            }
            nb = self.nwb.notebook()[int(self.trace_id[0])][self.headstage_id]
            self._meta.update(nb)
        return self._meta

    def __repr__(self):
        meta = self.meta()
        mode = meta['Clamp Mode']
        if mode == 0:  # VC
            extra = "mode=VC holding=%d" % int(np.round(meta['V-Clamp Holding Level']))
        elif mode == 1:  # IC
            extra = "mode=IC holding=%d" % int(np.round(meta['I-Clamp Holding Level']))

        return "<Trace %d.%d  stim=%s %s>" % (self.trace_id[0], self.headstage_id, meta['stim_name'], extra)


class Sweep(object):
    """Represents one recorded sweep with multiple channels.
    """
    def __init__(self, nwb, sweep_id):
        self.nwb = nwb
        self.sweep_id = sweep_id
        self._channels = None
        self._meta = None
        self._traces = None
        self._notebook_entry = None

    def channels(self):
        """Return a list of AD channels participating in this sweep.
        """
        if self._channels is None:
            chans = []
            for k in self.nwb.hdf['acquisition/timeseries'].keys():
                if not k.startswith('data_%05d_' % self.sweep_id):
                    continue
                chans.append(int(k.split('_')[-1][2:]))
            self._channels = sorted(chans)
        return self._channels

    def traces(self):
        """Return a dict of Traces in this sweep, one per channel.
        """
        if self._traces is None:
            self._traces = OrderedDict([(ch, Trace(self, self.sweep_id, ch)) for ch in self.channels()])
        return self._traces

    def meta(self):
        """Return nested dicts containing meta-information about each
        channel in this sweep.
        """
        if self._meta is None:
            traces = self.traces()
            self._meta = {chan: traces[chan].meta() for chan in self.channels()}
        return self._meta
        
    def data(self):
        """Return a single array containing recorded data and stimuli from all channels recorded
        during this sweep.
        
        The array shape is (channels, samples, 2).
        """
        traces = self.traces()
        chan_data = [traces[ch].data() for ch in sorted(list(traces))]
        arr = np.empty((len(chan_data),) + chan_data[0].shape, chan_data[0].dtype)
        for i,data in enumerate(chan_data):
            arr[i] = data
        return arr

    def describe(self):
        """Return a string description of this sweep.
        """
        return "\n".join(map(repr, self.traces().values()))


class SweepGroup(object):
    """Represents a collection of Sweeps that were acquired contiguously and
    all share the same stimulus parameters.
    """
    def __init__(self, nwb, sweeps):
        self.nwb = nwb
        self.sweeps = sweeps
        
    def meta(self):
        """Return metadata from the first sweep in this group.
        """
        return self.sweeps[0].meta()
        
    def data(self):
        """Return a single array containing all data from all sweeps in this
        group.
        
        The array shape is (sweeps, channels, samples, 2).
        """
        sweeps = [s.data() for s in self.sweeps]
        data = np.empty((len(sweeps),) + sweeps[0].shape, dtype=sweeps[0].dtype)
        for i in range(len(sweeps)):
            data[i] = sweeps[i]
        return data

    def describe(self):
        """Return a string description of this group (taken from the first sweep).
        """
        return self.sweeps[0].describe()

    def __repr__(self):
        ids = self.sweeps[0].sweep_id, self.sweeps[-1].sweep_id
        return "<SweepGroup %d-%d>" % ids
