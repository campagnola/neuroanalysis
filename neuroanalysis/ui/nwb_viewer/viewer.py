import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from neuroanalysis.miesnwb import MiesNwb, SweepGroup

from .sweep_view import SweepView
from .multipatch_view import MultipatchMatrixView
from .analyzer_view import AnalyzerView
from .pair_view import PairView


class MiesNwbExplorer(QtGui.QSplitter):
    """Widget for listing and selecting recordings in a MIES-generated NWB file.
    """
    selection_changed = QtCore.Signal(object)
    channels_changed = QtCore.Signal(object)

    def __init__(self, nwb=None):
        QtGui.QSplitter.__init__(self)
        self.setOrientation(QtCore.Qt.Vertical)

        self._nwb = None
        self._channel_selection = {}

        # self.menu = QtGui.QMenuBar()
        # self.layout.addWidget(self.menu, 0, 0)

        # self.show_menu = QtGui.QMenu("Show")
        # self.menu.addMenu(self.show_menu)
        # self.groupby_menu = QtGui.QMenu("Group by")
        # self.menu.addMenu(self.groupby_menu)

        self.sweep_tree = QtGui.QTreeWidget()
        self.sweep_tree.setColumnCount(3)
        self.sweep_tree.setHeaderLabels(['Stim Name', 'Clamp Mode', 'Holding'])
        self.sweep_tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.addWidget(self.sweep_tree)
        
        self.channel_list = QtGui.QListWidget()
        self.addWidget(self.channel_list)
        self.channel_list.itemChanged.connect(self._channel_list_changed)

        self.meta_tree = pg.DataTreeWidget()
        self.addWidget(self.meta_tree)

        self.set_nwb(nwb)

        self.sweep_tree.itemSelectionChanged.connect(self._selection_changed)

    def set_nwb(self, nwb):
        self._nwb = nwb
        self._channel_selection = {}
        self.update_sweep_tree()

    def update_sweep_tree(self):
        self.sweep_tree.clear()

        if self._nwb is None:
            return

        for group in self._nwb.sweep_groups():
            meta = [group.sweeps[0].traces()[ch].meta() for ch in group.sweeps[0].channels()]

            mode = []
            holding = []
            for m in meta:
                mode.append('V' if m.get('Clamp Mode', '') == 0 else 'I')
                holding.append(m.get('%s-Clamp Holding Level'%mode[-1], ''))
            holding = ' '.join(['%0.1f'%h if h is not None else '__._' for h in holding])
            mode = ' '.join(mode)

            gitem = QtGui.QTreeWidgetItem([meta[0]['stim_name'], str(mode), str(holding)])
            gitem.data = group
            self.sweep_tree.addTopLevelItem(gitem)
            for sweep in group.sweeps:
                item = QtGui.QTreeWidgetItem([str(sweep.sweep_id)])
                item.data = sweep
                gitem.addChild(item)

        self.sweep_tree.header().resizeSections(QtGui.QHeaderView.ResizeToContents)

    def selection(self):
        """Return a list of selected groups and/or sweeps. 
        """
        items = self.sweep_tree.selectedItems()
        selection = []
        for item in items:
            if item.parent() in items:
                continue
            selection.append(item.data)
        return selection

    def selected_channels(self):
        chans = []
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                chans.append(item.channel_index)
        return chans

    def _update_channel_list(self):
        self.channel_list.itemChanged.disconnect(self._channel_list_changed)
        try:
            # clear channel list
            while self.channel_list.count() > 0:
                self.channel_list.takeItem(0)
            
            # bail out if nothing is selected
            sel = self.selection()
            if len(sel) == 0:
                return
            
            # get a list of all channels across all selected items
            channels = []
            for item in sel:
                if isinstance(item, SweepGroup):
                    if len(item.sweeps) == 0:
                        continue
                    item = item.sweeps[0]
                channels.extend([(ch, item.traces()[ch].headstage_id) for ch in item.channels()])
            channels = list(set(channels))
            channels.sort()
            
            # add new items to the channel list, all selected
            for ch,chid in channels:
                item = QtGui.QListWidgetItem(str(chid))
                item.channel_index = ch
                self._channel_selection.setdefault(ch, True)
                # restore previous check state, if any.
                checkstate = QtCore.Qt.Checked if self._channel_selection.setdefault(ch, True) else QtCore.Qt.Unchecked
                item.setCheckState(checkstate)
                self.channel_list.addItem(item)
        finally:
            self.channel_list.itemChanged.connect(self._channel_list_changed)

    def _selection_changed(self):
        sel = self.selection()
        if len(sel) == 1:
            if isinstance(sel[0], SweepGroup):
                self.meta_tree.setData(sel[0].meta())
            else:
                self.meta_tree.setData(sel[0].meta(all_chans=True))
        else:
            self.meta_tree.clear()
        self._update_channel_list()
        self.selection_changed.emit(sel)
        
    def _channel_list_changed(self, item):
        self.channels_changed.emit(self.selected_channels())
        self._channel_selection[item.channel_index] = item.checkState() == QtCore.Qt.Checked


class MiesNwbViewer(QtGui.QWidget):
    """Combination of a MiesNwvExplorer for selecting sweeps and a tab widget
    containing multiple views, each performing a different analysis.
    """
    analyzer_changed = QtCore.Signal(object)
    
    def __init__(self, nwb=None):
        QtGui.QWidget.__init__(self)
        self.nwb = nwb
        
        self.layout = QtGui.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.hsplit = QtGui.QSplitter()
        self.hsplit.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.hsplit, 0, 0)
        
        self.vsplit = QtGui.QSplitter()
        self.vsplit.setOrientation(QtCore.Qt.Vertical)
        self.hsplit.addWidget(self.vsplit)

        self.explorer = MiesNwbExplorer(self.nwb)
        self.explorer.selection_changed.connect(self.data_selection_changed)
        self.explorer.channels_changed.connect(self.data_selection_changed)
        self.vsplit.addWidget(self.explorer)

        self.ptree = pg.parametertree.ParameterTree()
        self.vsplit.addWidget(self.ptree)
        
        self.tabs = QtGui.QTabWidget()
        self.hsplit.addWidget(self.tabs)
        
        self.reload_btn = QtGui.QPushButton("Reload views")
        self.reload_btn.clicked.connect(self.reload_views)
        self.vsplit.addWidget(self.reload_btn)

        self.views = []
        self.create_views()
        
        self.resize(1000, 800)
        self.hsplit.setSizes([150, 850])

        self.tab_changed()
        self.tabs.currentChanged.connect(self.tab_changed)

    def set_nwb(self, nwb):
        self.nwb = nwb
        self.explorer.set_nwb(nwb)

    def data_selection_changed(self):
        sweeps = self.selected_sweeps()
        chans = self.selected_channels()
        with pg.BusyCursor():
            self.tabs.currentWidget().data_selected(sweeps, chans)

    def tab_changed(self):
        w = self.tabs.currentWidget()
        if w is None:
            self.ptree.clear()
            return
        self.ptree.setParameters(w.params, showTop=False)
        sweeps = self.selected_sweeps()
        chans = self.selected_channels()
        w.data_selected(sweeps, chans)
        self.analyzer_changed.emit(self)
        
    def selected_analyzer(self):
        return self.tabs.currentWidget()

    def selected_sweeps(self, selection=None):
        if selection is None:
            selection = self.explorer.selection()
        sweeps = []
        for item in selection:
            if isinstance(item, SweepGroup):
                sweeps.extend(item.sweeps)
            else:
                sweeps.append(item)
        return sweeps
    
    def selected_channels(self):
        return self.explorer.selected_channels()

    def reload_views(self):
        """Remove all existing views, reload their source code, and create new
        views.
        """
        self.clear_views()
        pg.reload.reloadAll(debug=True)
        self.create_views()

    def clear_views(self):
        self.tabs.clear()
        self.views = []
        
    def create_views(self):
        self.clear_views()
        self.views = [
            ('Sweep', SweepView(self)),
            ('Matrix', MultipatchMatrixView(self)),
            ('Sandbox', AnalyzerView(self)),
            ('Pair', PairView(self)),
        ]

        for name, view in self.views:
            self.tabs.addTab(view, name)



class AnalysisView(QtGui.QWidget):
    """Example skeleton for an analysis view.
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        # Views must have self.params
        # This implements the controls that are unique to this view.
        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'lowpass', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 1},
            {'name': 'average', 'type': 'bool', 'value': False},
        ])
        self.params.sigTreeStateChanged.connect(self._update_analysis)

    def data_selected(self, sweeps, channels):
        """Called when the user selects a different set of sweeps.
        """
        self.sweeps = sweeps
        self.channels = channels
        self.update_analysis()

    def update_analysis(self, param, changes):
        """Called when the user changes control parameters.
        """
        pass


        
if __name__ == '__main__':
    import sys
    from pprint import pprint
    pg.dbg()
    
    filename = sys.argv[1]
    nwb = MiesNwb(filename)
    # sweeps = nwb.sweeps()
    # traces = sweeps[0].traces()
    # # pprint(traces[0].meta())
    # groups = nwb.sweep_groups()
    # for i,g in enumerate(groups):
    #     print "--------", i, g
    #     print g.describe()

    # d = groups[7].data()
    # print d.shape

    app = pg.mkQApp()
    w = MiesNwbViewer(nwb)
    w.show()
    # w.show_group(7)
