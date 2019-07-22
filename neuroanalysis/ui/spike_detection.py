from __future__ import division, print_function

import time
import pyqtgraph as pg
import pyqtgraph.console


class SpikeDetectUI(object):
    """Used to display details of spike detection analysis.
    """
    def __init__(self, title=None):
        self.pw = pg.GraphicsLayoutWidget()
        self.plt1 = self.pw.addPlot(title=title)
        self.plt2 = self.pw.addPlot(row=1, col=0)
        self.plt2.setXLink(self.plt1)
        self.plt3 = self.pw.addPlot(row=2, col=0)
        self.plt3.setXLink(self.plt1)
        
        self.console = pg.console.ConsoleWidget()
        
        self.widget = pg.QtGui.QSplitter(pg.QtCore.Qt.Vertical)
        self.widget.addWidget(self.pw)        
        self.widget.addWidget(self.console)
        self.widget.resize(1000, 900)
        self.widget.setSizes([800, 200])
        self.widget.show()

    
    def clear(self):
        self.plt1.clear()
        self.plt2.clear()
        self.plt3.clear()

    def show_result(self, spikes):
        for plt in [self.plt1, self.plt2, self.plt3]:
            for spike in spikes:
                if spike.get('onset_time') is not None:
                    plt.addLine(x=spike['onset_time'])
                if spike.get('max_slope_time') is not None:
                    plt.addLine(x=spike['max_slope_time'], pen='b')
                if spike.get('peak_time') is not None:
                    plt.addLine(x=spike['peak_time'], pen='g')


class SpikeDetectTestUi(object):
    def __init__(self):
        pg.mkQApp()

        self.widget = pg.QtGui.QSplitter(pg.QtCore.Qt.Horizontal)
        self.widget.resize(1600, 1000)

        self.display1 = SpikeDetectUI('expected result')
        self.display2 = SpikeDetectUI('current result')
        self.widget.addWidget(self.display1.widget)
        self.widget.addWidget(self.display2.widget)

        self.ctrl = pg.QtGui.QWidget()
        self.widget.addWidget(self.ctrl)
        self.ctrl_layout = pg.QtGui.QVBoxLayout()
        self.ctrl.setLayout(self.ctrl_layout)
        self.diff_widget = pg.DiffTreeWidget()
        self.ctrl_layout.addWidget(self.diff_widget)

        self.pass_btn = pg.QtGui.QPushButton('pass')
        self.fail_btn = pg.QtGui.QPushButton('fail')
        self.ctrl_layout.addWidget(self.pass_btn)
        self.ctrl_layout.addWidget(self.fail_btn)

        self.pass_btn.clicked.connect(self.pass_clicked)
        self.fail_btn.clicked.connect(self.fail_clicked)

        self.last_btn_clicked = None
        self.widget.setSizes([400, 400, 800])

    def pass_clicked(self):
        self.last_btn_clicked = 'pass'

    def fail_clicked(self):
        self.last_btn_clicked = 'fail'

    def user_passfail(self):
        self.widget.show()
        while True:
            pg.QtGui.QApplication.processEvents()
            last_btn_clicked = self.last_btn_clicked
            self.last_btn_clicked = None

            if last_btn_clicked == 'fail':
                raise Exception("User rejected test result.")
            elif last_btn_clicked == 'pass':
                break
            
            time.sleep(0.03)

    def show_results(self, expected, current):
        self.display1.show_result(expected)
        self.display2.show_result(current)
        self.diff_widget.setData(expected, current)

    def clear(self):
        self.display1.clear()
        self.display2.clear()
        self.diff_widget.setData(None, None)
