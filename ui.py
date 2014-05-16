import os
import sys
from PySide.QtCore import *
from PySide.QtGui import *
from mkhdr import *
from PIL import Image
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HDRSignal(QObject):
    sig = Signal(Image)


class HDRThread(QThread):
    def __init__(self, parent=None):
        super(HDRThread, self).__init__(parent)
        self.signal = HDRSignal()
        self.status = 0

    def run(self):
        parent = self.parent()
        images = parent.images
        times = parent.times
        args = parent.get_parameter()
        parent.hdr, parent.g = make_hdr(images, times, args)


class ImageWindow(QMainWindow):
    def __init__(self, args):
        super(ImageWindow, self).__init__()

        self.args = args
        self.images = None
        self.times = None
        self.hdr = None
        self.g = None

        self.setWindowTitle('mkhdr')
        self.loadAction = QAction('&Load', self)
        self.loadAction.triggered.connect(self.load_images)

        self.hdrAction = QAction('HDR', self)
        self.hdrAction.triggered.connect(self.gen_hdr)
        self.hdr_thread = HDRThread(self)
        # self.hdr_thread.signal.sig.connect(self.update_hdr)
        self.hdr_thread.started.connect(self.hdr_thread_start)
        self.hdr_thread.finished.connect(self.hdr_thread_finished)
        self.hdr_thread.finished.connect(self.update_hdr)

        self.saveAction = QAction('Save', self)
        self.saveAction.triggered.connect(self.save_hdr)

        self.toolbar = self.addToolBar('toolbar')
        self.toolbar.addAction(self.loadAction)
        self.toolbar.addAction(self.hdrAction)
        self.toolbar.addAction(self.saveAction)

        #main widget
        main_widget = QWidget(self)
        hbox = QHBoxLayout(main_widget)

        #list view of original images
        image_listview = QListView(self)
        image_listview.setViewMode(QListView.IconMode)
        image_listview.setIconSize(QSize(200, 200))
        self.listview_model = QStandardItemModel(image_listview)
        image_listview.setModel(self.listview_model)

        #main view of the hdr image
        self.image_scene = QGraphicsScene()
        self.mainview = QGraphicsView(self.image_scene)

        #toolbox of different parameters
        toolbox = QWidget(self)
        formbox = QFormLayout(toolbox)

        font = QFont()
        font.setBold(True)

        label = QLabel('radinance map')
        label.setFont(font)
        formbox.addRow(label)
        self.lambda_box = QSpinBox(toolbox)
        self.lambda_box.setRange(0, 200)
        formbox.addRow('lambda', self.lambda_box)

        self.samples_box = QSpinBox(toolbox)
        self.samples_box.setRange(50, 500)
        formbox.addRow('samples', self.samples_box)

        label = QLabel('tone mapping operators')
        label.setFont(font)
        formbox.addRow(label)
        self.tone_mapping_box = QComboBox(toolbox)
        self.tone_mapping_box.addItems(["global_simple",
                                        "global_reinhards",
                                        "local_durand"])
        formbox.addRow('tone mapping op', self.tone_mapping_box)
        self.saturation_box = QDoubleSpinBox(toolbox)
        self.saturation_box.setRange(0, 10)
        self.saturation_box.setSingleStep(0.01)
        formbox.addRow('saturation', self.saturation_box)

        self.gamma_box = QDoubleSpinBox(toolbox)
        self.saturation_box.setRange(0.1, 5)
        self.saturation_box.setSingleStep(0.1)
        formbox.addRow('gamma', self.gamma_box)

        label = QLabel('parameters for local durand')
        label.setFont(font)
        formbox.addRow(label)
        self.sigma_r_box = QDoubleSpinBox(toolbox)
        self.sigma_r_box.setRange(0, 200)
        self.sigma_r_box.setSingleStep(0.01)
        formbox.addRow('sigma r', self.sigma_r_box)

        self.sigma_d_box = QSpinBox(toolbox)
        self.sigma_d_box.setRange(0, 200)
        formbox.addRow('sigma d', self.sigma_d_box)

        label = QLabel('parameters for global reinhards')
        label.setFont(font)
        formbox.addRow(label)
        self.a_box = QDoubleSpinBox(toolbox)
        self.a_box.setRange(0, 1)
        self.a_box.setSingleStep(0.01)
        formbox.addRow('a', self.a_box)

        self._reset_parameters()

        #figure to show g plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        formbox.addRow(self.canvas)

        #add widgets to main layout
        hbox.addWidget(image_listview, stretch=1)
        hbox.addWidget(self.mainview, stretch=4)
        hbox.addWidget(toolbox, stretch=1)

        main_widget.setLayout(hbox)
        self.setCentralWidget(main_widget)
        self.statusBar().showMessage('Ready')
        self.show()

    def load_images(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open files', './')
        if len(fnames) == 0:
            return
        if len(fnames) < 2:
            QMessageBox.warning(self, 'warning',
                                'please choose no less than 2 images')
        else:
            print(fnames)
            self.images, self.times = read_images(fnames)
            self.listview_model.clear()
            for f in fnames:
                icon = QIcon(f)
                item = QStandardItem(icon, os.path.basename(f))
                self.listview_model.appendRow(item)
                self.statusBar().showMessage('Images loaded')

    def save_hdr(self):
        if self.hdr:
            fname, _ = QFileDialog.getSaveFileName(self, 'Save HDF', './')
            self.hdr.save(fname)
            self.statusBar().showMessage('HDR image saved')
        else:
            QMessageBox.warning(self, 'warning',
                                'Please generate a HDR image before saving')

    def gen_hdr(self):
        if not self.hdr_thread.isRunning():
            self.statusBar().showMessage("generating HDR image...")
            self.hdrAction.setEnabled(False)
            self.hdr_thread.start()

    def hdr_thread_start(self):
        self.hdrAction.setEnabled(False)

    def hdr_thread_finished(self):
        self.hdrAction.setEnabled(True)

    def update_hdr(self):
        data = self.hdr.tostring('raw', 'RGB')
        img = QImage(data, self.hdr.size[0], self.hdr.size[1],
                     QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.image_scene.clear()
        self.image_scene.addPixmap(pixmap)
        self.mainview.fitInView(self.mainview.sceneRect(),
                                aspectRadioMode=Qt.KeepAspectRatio)
        self.plot_g()
        self.statusBar().showMessage('HDR image generated')

    def get_parameter(self):
        args = {}
        args['lambda'] = self.lambda_box.value()
        args['samples'] = self.samples_box.value()
        args['tone_mapping_op'] = self.tone_mapping_box.currentText()
        args['sigma_r'] = self.sigma_r_box.value()
        args['sigma_d'] = self.sigma_d_box.value()
        args['a'] = self.a_box.value()
        args['saturation'] = self.saturation_box.value()
        args['gamma'] = self.gamma_box.value()
        return args

    def _reset_parameters(self):
        args = self.args
        self.lambda_box.setValue(args['lambda'])
        self.samples_box.setValue(args['samples'])
        index = self.tone_mapping_box.findText(args['tone_mapping_op'])
        self.tone_mapping_box.setCurrentIndex(index)
        self.sigma_r_box.setValue(args['sigma_r'])
        self.sigma_d_box.setValue(args['sigma_d'])
        self.a_box.setValue(args['a'])
        self.saturation_box.setValue(args['saturation'])
        self.gamma_box.setValue(args['gamma'])

    def plot_g(self):
        x = range(0, 256)
        g = self.g
        channels = g.shape[0]
        ax = self.figure.add_subplot(111)
        ax.clear()
        for channel in range(channels):
            ax.plot(g[channel, :], x)
        # refresh canvas
        self.canvas.draw()

def start_ui(argv, args):
    app = QApplication(argv)
    window = ImageWindow(args)
    sys.exit(app.exec_())
