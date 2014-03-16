import os
import sys
from PySide.QtCore import *
from PySide.QtGui import *
from mkhdr import *
from PIL import Image


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
        parent.hdr = make_hdr(images, times)
        # self.signal.emit(hdr)


class ImageWindow(QMainWindow):
    def __init__(self):
        super(ImageWindow, self).__init__()

        self.images = None
        self.times = None
        self.hdr = None

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

        main_widget = QWidget(self)
        hbox = QHBoxLayout(main_widget)
        self.image_listview = QListView(self)
        self.image_listview.setViewMode(QListView.IconMode)
        self.image_listview.setIconSize(QSize(200, 200))
        self.listview_model = QStandardItemModel(self.image_listview)
        self.image_listview.setModel(self.listview_model)

        self.image_scene = QGraphicsScene()
        mainview = QGraphicsView(self.image_scene)

        hbox.addWidget(self.image_listview, stretch=1)
        hbox.addWidget(mainview, stretch=4)

        main_widget.setLayout(hbox)
        self.setCentralWidget(main_widget)
        self.statusBar().showMessage('Ready')
        self.show()

    def load_images(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open files', './')
        if len(fnames) == 0:
            return
        if len(fnames) < 4:
            QMessageBox.warning(self, 'warning',
                                'please choose no less than 4 images')
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
            self.hdrAction.setEnabled(False)
            self.hdr_thread.start()
            # self.hdr = make_hdr(self.images, self.times)

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
        self.statusBar().showMessage('HDR image generated')


def start_ui(argv):
    app = QApplication(argv)
    window = ImageWindow()
    sys.exit(app.exec_())
