import sys
from PySide import QtCore
from PySide.QtGui import (QMainWindow, QApplication, QAction,
                          QFileDialog, QMessageBox)
from mkhdr import *

images = None
times = None
hdr = None


class ImageWindow(QMainWindow):
    def __init__(self):
        super(ImageWindow, self).__init__()
        self.setWindowTitle('mkhdr')

        loadAction = QAction('&Load', self)
        loadAction.triggered.connect(self.load_images)

        hdrAction = QAction('HDR', self)
        hdrAction.triggered.connect(self.gen_hdr)

        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.save_hdr)

        self.toolbar = self.addToolBar('toolbar')
        self.toolbar.addAction(loadAction)
        self.toolbar.addAction(hdrAction)
        self.toolbar.addAction(saveAction)

        self.show()

    def load_images(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open files', './')
        if len(fnames) < 4:
            QMessageBox.warning(self, 'warning',
                                'please choose no less than 4 images')
        else:
            print(fnames)
            images, times = read_images(fnames)

    def save_hdr(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save HDF', './')
        print(fname)
        if hdr:
            hdr.save(fname)
        else:
            QMessageBox.warning(self, 'warning',
                                'Please choose original images first')

    def gen_hdr(self):
        hdr = make_hdr(images, times)


def start_ui(argv):
    app = QApplication(argv)
    window = ImageWindow()
    sys.exit(app.exec_())
