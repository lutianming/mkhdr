import os
import sys
from PySide.QtCore import *
from PySide.QtGui import *
from mkhdr import *
from PIL import ImageQt


class ImageWindow(QMainWindow):
    def __init__(self):
        super(ImageWindow, self).__init__()

        self.images = None
        self.times = None
        self.hdr = None

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

        main_widget = QWidget(self)
        hbox = QHBoxLayout(main_widget)
        self.image_listview = QListView(self)
        self.image_listview.setViewMode(QListView.IconMode)
        self.image_listview.setIconSize(QSize(200, 200))
        self.listview_model = QStandardItemModel(self.image_listview)
        self.image_listview.setModel(self.listview_model)

        self.image_scene = QGraphicsScene()
        mainview = QGraphicsView(self.image_scene)
        # self.image_label = QLabel(self)
        # self.image_label.setText('test')

        hbox.addWidget(self.image_listview, stretch=1)
        # hbox.addWidget(self.image_label, stretch=4)
        hbox.addWidget(mainview, stretch=4)

        main_widget.setLayout(hbox)
        self.setCentralWidget(main_widget)
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
            # for img in images:
            #     tmp = img.copy()
            #     tmp.thumbnail((128, 128))
            #     imgqt = ImageQt.ImageQt(tmp)
            #     pixmap = QPixmap.fromImage(imgqt)
            #     icon = QIcon(pixmap)

            #     item = QStandardItem(icon, 'image')
            #     self.listview_model.appendRow(item)
            for f in fnames:
                icon = QIcon(f)
                item = QStandardItem(icon, os.path.basename(f))
                self.listview_model.appendRow(item)

    def save_hdr(self):
        if self.hdr:
            fname, _ = QFileDialog.getSaveFileName(self, 'Save HDF', './')
            self.hdr.save(fname)
        else:
            QMessageBox.warning(self, 'warning',
                                'Please generate a HDR image before saving')

    def gen_hdr(self):
        self.hdr = make_hdr(self.images, self.times)
        data = self.hdr.tostring('raw', 'RGB')
        img = QImage(data, self.hdr.size[0], self.hdr.size[1],
                     QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.image_scene.clear()
        self.image_scene.addPixmap(pixmap)

def start_ui(argv):
    app = QApplication(argv)
    window = ImageWindow()
    sys.exit(app.exec_())
