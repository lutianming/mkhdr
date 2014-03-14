import sys
from PySide.QtCore import *
from PySide.QtGui import *


class ImageWindow(QMainWindow):
    def __init__(self):
        super(ImageWindow, self).__init__()
        self.setWindowTitle('mkhdr')
        self.show()


def start_ui(argv):
    app = QApplication(argv)
    window = ImageWindow()
    sys.exit(app.exec_())
