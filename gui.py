import numpy as np
import os
import functools
import math
from PIL import Image
from PIL.ImageQt import ImageQt

from zoedepth.utils.misc import colorize


from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QPoint
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic


def clear_layout(layout):
    for i in reversed(range(layout.count())): 
        widgetToRemove = layout.itemAt(i).widget()
        layout.removeWidget(widgetToRemove)
        widgetToRemove.deleteLater()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)

        self.openRootFolder.clicked.connect(self.open_root_folder)

        # temp
        self.open_root_folder()
        with Image.open("second_results/calibrated/RCNX0332_raw.png") as raw_img:
            self.data = np.asarray(raw_img) / 256
        with Image.open("second_results/RCNX0332_colored.png") as img:
            rounded = np.floor(self.data)
            colored = colorize(rounded)
            img = Image.fromarray(colored)
            qt_depthmap = ImageQt(img)
            self.pixmap = QPixmap.fromImage(qt_depthmap).scaledToWidth(400)
            self.imgDisplay.setPixmap(self.pixmap)
        
        self.imgDisplay.setMouseTracking(True)
        self.imgDisplay.mouseMoveEvent = self.update_dist
    
    def update_dist(self, e):
        rect = self.pixmap.rect()
        x = math.floor(self.data.shape[1] * e.position().x() / rect.width())
        y = math.floor(self.data.shape[0] * e.position().y() / rect.height())
        depth_val = round(self.data[y][x], 1)

        box = self.meterContainer
        dest = box.parentWidget().mapFromGlobal(e.globalPosition())
        box.move(int(dest.x()) + 20, int(dest.y()) + 10)
        self.meterDisplay.setText(f"{depth_val} m ")
        

    def open_root_folder(self):
        # dialog = QFileDialog(self)
        # dialog.setFileMode(QFileDialog.FileMode.Directory)
        # if dialog.exec():
        #     self.root_path = dialog.selectedFiles()[0]
        #     self.rootFolderLabel.setText(self.root_path)
        self.root_path = "."
        if True:

            # display deployments
            grid = self.deploymentGrid
            grid.setColumnStretch(1, 1)
            clear_layout(grid)
            for item in os.listdir(self.root_path):
                if not os.path.isdir(item):
                    continue
                row = grid.rowCount()
                button = QPushButton("Calibrate")
                button.clicked.connect(functools.partial(self.open_calibration_screen, item))
                grid.addWidget(button, row, 0)
                grid.addWidget(QLabel(item), row, 1)
    
    def open_calibration_screen(self, deployment):
        print(deployment)
        self.screens.setCurrentWidget(self.calibrationScreen)



if __name__ == '__main__':
    # create application
    app = QApplication([]) # replace [] with sys.argv if want to use command line args

    # create window
    window = MainWindow()
    window.show()

    # allow ctrl + C to work
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # start the event loop
    app.exec()