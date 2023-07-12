import numpy as np
import os
import functools
import math
from PIL import Image


from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QPoint
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QLineEdit
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QDoubleValidator
from PyQt6 import uic

from gui_utils import clear_layout_contents, depth_to_pixmap
from calibration_manager import CalibrationManager
from zoe_manager import ZoeManager


# TODO

# zoedepth calibration image computation slow? I wonder if you can pick default calibration images (e.g. first image in deployment) ahead of time and start processing in the background
# or - should be able to save a calibration without depth computed yet, and have the depth + linear regression finishing computing in a thread somewhere

# note that there is a lag spike right when the model is loaded, when it is being assigned to self.zoe I'm pretty sure, can figure out how to do that in a thread

# implement itemChange event for points, to avoid dragging out of bounds
# https://stackoverflow.com/questions/3548254/restrict-movable-area-of-qgraphicsitem

# graphics view sizing (ideally allow the images to get bigger if they're able)
# this involves changing the x,y coords in calibrations.json to fractional units of total width/height

# read depth values from depth view on hover

# Calibration "save and next" button or something like that? For faster workflow

# Perhaps for more flexible calibration in the future, add an option to draw a box around parts of the image that aren't part of the environment
# for instance, people holding a sign, or animals
# And it could potentially be nice to be able to use multiple images? If the researcher used the take-pics-of-person-at-distance method








class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "gui.ui")
        uic.loadUi(ui_path, self)

        self.root_path = None

        self.openRootFolder.clicked.connect(self.open_root_folder)
        
        self.calibration_manager = CalibrationManager(self)
        self.zoe_manager = ZoeManager(self)

        self.deployment_hboxes = {} # keep references to these so we can move them between the uncalibrated and calibrated lists

        # temp
        self.open_root_folder("C:/Users/AdamK/Documents/ZoeDepth/test")
        # with Image.open("second_results/calibrated/RCNX0332_raw.png") as raw_img:
        #     self.data = np.asarray(raw_img) / 256
        #     self.pixmap = depth_to_pixmap(self.data, rescale_width=400)
        #     self.imgDisplay.setPixmap(self.pixmap)
        
        # self.imgDisplay.setMouseTracking(True)
        # self.imgDisplay.mouseMoveEvent = self.update_dist
        
        self.resize(QSize(800, 600))
    
    # def update_dist(self, e):
    #     rect = self.pixmap.rect()
    #     x = math.floor(self.data.shape[1] * e.position().x() / rect.width())
    #     y = math.floor(self.data.shape[0] * e.position().y() / rect.height())
    #     depth_val = round(self.data[y][x], 1)

    #     box = self.meterContainer
    #     dest = box.parentWidget().mapFromGlobal(e.globalPosition())
    #     box.move(int(dest.x()) + 20, int(dest.y()) + 10)
    #     self.meterDisplay.setText(f"{depth_val} m ")
        

    def open_root_folder(self, root_path=None):
        if root_path:
            self.root_path = root_path
        else:
            dialog = QFileDialog(parent=self, caption="Choose Deployments Root Folder")
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            if not dialog.exec():
                return
            self.root_path = dialog.selectedFiles()[0]

        self.root_path = os.path.normpath(self.root_path)    # normpath to keep slashses standardized, in case that matters
        self.rootFolderLabel.setText(self.root_path)
        self.calibration_manager.set_root_path(self.root_path)
        print(self.root_path)
        
        # display deployments

        clear_layout_contents(self.uncalibratedDeployments)
        clear_layout_contents(self.calibratedDeployments)
        self.deployment_hboxes = {}

        json_data = self.calibration_manager.get_json()
        
        for path in os.listdir(self.root_path):
            if not os.path.isdir(os.path.join(self.root_path, path)):
                continue
            if path == "calibration":
                continue
            button = QPushButton("Calibrate")
            button.clicked.connect(functools.partial(self.calibration_manager.init_calibration, path))   # functools for using the current value of item, not whatever it ends up being
            hbox = QHBoxLayout()
            self.deployment_hboxes[path] = hbox
            hbox.widget_name = "deployment_" + path + "_hbox"
            hbox.addWidget(button)
            hbox.addWidget(QLabel(path))
            hbox.addStretch()
            if path in json_data:
                self.calibratedDeployments.addLayout(hbox)
            else:
                self.uncalibratedDeployments.addLayout(hbox)
    
    


    


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