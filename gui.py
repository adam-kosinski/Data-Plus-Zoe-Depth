import numpy as np
import os
import functools
from PIL import Image

from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QPushButton
)
from PyQt6 import uic

from gui_utils import clear_layout_contents, depth_to_pixmap
from calibration_manager import CalibrationManager
from depth_estimation_worker import DepthEstimationWorker



# TODO

# automatic cropping

# if implement localized method, make sure it supports any image size (padding shouldn't be fixed)

# automatic labeled output as an option (see label_results.py)

# bundle the zoedepth weights with the build, instead of downloading from the internet, to remove dependence on the internet download

# make deployments its own folder, so we don't have to check for our created folders

# if someone picks a calibration image and then picks a different one, both depth callbacks will occur, but I'm not sure the order is guaranteed
# - would be nice of having a way to cancel a job. Perhaps using processes works better
# - or just disconnect the result signal, zoedepth will finish but won't do anything with its output

# zoedepth calibration image computation slow? I wonder if you can pick default calibration images (e.g. first image in deployment) ahead of time and start processing in the background
# or - should be able to save a calibration without depth computed yet, and have the depth + linear regression finishing computing in a thread somewhere

# note that there is a GUI lag spike when the model is loaded

# sort calibration deployments by deployment name, when adding to calibrated box

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

        self.threadpool = QThreadPool()

        self.calibration_manager = CalibrationManager(self)
        self.zoedepth_model = None

        self.deployment_hboxes = {} # keep references to these so we can move them between the uncalibrated and calibrated lists

        self.output_rows = []

        # event listeners
        self.openRootFolder.clicked.connect(self.open_root_folder)
        self.runButton.clicked.connect(self.run_depth_estimation)


        # temp
        # self.open_root_folder("C:/Users/AdamK/Documents/ZoeDepth/bigger_test")
        self.resize(QSize(800, 600))
    
    
        

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
            if path == "calibration" or path == "detections" or path == "depth_maps" or path == "segmentation" or path == "labeled_output":
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
    
    def run_depth_estimation(self):
        # self.runButton.setEnabled(False)
        print("RUNNING DEPTH ESTIMATION ===============================")

        # reset rows so we don't get duplicates
        self.output_rows = []
        
        worker = DepthEstimationWorker(self)
        self.threadpool.start(worker)



            
            
            
            






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