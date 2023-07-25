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

# bundle the zoedepth weights with the build, instead of downloading from the internet, to remove dependence on the internet download

# if someone picks a calibration image and then picks a different one, both depth callbacks will occur, but I'm not sure the order is guaranteed
# - would be nice of having a way to cancel a job. Perhaps using processes works better
# - or just disconnect the result signal, zoedepth will finish but won't do anything with its output

# zoedepth calibration image computation slow? I wonder if you can pick default calibration images (e.g. first image in deployment) ahead of time and start processing in the background
# or - should be able to save a calibration without depth computed yet, and have the depth + linear regression finishing computing in a thread somewhere

# note that there is a GUI lag spike when the model is loaded

# sort calibration deployments by deployment name, when adding to calibrated box

# graphics view sizing (ideally allow the images to get bigger if they're able)
# this involves changing the x,y coords in calibrations.json to fractional units of total width/height

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
        self.deployments_dir = None

        self.threadpool = QThreadPool()

        self.calibration_manager = CalibrationManager(self)
        self.zoedepth_model = None

        self.deployment_hboxes = {} # keep references to these so we can move them between the uncalibrated and calibrated lists
        self.is_deployment_calibrated = {}  # deployment: bool (whether calibrated)

        self.csv_output_rows = []

        # event listeners
        self.openRootFolder.clicked.connect(self.open_root_folder)
        self.runButton.clicked.connect(self.run_depth_estimation)


        # temp
        self.open_root_folder("C:/Users/AdamK/Documents/ZoeDepth/test")
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
        self.deployments_dir = os.path.join(self.root_path, "deployments")
        self.rootFolderLabel.setText(self.root_path)
        self.calibration_manager.update_root_path()
        print(self.root_path)
        
        # display deployments

        clear_layout_contents(self.uncalibratedDeployments)
        clear_layout_contents(self.calibratedDeployments)
        self.deployment_hboxes = {}

        calibration_json = self.calibration_manager.get_json()
        
        for deployment in os.listdir(self.deployments_dir):
            if not os.path.isdir(os.path.join(self.deployments_dir, deployment)):
                continue
            button = QPushButton()
            button.clicked.connect(functools.partial(self.calibration_manager.init_calibration, deployment))   # functools for using the current value of item, not whatever it ends up being
            hbox = QHBoxLayout()
            self.deployment_hboxes[deployment] = hbox
            hbox.widget_name = "deployment_" + deployment + "_hbox"
            hbox.addWidget(button)
            hbox.addWidget(QLabel(deployment))
            hbox.addStretch()
            if deployment in calibration_json:
                self.calibratedDeployments.addLayout(hbox)
                self.is_deployment_calibrated[deployment] = True
                button.setText("Edit Calibration")
            else:
                self.uncalibratedDeployments.addLayout(hbox)
                self.is_deployment_calibrated[deployment] = False
                button.setText("Calibrate")
    
    def run_depth_estimation(self):
        # self.runButton.setEnabled(False)
        print("RUNNING DEPTH ESTIMATION ===============================")

        # reset rows so we don't get duplicates
        self.csv_output_rows = []
        
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