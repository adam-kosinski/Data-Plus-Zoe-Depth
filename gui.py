import numpy as np
import os
import functools
from PIL import Image
import subprocess
import platform

from PyQt5.QtCore import QSize, Qt, QRunnable, QThreadPool, pyqtSignal, QThread, QObject
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton
)

from PyQt5.QtGui import QIcon

from PyQt5 import uic


from gui_utils import clear_layout_contents, depth_to_pixmap
from calibration_manager import CalibrationManager
from crop_manager import CropManager
from depth_estimation_worker import DepthEstimationWorker


# icon stuff for windows
try:
    from ctypes import windll  # Only exists on Windows.
    # app id is company.product.subproduct.version
    myappid = 'DataPlus.WildlifeDepthEstimation.Tool.v1.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass



# TODO

# if we update megadetector to only use inference images, update the total relative time calculation for the progress bar to reflect this

# automatic cropping

# if implement localized method, make sure it supports any image size (padding shouldn't be fixed)

# bundle the zoedepth weights with the build, instead of downloading from the internet, to remove dependence on the internet download

# estimated time remaining? can get a sense of how fast the person's computer is after we've done either megadetector or zoedepth once, and then use relative times to extrapolate?

# if someone picks a calibration image and then picks a different one, both depth callbacks will occur, but I'm not sure the order is guaranteed
# - would be nice of having a way to cancel a job. Perhaps using processes works better
# - or just disconnect the result signal, zoedepth will finish but won't do anything with its output

# zoedepth calibration image computation slow? I wonder if you can pick default calibration images (e.g. first image in deployment) ahead of time and start processing in the background
# or - should be able to save a calibration without depth computed yet, and have the depth + linear regression finishing computing in a thread somewhere

# note that there is a GUI lag spike when the model is loaded

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
        self.stopButton.hide()
        self.openOutputCSVButton.hide()
        self.openOutputVisualizationButton.hide()
        self.set_progress_message("")   # done here so in Qt Designer the label would still be visible

        self.root_path = None
        self.deployments_dir = None

        self.threadpool = QThreadPool()

        self.calibration_manager = CalibrationManager(self)
        self.crop_manager = CropManager(self)
        self.zoedepth_model = None

        self.deployment_hboxes = {} # keep references to these so we can move them between the uncalibrated and calibrated lists

        self.csv_output_rows = []

        # event listeners
        self.openRootFolder.clicked.connect(self.open_root_folder)
        self.runButton.clicked.connect(self.run_depth_estimation)
        self.openOutputCSVButton.clicked.connect(lambda: self.system_open(os.path.join(self.root_path, "output.csv")))
        self.openOutputVisualizationButton.clicked.connect(lambda: self.system_open(os.path.join(self.root_path, "output_visualization")))

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
            
            if not os.path.exists(os.path.join(dialog.selectedFiles()[0], "deployments")):
                QMessageBox.warning(self, "Invalid Folder Structure", "No subfolder named 'deployments' found, please choose a root folder with a subfolder named 'deployments.'")
                return
            
            self.root_path = dialog.selectedFiles()[0]

            self.stopButton.hide()
            self.openOutputCSVButton.hide()
            self.openOutputVisualizationButton.hide()
            self.set_progress_message("")

        self.root_path = os.path.normpath(self.root_path)    # normpath to keep slashses standardized, in case that matters
        self.deployments_dir = os.path.join(self.root_path, "deployments")
        self.rootFolderLabel.setText(self.root_path)

        print("root", self.root_path)
        self.calibration_manager.update_root_path()
        self.crop_manager.update_root_path()

        self.openCropScreenButton.setEnabled(True)
        
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
                button.setText("Edit Calibration")
            else:
                self.uncalibratedDeployments.addLayout(hbox)
                button.setText("Calibrate")
        
        self.runButton.setEnabled(True)
        self.set_progress_message('Click "Run Distance Estimation" to start, only the calibrated deployments will be processed')
        self.progressBar.reset()
    
    
    def run_depth_estimation(self):
        print("RUNNING DEPTH ESTIMATION ===============================")

        self.setAllButtonsEnabled(False)
        self.stopButton.show()
        self.openOutputCSVButton.hide()
        self.openOutputVisualizationButton.hide()

        # reset rows so we don't get duplicates
        self.csv_output_rows = []
        
        worker = DepthEstimationWorker(self)
        self.stopButton.clicked.connect(worker.stop)
        worker.signals.warning_popup.connect(self.warning_popup)
        worker.signals.message.connect(self.set_progress_message)
        worker.signals.progress.connect(self.set_progress_bar_value)
        worker.signals.stopped.connect(self.depth_estimation_thread_finished)   # separate signal than done for clarity, and in case we want different behavior in the future
        worker.signals.done.connect(self.depth_estimation_thread_finished)
        
        self.progressBar.reset()
        self.threadpool.start(worker)

    

    def setAllButtonsEnabled(self, enable):
        self.runButton.setEnabled(enable)
        self.openRootFolder.setEnabled(enable)
        self.openCropScreenButton.setEnabled(enable)
        for hbox in self.deployment_hboxes.values():
            button = hbox.itemAt(0).widget()
            button.setEnabled(enable)

    def warning_popup(self, title, message):
        QMessageBox.warning(self, title, message)

    def set_progress_message(self, message):
        self.progressMessage.setText(message)
    
    def set_progress_bar_value(self, value):
        self.progressBar.setValue(round(value))

    def depth_estimation_thread_finished(self):
        self.setAllButtonsEnabled(True)
        self.stopButton.hide()
        self.openOutputCSVButton.show()
        if os.path.exists(os.path.join(self.root_path, "output_visualization")):
            self.openOutputVisualizationButton.show()
    
    def system_open(self, filepath):
        if not os.path.exists(filepath):
            return
        
        # cross-platform solution from here: https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', filepath))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(filepath)
        else:                                   # linux variants
            subprocess.call(('xdg-open', filepath))







if __name__ == '__main__':
    # create application
    app = QApplication([]) # replace [] with sys.argv if want to use command line args

    # create window
    window = MainWindow()
    basedir = os.path.dirname(__file__)
    window.setWindowIcon(QIcon(os.path.join(basedir, "deer.ico")))
    window.show()

    # allow ctrl + C to work
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # start the event loop
    app.exec()