import numpy as np
import os
import functools
from PIL import Image
import json
import csv

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
from zoedepth.utils.misc import save_raw_16bit

# megadetector stuff
import sys
sys.path.append("MegaDetector")
sys.path.append("yolov5")
import run_detector_batch


# TODO

# if someone picks a calibration image and then picks a different one, both depth callbacks will occur, but I'm not sure the order is guaranteed
# - would be nice of having a way to cancel a job. Perhaps using processes works better
# - or just disconnect the result signal, zoedepth will finish but won't do anything with its output

# share the threadpool, but somehow give zoedepth priority over depth estimation threads so they don't wait on each other to finish
# - or just better understand how two threadpools work

# Consider having multiple zoe managers for potential speed-up? Since it didn't work so well using one manager for multiple threads
# - probably there was weirdness with multiple models trying to access the manager's zoe model object at the same time

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
        self.zoe_manager = ZoeManager(self)

        self.deployment_hboxes = {} # keep references to these so we can move them between the uncalibrated and calibrated lists

        self.output_rows = []

        # event listeners
        self.openRootFolder.clicked.connect(self.open_root_folder)
        self.runButton.clicked.connect(self.run_depth_estimation)


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
            if path == "calibration" or path == "detections" or path == "depth_maps":
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

        # get calibrations, will only run on calibrated deployments
        json_data = self.calibration_manager.get_json()
        
        for deployment in json_data:
            worker = DepthEstimationWorker(self, deployment, json_data[deployment])
            worker.signals.result.connect(self.process_animal_depth_results)
            self.threadpool.start(worker)
    
    def process_animal_depth_results(self, new_rows):
        if len(new_rows) == 0:
            return

        self.output_rows += new_rows

        # sort csv since depths come back in a weird order due to threading
        self.output_rows.sort(key=lambda obj: obj["filename"])      # secondary sort
        self.output_rows.sort(key=lambda obj: obj["deployment"])    # primary sort

        # update the output csv
        output_fpath = os.path.join(self.root_path, "output.csv")
        with open(output_fpath, 'w', newline='') as csvfile:
            fieldnames = list(self.output_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.output_rows:
                writer.writerow(row)



class DepthEstimationSignals(QObject):
    megadetector_done = pyqtSignal(object)  # doesn't carry data, but not providing an argument broke stuff
    zoedepth_progress = pyqtSignal(int, int)   # current index (starting at 1), total files to process
    result = pyqtSignal(object)    # contains the depth estimates, MainWindow will do the writing to output file



class DepthEstimationWorker(QRunnable):
    def __init__(self, main_window, deployment, calibration_json):
        super().__init__()
        self.main_window = main_window
        self.deployment = deployment
        self.calibration_json = calibration_json
        self.input_dir = os.path.join(main_window.root_path, deployment)
        self.detections_dir = os.path.join(self.main_window.root_path, "detections")
        self.depth_maps_dir = os.path.join(self.main_window.root_path, "depth_maps", deployment)
        os.makedirs(self.detections_dir, exist_ok=True)
        os.makedirs(self.depth_maps_dir, exist_ok=True)
        self.signals = DepthEstimationSignals()


    def run(self):

        # run megadetector

        output_file = os.path.join(self.detections_dir, self.deployment + ".json")

        if not os.path.exists(output_file):
            arg_string = f"megadetector_weights/md_v5a.0.0.pt {self.input_dir} {output_file} --threshold 0.2"
            run_detector_batch.main(arg_string)
        
        with open(output_file) as json_file:
            self.bbox_json = json.load(json_file)
        self.signals.megadetector_done.emit(None)
        
        # run zoedepth
        for file in os.listdir(self.input_dir):
            abs_path = os.path.join(self.input_dir, file)
            
            # check if an image
            ext = os.path.splitext(file)[1].lower()
            if not (ext == ".jpg" or ext == ".jpeg" or ext == ".png"):
                continue
            
            # check if animals detected in image
            if not self.get_detections(abs_path):
                continue
            
            # check if depth was already calculated, if so use it
            depth_basename = os.path.splitext(file)[0] + "_raw.png"
            depth_path = os.path.join(self.depth_maps_dir, depth_basename)
            if os.path.exists(depth_path):
                with Image.open(depth_path) as depth_img:
                    depth = np.asarray(depth_img) / 256
                self.process_depth(depth, abs_path)
                continue
            
            # run zoedepth to get depth
            self.main_window.zoe_manager.infer(abs_path, self.zoe_callback)

    def get_detections(self, img_abspath):
        for obj in self.bbox_json["images"]:
            if os.path.samefile(obj["file"], img_abspath):
                animal_detections = list(filter(lambda detection: detection["category"] == "1", obj["detections"]))   # category animal
                return animal_detections if len(animal_detections) > 0 else None
        return None

    def zoe_callback(self, depth, img_abspath):
        save_basename = os.path.splitext(os.path.basename(img_abspath))[0] + "_raw.png"
        save_path = os.path.join(self.depth_maps_dir, save_basename)
        save_raw_16bit(depth, save_path)
        self.process_depth(depth, img_abspath)
    
    def process_depth(self, depth, img_abspath):
        print("processing depth", self.deployment, img_abspath)

        output = []

        # get calibrated reference depth
        rel_depth_path = os.path.join(self.main_window.root_path, self.calibration_json["rel_depth_path"])
        with Image.open(rel_depth_path) as calib_depth_img:
            calib_depth = np.asarray(calib_depth_img) / 256
        slope = self.calibration_json["slope"]
        intercept = self.calibration_json["intercept"]
        calib_depth = calib_depth * slope + intercept

        # calibrate
        # lazy calibration
        norm_depth = (depth - np.mean(depth)) / np.std(depth)
        depth = np.maximum(0, norm_depth * np.std(calib_depth) + np.mean(calib_depth))

        detections = self.get_detections(img_abspath)

        for detection in detections:
            # crop depth to bbox
            b = detection["bbox"]
            h = depth.shape[0]
            w = depth.shape[1]
            bbox_x = int(w*b[0])
            bbox_y = int(h*b[1])
            bbox_w = int(w*b[2])
            bbox_h = int(h*b[3])
            bbox_depth = depth[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

            # get 20th percentile in bbox_depth
            depth_estimate = np.percentile(bbox_depth, 20)
            # get sampled point
            # print(np.where(np.isclose(bbox_depth, depth_estimate)))
            output.append({
                "deployment": self.deployment,
                "filename": os.path.basename(img_abspath),
                "animal_depth": depth_estimate,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_width": bbox_w,
                "bbox_height": bbox_h
            })
        
        self.signals.result.emit(output)
            
            
            
            






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