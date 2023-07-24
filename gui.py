import numpy as np
import os
import functools
from PIL import Image
import json
import csv
import torch

from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QPoint, QThread
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
from zoe_worker import build_zoedepth_model
from zoedepth.utils.misc import save_raw_16bit, colorize
import run_segmentation

# megadetector stuff
import sys
sys.path.append("MegaDetector")
sys.path.append("yolov5")
import run_detector_batch


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



SEGMENTATION_RESIZE_FACTOR = 4






class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "gui.ui")
        uic.loadUi(ui_path, self)

        self.root_path = None

        self.threadpool = QThreadPool()

        self.calibration_manager = CalibrationManager(self)
        # self.zoe_manager = ZoeManager(self)
        self.zoedepth_model = None

        self.deployment_hboxes = {} # keep references to these so we can move them between the uncalibrated and calibrated lists

        self.output_rows = []

        # event listeners
        self.openRootFolder.clicked.connect(self.open_root_folder)
        self.runButton.clicked.connect(self.run_depth_estimation)


        # temp
        # self.open_root_folder("C:/Users/AdamK/Documents/ZoeDepth/bigger_test")
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



class DepthEstimationSignals(QObject):
    megadetector_done = pyqtSignal(object)  # doesn't carry data, but not providing an argument broke stuff
    # zoedepth_progress = pyqtSignal(int, int)   # current index (starting at 1), total files to process
    done = pyqtSignal(object)   # doesn't carry data



class DepthEstimationWorker(QRunnable):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.root_path = main_window.root_path
        self.calibration_json = main_window.calibration_manager.get_json()

        self.signals = DepthEstimationSignals()


    def run(self):
        # get inference files for each deployment

        inference_file_dict = {}
        for deployment in self.calibration_json:
            inference_file_dict[deployment] = self.choose_inference_files(deployment)


        # run megadetector (do all of this before zoedepth to avoid weird conflicts when building the zoe model)
        
        detections_dir = os.path.join(self.root_path, "detections")
        os.makedirs(detections_dir, exist_ok=True)
        
        for deployment in self.calibration_json:
            print(deployment)
            
            output_file = os.path.join(detections_dir, deployment + ".json")
            if not os.path.exists(output_file):
                input_dir = os.path.join(self.root_path, deployment)
                arg_string = f"megadetector_weights/md_v5a.0.0.pt {input_dir} {output_file} --threshold 0.5"
                run_detector_batch.main(arg_string)
            print("Megadetector done with deployment", deployment)
        
        print("Megadetector done")
        self.signals.megadetector_done.emit(None)



        # run zoedepth, segmentation, and get animal distances

        # by default, don't load zoedepth, only load it if we need it
        zoe = None

        for deployment in self.calibration_json:
            inference_files = inference_file_dict[deployment]

            input_dir = os.path.join(self.root_path, deployment)
            segmentation_dir = os.path.join(self.root_path, "segmentation", deployment)
            depth_maps_dir = os.path.join(self.root_path, "depth_maps", deployment)
            os.makedirs(segmentation_dir, exist_ok=True)
            os.makedirs(depth_maps_dir, exist_ok=True)
            with open(os.path.join(detections_dir, deployment + ".json")) as json_file:
                bbox_json = json.load(json_file)

            # get calibrated reference depth for this deployment
            calib_depth = self.get_calib_depth(deployment)

            
            # segmentation
            segmentation_filepath_dict = run_segmentation.main(input_dir, output_dir=segmentation_dir, resize_factor=SEGMENTATION_RESIZE_FACTOR, inference_files=inference_files)


            for image_abs_path in inference_files:
                
                # check if an image
                ext = os.path.splitext(image_abs_path)[1].lower()
                if not (ext == ".jpg" or ext == ".jpeg" or ext == ".png"):
                    continue
                
                # get detections, and skip this image if no animals detected
                detections = []
                for obj in bbox_json["images"]:
                    if obj["file"] == image_abs_path:
                        detections = list(filter(lambda detection: detection["category"] == "1", obj["detections"]))   # category animal
                if len(detections) == 0:
                    continue
                
                # run zoedepth
                print("Getting depth for", image_abs_path)
                # check if depth was already calculated, if so use it and skip zoedepth calculation
                depth_basename = os.path.splitext(os.path.basename(image_abs_path))[0] + "_raw.png"
                depth_path = os.path.join(depth_maps_dir, depth_basename)
                if os.path.exists(depth_path):
                    with Image.open(depth_path) as depth_img:
                        depth = np.asarray(depth_img) / 256
                else:
                    # run zoedepth to get depth, save raw file
                    # load zoedepth if this is the first time we're using it
                    if not self.main_window.zoedepth_model:
                        print("Loading ZoeDepth")
                        self.main_window.zoedepth_model = build_zoedepth_model()

                    print("Running zoedepth on", image_abs_path)
                    with Image.open(image_abs_path).convert("RGB") as image:
                        depth = self.main_window.zoedepth_model.infer_pil(image)
                    save_basename = os.path.splitext(os.path.basename(image_abs_path))[0] + "_raw.png"
                    save_path = os.path.join(depth_maps_dir, save_basename)
                    save_raw_16bit(depth, save_path)
                
                # calibrate
                # lazy calibration
                norm_depth = (depth - np.mean(depth)) / np.std(depth)
                depth = np.maximum(0, norm_depth * np.std(calib_depth) + np.mean(calib_depth))

                # extract animal depths and save
                # we could do this after processing all the files from this deployment, but one file at a time lets you see the results popping up in real time :)

                output = []

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

                    # get bbox depth mask
                    with Image.open(segmentation_filepath_dict[image_abs_path]) as mask_img:
                        mask_img = mask_img.resize((mask_img.width * SEGMENTATION_RESIZE_FACTOR, mask_img.height * SEGMENTATION_RESIZE_FACTOR))
                        animal_mask = np.asarray(mask_img)
                        bbox_animal_mask = animal_mask[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
                        bbox_animal_mask = bbox_animal_mask.astype(bool)
                    
                    # get depth estimate
                    segmentation_median_estimate = np.median(bbox_depth[bbox_animal_mask]) if True in bbox_animal_mask else None
                    percentile_estimate = np.percentile(bbox_depth, 20)
                    if segmentation_median_estimate:
                        depth_estimate = segmentation_median_estimate
                        print("segmentation estimate")
                    else:
                        depth_estimate = percentile_estimate
                        print("percentile estimate")
                    print(depth_estimate)

                    # get sampled point (point with depth value = depth estimate, that's closest to bbox center)
                    if segmentation_median_estimate:
                        value_near_estimate = bbox_depth[bbox_animal_mask][np.argmin(np.abs(bbox_depth[bbox_animal_mask] - depth_estimate))]
                    else:
                        value_near_estimate = bbox_depth.flatten()[np.argmin(np.abs(bbox_depth.flatten() - depth_estimate))]
                    close_to_estimate = np.abs(bbox_depth - value_near_estimate) < 0.02
                    ys, xs = np.where(close_to_estimate & bbox_animal_mask) if segmentation_median_estimate else np.where(close_to_estimate)
                    dxs = xs - bbox_w/2
                    dys = ys - bbox_h/2
                    dists_to_center = np.hypot(dxs, dys)
                    idx = np.argmin(dists_to_center)
                    sample_x = bbox_x + xs[idx]
                    sample_y = bbox_y + ys[idx]
                    
                    output.append({
                        "deployment": deployment,
                        "filename": os.path.basename(image_abs_path),
                        "animal_depth": depth_estimate,
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_width": bbox_w,
                        "bbox_height": bbox_h,
                        "sample_x": sample_x,
                        "sample_y": sample_y
                    })

                # write results

                if len(output) == 0:
                    continue

                self.main_window.output_rows += output

                # update the output csv
                output_fpath = os.path.join(self.root_path, "output.csv")
                with open(output_fpath, 'w', newline='') as csvfile:
                    fieldnames = list(output[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in self.main_window.output_rows:
                        writer.writerow(row)
        
        print("DONE!!!!!!!!!!!!!!!!!!!!")
        self.signals.done.emit(None)


    def choose_inference_files(self, deployment):
        # not all image files in a deployment are necessarily being used for inference
        # some are there just to help the segmentation work better
        # this function returns a list of absolute image paths that we want to do inference on
        inference_files = []
        deployment_dir = os.path.join(self.root_path, deployment)
        for file in os.listdir(deployment_dir):
            s = os.path.splitext(file)[0]
            if "-" not in s:
                inference_files.append(os.path.abspath(os.path.join(deployment_dir, file)))
        return inference_files


    def get_calib_depth(self, deployment):
        rel_depth_path = os.path.join(self.root_path, self.calibration_json[deployment]["rel_depth_path"])
        with Image.open(rel_depth_path) as calib_depth_img:
            calib_depth = np.asarray(calib_depth_img) / 256
        slope = self.calibration_json[deployment]["slope"]
        intercept = self.calibration_json[deployment]["intercept"]
        calib_depth = calib_depth * slope + intercept
        return calib_depth

            
            
            
            






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