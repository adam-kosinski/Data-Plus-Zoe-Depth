import numpy as np
import os
import functools
from PIL import Image
import json
import csv

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from zoe_worker import build_zoedepth_model
from zoedepth.utils.misc import save_raw_16bit
import run_segmentation
from label_results import label_results

# megadetector stuff
import sys
sys.path.append("MegaDetector")
sys.path.append("yolov5")
import run_detector_batch


SEGMENTATION_RESIZE_FACTOR = 4



class DepthEstimationSignals(QObject):
    megadetector_done = pyqtSignal(object)  # doesn't carry data, but not providing an argument broke stuff
    # zoedepth_progress = pyqtSignal(int, int)   # current index (starting at 1), total files to process
    done = pyqtSignal(object)   # doesn't carry data



class DepthEstimationWorker(QRunnable):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.root_path = main_window.root_path
        self.deployments_dir = main_window.deployments_dir
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
                input_dir = os.path.join(self.deployments_dir, deployment)
                arg_string = f"megadetector_weights/md_v5a.0.0.pt {input_dir} {output_file} --threshold 0.5"
                run_detector_batch.main(arg_string)
            print("Megadetector done with deployment", deployment)
        
        print("Megadetector done")
        self.signals.megadetector_done.emit(None)



        # run zoedepth, segmentation, and get animal distances

        # by default, don't load zoedepth, only load it if we need it

        for deployment in self.calibration_json:
            inference_files = inference_file_dict[deployment]

            input_dir = os.path.join(self.deployments_dir, deployment)
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

                self.main_window.csv_output_rows += output

                # update the output csv
                output_fpath = os.path.join(self.root_path, "output.csv")
                with open(output_fpath, 'w', newline='') as csvfile:
                    fieldnames = list(output[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in self.main_window.csv_output_rows:
                        writer.writerow(row)
        

        label_results(self.root_path)
        
        print("DONE!!!!!!!!!!!!!!!!!!!!")
        self.signals.done.emit(None)


    def choose_inference_files(self, deployment):
        # not all image files in a deployment are necessarily being used for inference
        # some are there just to help the segmentation work better
        # this function returns a list of absolute image paths that we want to do inference on
        inference_files = []
        input_dir = os.path.join(self.deployments_dir, deployment)
        for file in os.listdir(input_dir):
            s = os.path.splitext(file)[0]
            if "-" not in s:
                inference_files.append(os.path.abspath(os.path.join(input_dir, file)))
        return inference_files


    def get_calib_depth(self, deployment):
        rel_depth_path = os.path.join(self.root_path, self.calibration_json[deployment]["rel_depth_path"])
        with Image.open(rel_depth_path) as calib_depth_img:
            calib_depth = np.asarray(calib_depth_img) / 256
        slope = self.calibration_json[deployment]["slope"]
        intercept = self.calibration_json[deployment]["intercept"]
        calib_depth = calib_depth * slope + intercept
        return calib_depth
