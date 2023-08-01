import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json
import csv
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage import filters
from skimage.morphology import disk, opening
from r_pca import R_pca
import random
import platform

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from zoe_worker import build_zoedepth_model
from zoedepth.utils.misc import save_raw_16bit, colorize

# megadetector stuff
import sys
sys.path.append("MegaDetector")
sys.path.append("yolov5")
import run_detector_batch


SEGMENTATION_RESIZE_FACTOR = 4

# measured times in seconds used for progress bar, treat as relative though in case computer faster etc
START_TIME = 4  # show a little bit of the progress bar right at the start to let the user know stuff is happening
MEGADETECTOR_TIME_PER_IMAGE = 4.5
SEGMENTATION_TIME_PER_IMAGE = 2
ZOEDEPTH_BUILD_TIME = 12
ZOEDEPTH_INFER_TIME_PER_IMAGE = 20


class DepthEstimationSignals(QObject):
    message = pyqtSignal(str)
    warning_popup = pyqtSignal(str, str)    # title, message
    progress = pyqtSignal(float)  # 0-100 for progress percentage
    stopped = pyqtSignal()  # separate signal than done for clarity, and perhaps useful in the future
    done = pyqtSignal()



class DepthEstimationWorker(QRunnable):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.root_path = main_window.root_path
        self.deployments_dir = main_window.deployments_dir
        self.calibration_json = main_window.calibration_manager.get_json()

        self.signals = DepthEstimationSignals()
        self.stop_requested = False

        self.inference_file_dict = {}
        self.total_relative_time_estimated = 0  # initialized by summing config values - see run() function
        self.progress = 0   # 0-100


    def stop(self):
        print("stop")
        self.stop_requested = True
        self.signals.message.emit("Finding a good stopping place :)")


    # QRunnable override
    def run(self):

        # make sure we can edit the output file, also create the output file so that it exists even if there is no output
        output_filepath = os.path.join(self.root_path, "output.csv")
        try:
            with open(output_filepath, 'w', newline=''):
                pass
        except:
            self.signals.warning_popup.emit("Failed to Access Output File", f"Failed to access the output file:\n{output_filepath}\n\nThis may be because the file is open in another program. Please close other programs accessing the file and then click 'Run Depth Estimation' again.")
            self.signals.message.emit("Failed to access output file, please try running again.")
            self.signals.stopped.emit()
            return

        # get inference files for each deployment
        self.signals.message.emit("Selecting images to do inference on")
        self.inference_file_dict = {}
        for deployment in self.calibration_json:
            self.inference_file_dict[deployment] = self.choose_inference_files(deployment)

        # prep progress bar estimation
        n_images = 0
        n_inference_images = 0
        for deployment in os.listdir(self.deployments_dir):
            if not os.path.isdir(os.path.join(self.main_window.deployments_dir, deployment)):
                continue
            n_images += self.n_files_in_deployment(deployment)
            n_inference_images += self.n_files_in_deployment(deployment, inference_only=True)
        self.total_relative_time_estimated = START_TIME + ZOEDEPTH_BUILD_TIME + n_images * (MEGADETECTOR_TIME_PER_IMAGE + SEGMENTATION_TIME_PER_IMAGE) + n_inference_images * ZOEDEPTH_INFER_TIME_PER_IMAGE
        self.progress = 0   # 0-100

        self.increment_progress(START_TIME)



        # check that all images are the same dimensions in each deployment
        self.signals.message.emit("Checking image dimensions")
        for deployment in self.calibration_json:
            image_name = None
            image_width = None
            image_height = None
            for file in os.listdir(os.path.join(self.deployments_dir, deployment)):
                ext = os.path.splitext(file)[1].lower()
                if not (ext == ".jpg" or ext == ".jpeg" or ext == ".png"):
                    continue
                with Image.open(os.path.join(self.deployments_dir, deployment, file)) as image:
                    if not image_width:
                        image_name = file
                        image_width = image.size[0]
                        image_height = image.size[1]
                    elif image.size[0] != image_width or image.size[1] != image_height:
                        self.signals.warning_popup.emit("Image Sizes Do Not Match", f"The following images in deployment '{deployment}' have different sizes:\n\n{image_name}: {image_width} x {image_height}\n{file}: {image.size[0]} x {image.size[1]}\n\nAll images in a deployment must have the same dimensions for automatic cropping and segmentation to work properly. Please correct this and then click 'Run Distance Estimation' again.")
                        self.signals.message.emit(f"Images in deployment '{deployment}' not all the same size, please fix and try running again.")
                        self.signals.stopped.emit()
                        return



        # run megadetector (do all of this before zoedepth to avoid weird conflicts when building the zoe model)

        detections_dir = os.path.join(self.root_path, "detections")
        os.makedirs(detections_dir, exist_ok=True)
        
        for deployment in self.calibration_json:
            self.signals.message.emit(f"Locating animal bounding boxes - {deployment}")
            print(deployment)
            
            output_file = os.path.join(detections_dir, deployment + ".json")
            if not os.path.exists(output_file):
                input_dir = os.path.join(self.deployments_dir, deployment)
                args = ["megadetector_weights/md_v5a.0.0.pt", input_dir, output_file, "--threshold", "0.5"]
                run_detector_batch.main(args)

                # bboxes were evaluated on uncropped image, adjust for cropped image
                self.main_window.crop_manager.crop_megadetector_bboxes(output_file, deployment)

                if self.stop_requested:
                    self.signals.message.emit("Stopped")
                    self.signals.stopped.emit()
                    return
            
            self.increment_progress(self.n_files_in_deployment(deployment) * MEGADETECTOR_TIME_PER_IMAGE)
            print("Megadetector done with deployment", deployment)
        
        print("Megadetector done")
        



        # do the rest of the pipeline after megadetector
        

        # if zoedepth was already built, reflect that in the progress bar
        if self.main_window.zoedepth_model:
            self.increment_progress(ZOEDEPTH_BUILD_TIME)

        for deployment in self.calibration_json:
            inference_files = self.inference_file_dict[deployment]

            input_dir = os.path.join(self.deployments_dir, deployment)
            depth_maps_dir = os.path.join(self.root_path, "depth_maps", deployment)
            os.makedirs(depth_maps_dir, exist_ok=True)

            detection_json_path = os.path.join(detections_dir, deployment + ".json")
            if not os.path.exists(detection_json_path):
                continue
            with open(detection_json_path) as json_file:
                bbox_json = json.load(json_file)

            # get calibrated reference depth for this deployment
            calib_depth = self.get_calib_depth(deployment)

            
            # segmentation
            self.signals.message.emit(f"{deployment} - running segmentation")
            segmentation_filepath_dict = self.run_segmentation(deployment, resize_factor=SEGMENTATION_RESIZE_FACTOR, inference_files=inference_files)
            self.increment_progress(self.n_files_in_deployment(deployment) * SEGMENTATION_TIME_PER_IMAGE)
            if self.stop_requested:
                self.signals.message.emit("Stopped")
                self.signals.stopped.emit()
                return
            
            

            # run zoedepth

            for i, image_abs_path in enumerate(inference_files):
                self.increment_progress(ZOEDEPTH_INFER_TIME_PER_IMAGE)  # do this at the beginning (instead of end) for accurate accounting, since there are so many spots with the continue keyword that skip the end

                if self.stop_requested:
                    self.signals.message.emit("Stopped")
                    self.signals.stopped.emit()
                    return
                
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
                        self.signals.message.emit("Building depth model")
                        self.main_window.zoedepth_model = build_zoedepth_model()
                        self.increment_progress(ZOEDEPTH_BUILD_TIME)
                        if self.stop_requested:
                            self.signals.message.emit("Stopped")
                            self.signals.stopped.emit()
                            return

                    self.signals.message.emit(f"{deployment} - calculating depth for image {i+1}/{len(inference_files)}")

                    print("Running zoedepth on", image_abs_path)
                    with Image.open(image_abs_path).convert("RGB") as image:
                        cropped_image = self.main_window.crop_manager.crop(image, deployment)
                        depth = self.main_window.zoedepth_model.infer_pil(cropped_image)
                    save_basename = os.path.splitext(os.path.basename(image_abs_path))[0] + "_raw.png"
                    save_path = os.path.join(depth_maps_dir, save_basename)
                    save_raw_16bit(depth, save_path)
                
                self.signals.message.emit(f"{deployment} - calculating depth for image {i+1}/{len(inference_files)}")
                
                # calibrate
                # lazy calibration
                norm_depth = (depth - np.mean(depth)) / np.std(depth)
                depth = np.maximum(0, norm_depth * np.std(calib_depth) + np.mean(calib_depth))

                # extract animal depths and save
                # we could do this after processing all the files from this deployment, but one file at a time lets you see the results popping up in real time :)

                output = []
                animal_mask_union = np.zeros(depth.shape).astype(bool)   # union of segmentation masks inside bounding boxes, used for visualization

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

                    # get bbox segmentation mask
                    with Image.open(segmentation_filepath_dict[image_abs_path]) as mask_img:
                        mask_img = mask_img.resize((mask_img.width * SEGMENTATION_RESIZE_FACTOR, mask_img.height * SEGMENTATION_RESIZE_FACTOR))
                        animal_mask_data = np.asarray(mask_img)
                        
                        # animal mask data might not be the exact same shape as our depth image, since we were doing resizing
                        # fix that by cropping the animal mask data to the depth map shape (fixing if it was too big)
                        # and then pasting it onto a numpy array of zeros of the depth map shape (fixing if it was too small)
                        animal_mask = np.zeros(depth.shape)
                        animal_mask_data = animal_mask_data[0:depth.shape[0], 0:depth.shape[1]]
                        animal_mask[0:animal_mask_data.shape[0], 0:animal_mask_data.shape[1]] = animal_mask_data

                        bbox_animal_mask = animal_mask[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
                        bbox_animal_mask = bbox_animal_mask.astype(bool)
                        animal_mask_union[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = bbox_animal_mask
                    
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
                        "animal_distance": depth_estimate,
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
                try:
                    with open(output_filepath, 'w', newline='') as csvfile:
                        fieldnames = list(output[0].keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in self.main_window.csv_output_rows:
                            writer.writerow(row)
                except:
                    self.signals.warning_popup.emit("Failed to Write Output", f"Failed to write to the output file:\n{output_filepath}\n\nThis may be because the file is open in another program. Please close other programs accessing the file and then click 'Run Depth Estimation' again.")
                    self.signals.message.emit("Failed to write output, please try running again.")
                    self.signals.stopped.emit()
                    return
                
                
                # create output visualization

                visualization_dir = os.path.join(self.root_path, "output_visualization", deployment)
                os.makedirs(visualization_dir, exist_ok=True)

                # label RGB image, once without segmentation mask and once with it
                with Image.open(image_abs_path) as rgb_image:
                    cropped_rgb_image = self.main_window.crop_manager.crop(rgb_image, deployment)
                    cropped_rgb_image_segmentation = cropped_rgb_image.copy()

                    # normal
                    for row in output:
                        self.draw_annotations(cropped_rgb_image, row)
                    rgb_output_file = os.path.join(visualization_dir, os.path.basename(image_abs_path))
                    cropped_rgb_image.save(rgb_output_file)

                    # with segmentation mask
                    animal_segmentation_mask_image = Image.fromarray(animal_mask_union * 255).convert("L")
                    cropped_rgb_image_segmentation.paste("yellow", (0,0), animal_segmentation_mask_image)
                    for row in output:
                        self.draw_annotations(cropped_rgb_image_segmentation, row)
                    name, ext = os.path.splitext(os.path.basename(image_abs_path))
                    rgb_segmentation_output_file = os.path.join(visualization_dir, name + "a_segmentation" + ext)
                    cropped_rgb_image_segmentation.save(rgb_segmentation_output_file)
                
                # label depth image
                depth_image = Image.fromarray(colorize(depth)).convert("RGB")
                for row in output:
                    self.draw_annotations(depth_image, row)
                name, ext = os.path.splitext(os.path.basename(image_abs_path))
                depth_output_file = os.path.join(visualization_dir, name + "b_depth" + ext)
                depth_image.save(depth_output_file)
        

        # keep scrollbar updated correctly if we never needed to build the depth model
        if not self.main_window.zoedepth_model:
            self.increment_progress(ZOEDEPTH_BUILD_TIME)
        
       
        print("DONE!!!!!!!!!!!!!!!!!!!!")
        self.increment_progress(self.total_relative_time_estimated)    # finish to 100%, in case accounting was a bit off
        self.signals.message.emit("Done!")
        self.signals.done.emit()


    def n_files_in_deployment(self, deployment, inference_only=False):
        if inference_only:
            if deployment not in self.inference_file_dict:
                return 0
            if not os.path.isdir(os.path.join(self.deployments_dir, deployment)):
                return 0
            
            return len(self.inference_file_dict[deployment])
        return len(os.listdir(os.path.join(self.deployments_dir, deployment)))


    def increment_progress(self, relative_time_increment):
        self.progress += 100 * relative_time_increment / self.total_relative_time_estimated
        self.progress = min(100, self.progress)
        self.signals.progress.emit(self.progress)


    def choose_inference_files(self, deployment):
        # not all image files in a deployment are necessarily being used for inference
        # some are there just to help the segmentation work better
        # this function returns a list of absolute image paths that we want to do inference on
        inference_files = []
        input_dir = os.path.join(self.deployments_dir, deployment)
        for file in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, file)):
                continue
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


    def draw_annotations(self, image, row):
        draw = ImageDraw.Draw(image)
                
        top_left = (int(row['bbox_x']), int(row['bbox_y']))
        bottom_right = (top_left[0] + int(row['bbox_width']), top_left[1] + int(row['bbox_height']))
        draw.rectangle((top_left, bottom_right), outline="red", width=3)

        radius = 10
        sample_top_left = (int(row['sample_x'])-radius, int(row['sample_y'])-radius)
        sample_bottom_right = (int(row['sample_x'])+radius, int(row['sample_y'])+radius)
        draw.arc((sample_top_left, sample_bottom_right), 0, 360, fill="red", width=5)

        distance = round(float(row['animal_distance']), ndigits=1)
        text = f"{distance} m"
        
        if platform.system() == 'Darwin':       # macOS
            font = ImageFont.truetype("Arial.ttf", size=24)
        else:    # Windows, hopefully works on linux???
            font = ImageFont.truetype("arial.ttf", size=24)
        
        bbox = draw.textbbox(top_left, text, font=font)
        draw.rectangle(bbox, fill="black")
        draw.text(top_left, text, fill="white", font=font)
    


    def run_segmentation(self, deployment, resize_factor=4, inference_files=None):
        # returns a dictionary: key = image absolute path, value = segmentation mask absolute path
        # if inference files is specified, will only save final output for those files
        out_dictionary = {}

        input_dir = os.path.join(self.deployments_dir, deployment)
        output_dir = os.path.join(self.root_path, "segmentation", deployment)
        os.makedirs(output_dir, exist_ok=True)
        
        vectors = []
        filenames = []
        gray_image_shape = None

        # Load and preprocess images
        print("Segmentation on ", input_dir)
        print("Preprocessing images")

        for file in os.listdir(input_dir):
            if self.stop_requested:
                return
            fpath = os.path.join(input_dir, file)
            ext = os.path.splitext(file)[1].lower()
            if os.path.isdir(file) or not(ext == ".png" or ext == ".jpg" or ext == ".jpeg"):
                continue

            filenames.append(file)

            with Image.open(fpath) as image:
                image = self.main_window.crop_manager.crop(image, deployment)
                image = image.resize((image.width // resize_factor, image.height // resize_factor))

                preprocessed_image = type2_preprocess(image) if is_night(image) else type1_preprocess(image)
                
                gray_image_shape = preprocessed_image.shape # store so we know how to reshape back to an image later
                vector = np.reshape(preprocessed_image, (preprocessed_image.shape[0]*preprocessed_image.shape[1], 1))
                vectors.append(vector)
        

        # Prep data matrix and do RPCA

        print("Running RPCA")
        M = np.hstack(vectors)
        print(M.shape)
        rpca = R_pca(M)
        L, S = rpca.fit(max_iter=5, iter_print=1)
        print("RPCA done, saving segmentation masks")


        # post processing and save

        median_footprint = np.ones((3,3))
        morphological_footprint = disk(1)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(S.shape[1]):
            if self.stop_requested:
                return
            if inference_files and os.path.abspath(os.path.join(input_dir, filenames[i])) not in inference_files:
                continue

            save_filename = os.path.splitext(filenames[i])[0] + ".png"

            sparse_img = np.reshape(S[:,i], gray_image_shape)

            threshold = np.std(sparse_img, ddof=1)
            thresholded = (np.multiply(sparse_img, sparse_img) > threshold) * 255

            filtered = filters.median(thresholded, median_footprint)
            filtered = opening(filtered, morphological_footprint)

            binary = filtered.astype(bool)
            save_path = os.path.join(output_dir, save_filename)
            Image.fromarray(binary).save(save_path)
        
            orig_abs_path = os.path.abspath(os.path.join(input_dir, filenames[i]))
            save_abs_path = os.path.abspath(save_path)
            out_dictionary[orig_abs_path] = save_abs_path
        
        print("Segmentation done")
        return out_dictionary



# Segmentation utility functions

def is_night(pil_image):
    np_image = np.asarray(pil_image.convert("RGB"))
    for i in range(5):
        x = random.randint(0, np_image.shape[1]-1)
        y = random.randint(0, np_image.shape[0]-1)
        equal = np_image[y,x,0] == np_image[y,x,1] and np_image[y,x,1] == np_image[y,x,2]
        if not equal:
            return False
    return True


def type1_preprocess(pil_image, beta=0):
    gray_image = np.asarray(pil_image.convert("L"))
    equalized_image = equalize_hist(gray_image) * 255
    equalized_image = equalized_image.astype(int)

    if beta == 0:
        return equalized_image

    radius = 1
    n_points = 8*radius
    METHOD = "uniform"
    lbp_img = local_binary_pattern(equalized_image, n_points, radius, METHOD)

    i_star = beta*lbp_img + (1-beta)*equalized_image
    return i_star


def type2_preprocess(pil_image, beta=0.35):
    gray_image = np.asarray(pil_image.convert("L"))
    blurred_image = filters.gaussian(gray_image, sigma=0.5, preserve_range=True).astype(int)

    if beta == 0:
        return blurred_image

    radius = 2
    n_points = 8*radius
    METHOD = "uniform"
    lbp_img = local_binary_pattern(blurred_image, n_points, radius, METHOD)

    i_star = beta*lbp_img + (1-beta)*blurred_image
    return i_star