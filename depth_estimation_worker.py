import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json
import csv
import math
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage import filters
from skimage.morphology import disk, opening
from r_pca import R_pca
import random
import platform
from datetime import datetime, timedelta

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from zoe_worker import build_zoedepth_model
from zoedepth.utils.misc import save_raw_16bit, colorize

from megadetector_onnx import MegaDetectorRuntime


SEGMENTATION_RESIZE_FACTOR = 4

# measured times in seconds used for progress bar, treat as relative though in case computer faster etc
# megadetector load time is insignificant
MEGADETECTOR_TIME_PER_IMAGE = 0.9
SEGMENTATION_TIME_PER_IMAGE = 2
ZOEDEPTH_BUILD_TIME = 12
ZOEDEPTH_INFER_TIME_PER_IMAGE = 20

TYPICAL_FRACTION_OF_IMAGES_WITH_ANIMALS = 0.2   # used to estimate zoedepth runtime at the beginning before we know how many animal images


class DepthEstimationSignals(QObject):
    message = pyqtSignal(str)   # displays next to the progress bar
    progress = pyqtSignal(float)  # 0-100 for progress percentage
    warning_popup = pyqtSignal(str, str)    # title, message
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

        self.total_relative_time_estimated = 0  # initialized by summing config values - see run() function
        self.progress = 0   # 0-100

        # config
        self.MEGADECTOR_CONF_THRESHOLD = 0.5
        self.MEGADETECTOR_CHECKPOINT_FREQUENCY = 10 # save every 10 images
        self.MAX_SEC_BETWEEN_SET_IMAGES = 600   # for segmentation sets


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


        # prep progress bar estimation
        n_images = 0
        for deployment in os.listdir(self.deployments_dir):
            if not os.path.isdir(os.path.join(self.deployments_dir, deployment)):
                continue
            n_images += len(self.get_image_filepaths(deployment))
        n_animal_images_guess = n_images * TYPICAL_FRACTION_OF_IMAGES_WITH_ANIMALS
        self.total_relative_time_estimated = ZOEDEPTH_BUILD_TIME + n_images * MEGADETECTOR_TIME_PER_IMAGE + n_animal_images_guess * (SEGMENTATION_TIME_PER_IMAGE + ZOEDEPTH_INFER_TIME_PER_IMAGE)
        self.progress = 0   # 0-100


        # check that all images are the same dimensions in each deployment
        self.signals.message.emit("Checking image dimensions")
        for deployment in self.calibration_json:
            first_image_filepath = None
            first_image_width = None
            first_image_height = None

            image_filepaths = self.get_image_filepaths(deployment)
            for filepath in image_filepaths:
                with Image.open(filepath) as image:
                    if not first_image_width:
                        first_image_filepath = filepath
                        first_image_width = image.size[0]
                        first_image_height = image.size[1]
                    elif image.size[0] != first_image_width or image.size[1] != first_image_height:
                        self.signals.warning_popup.emit("Image Sizes Do Not Match", f"The following images in deployment '{deployment}' have different sizes:\n\n{first_image_filepath}: {first_image_width} x {first_image_height}\n{filepath}: {image.size[0]} x {image.size[1]}\n\nAll images in a deployment must have the same dimensions for automatic cropping and segmentation to work properly. Please correct this and then click 'Run Distance Estimation' again.")
                        self.signals.message.emit(f"Images in deployment '{deployment}' not all the same size, please fix and try running again.")
                        self.signals.stopped.emit()
                        return




        detections_dir = os.path.join(self.root_path, "detections")
        os.makedirs(detections_dir, exist_ok=True)
  

        # if zoedepth was already built, reflect that in the progress bar
        if self.main_window.zoedepth_model:
            self.increment_progress(ZOEDEPTH_BUILD_TIME)

        self.signals.message.emit("Loading MegaDetector v5a model")
        megadetector_runtime = MegaDetectorRuntime()

        # run the pipeline for each calibrated deployment

        for deployment in self.calibration_json:
            image_filepaths = self.get_image_filepaths(deployment)


            # MEGADETECTOR (ONNX version for faster inference)

            # init json
            detection_json_path = os.path.join(detections_dir, deployment + ".json")
            if os.path.exists(detection_json_path):
                with open(detection_json_path) as json_file:
                    bbox_json = json.load(json_file)
            else:
                bbox_json = {"images": []}

            # init detections_dict (stores positive detections only, and can lookup by filename unlike megadetector json format)
            detections_dict = {}
            for image_data in bbox_json["images"]:
                # check if image still exists, just in case
                if not os.path.exists(os.path.join(self.deployments_dir, deployment, image_data["file"])):
                    continue
                # filter just in case, category 1 is for animals
                detections = list(filter(lambda detection: detection["category"] == "1" and
                                         detection["conf"] >= self.MEGADECTOR_CONF_THRESHOLD,
                                         image_data["detections"]))
                if len(detections) > 0:
                    detections_dict[image_data["file"]] = detections

            # run megadetector on images we haven't processed yet
            for i, filepath in enumerate(image_filepaths):
                self.signals.message.emit(f"{deployment} - Locating animal bounding boxes for image {i+1}/{len(image_filepaths)}")
                self.increment_progress(MEGADETECTOR_TIME_PER_IMAGE) # do this at the top so it's consistent regardless of continue statements

                if self.stop_requested:
                    self.signals.message.emit("Stopped")
                    self.signals.stopped.emit()
                    return
                
                # checkpointing
                if i % self.MEGADETECTOR_CHECKPOINT_FREQUENCY == 0:
                    with open(detection_json_path, mode='w') as json_file:
                        json.dump(bbox_json, json_file, indent=1)
                
                # skip images we've done already
                if os.path.basename(filepath) in map(lambda obj: obj["file"], bbox_json["images"]):
                    continue

                md_result = megadetector_runtime.run(filepath, conf_threshold=self.MEGADECTOR_CONF_THRESHOLD, categories=["1"])
                
                bbox_json["images"].append(md_result)
                
                if len(md_result["detections"]) > 0:
                    detections_dict[md_result["file"]] = md_result["detections"]

            # final save
            with open(detection_json_path, mode='w') as json_file:
                json.dump(bbox_json, json_file, indent=1)
            
            

            # to prepare for segmentation, organize images into sets, the segmentation algorithm will run on one of these sets at a time
            # (using two dicts as indices to convert back and forth between set id and file basenames)
            # see self.get_segmentation_sets() for more details
            basename_to_set, set_to_basenames = self.get_segmentation_sets(deployment)

            # deployment-level depth prep
            depth_maps_dir = os.path.join(self.root_path, "depth_maps", deployment)
            os.makedirs(depth_maps_dir, exist_ok=True)
            calib_depth = self.get_calib_depth(deployment)

            

            # progress bar stuff - pretend that we got our estimates right before, to make sure time adds up
            orig_estimated_zoedepth_time = len(image_filepaths) * TYPICAL_FRACTION_OF_IMAGES_WITH_ANIMALS * ZOEDEPTH_INFER_TIME_PER_IMAGE
            orig_estimated_segmentation_time = len(image_filepaths) * TYPICAL_FRACTION_OF_IMAGES_WITH_ANIMALS * SEGMENTATION_TIME_PER_IMAGE
            n_images_with_animals = len(detections_dict.keys())
            adjusted_zoedepth_time_per_animal_image = orig_estimated_zoedepth_time / n_images_with_animals
            adjusted_segmentation_time_per_animal_image = orig_estimated_segmentation_time / n_images_with_animals

            # iterate through files with animals detected

            for i, (image_basename, detections) in enumerate(detections_dict.items()):
                

                if self.stop_requested:
                    self.signals.message.emit("Stopped")
                    self.signals.stopped.emit()
                    return
                


                # segmentation

                # check if the mask exists already, if not do segmentation on its set
                mask_file_basename = os.path.splitext(image_basename)[0] + ".png"
                mask_file_path = os.path.join(self.root_path, "segmentation", deployment, mask_file_basename)

                if not os.path.exists(mask_file_path) and image_basename in basename_to_set:
                    print("Running segmentation on", image_basename)
                    self.signals.message.emit(f"{deployment} - running segmentation on image {i+1}/{len(detections_dict.keys())}")

                    set_id = basename_to_set[image_basename]
                    set_files = set_to_basenames[set_id]
                    files_with_animals = list(filter(lambda basename: basename in detections_dict, set_files))
                    self.run_segmentation(set_files, deployment, resize_factor=SEGMENTATION_RESIZE_FACTOR, files_to_save=files_with_animals)
                
                self.increment_progress(adjusted_segmentation_time_per_animal_image)



                
                print("Getting depth for", image_basename)
                self.signals.message.emit(f"{deployment} - calculating depth for image {i+1}/{len(detections_dict.keys())}")    # treat everything after this point (except model building) as belonging to this message, less flickering of UI messages

                # check if depth was already calculated, if so use it and skip zoedepth calculation
                depth_basename = os.path.splitext(image_basename)[0] + "_raw.png"
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
                        self.signals.message.emit(f"{deployment} - calculating depth for image {i+1}/{len(detections_dict.keys())}")

                    print("Running zoedepth on", image_basename)

                    with Image.open(os.path.join(self.deployments_dir, deployment, image_basename)).convert("RGB") as image:
                        cropped_image = self.main_window.crop_manager.crop(image, deployment)
                        depth = self.main_window.zoedepth_model.infer_pil(cropped_image)
                    save_basename = os.path.splitext(image_basename)[0] + "_raw.png"
                    save_path = os.path.join(depth_maps_dir, save_basename)
                    save_raw_16bit(depth, save_path)
                
                
                # calibrate
                # lazy calibration
                norm_depth = (depth - np.mean(depth)) / np.std(depth)
                depth = np.maximum(0, norm_depth * np.std(calib_depth) + np.mean(calib_depth))


                self.increment_progress(adjusted_zoedepth_time_per_animal_image)

                

                


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


                    # sample from segmentation mask if it exists
                    segmentation_median_estimate = None
                    if os.path.exists(mask_file_path):
                        with Image.open(mask_file_path) as mask_img:
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
                        animal_mask_union[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = bbox_animal_mask | animal_mask_union[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

                        segmentation_median_estimate = np.median(bbox_depth[bbox_animal_mask]) if True in bbox_animal_mask else None


                    # get depth estimate
                    if segmentation_median_estimate:
                        depth_estimate = segmentation_median_estimate
                    else:
                        depth_estimate = np.percentile(bbox_depth, 20)


                    # get sampled point for visualization (point with depth value = depth estimate, that's closest to bbox center)
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
                        "filename": image_basename,
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
                with Image.open(os.path.join(self.deployments_dir, deployment, image_basename)) as rgb_image:
                    cropped_rgb_image = self.main_window.crop_manager.crop(rgb_image, deployment)
                    cropped_rgb_image_segmentation = cropped_rgb_image.copy()

                    # normal
                    for row in output:
                        self.draw_annotations(cropped_rgb_image, row)
                    rgb_output_file = os.path.join(visualization_dir, image_basename)
                    cropped_rgb_image.save(rgb_output_file)

                    # with segmentation mask
                    animal_segmentation_mask_image = Image.fromarray(animal_mask_union * 255).convert("L")
                    cropped_rgb_image_segmentation.paste("yellow", (0,0), animal_segmentation_mask_image)
                    for row in output:
                        self.draw_annotations(cropped_rgb_image_segmentation, row)
                    name, ext = os.path.splitext(image_basename)
                    rgb_segmentation_output_file = os.path.join(visualization_dir, name + "a_segmentation" + ext)
                    cropped_rgb_image_segmentation.save(rgb_segmentation_output_file)
                
                # label depth image
                depth_image = Image.fromarray(colorize(depth)).convert("RGB")
                for row in output:
                    self.draw_annotations(depth_image, row)
                name, ext = os.path.splitext(image_basename)
                depth_output_file = os.path.join(visualization_dir, name + "b_depth" + ext)
                depth_image.save(depth_output_file)
        

        # keep scrollbar updated correctly if we never needed to build the depth model
        if not self.main_window.zoedepth_model:
            self.increment_progress(ZOEDEPTH_BUILD_TIME)
        
       
        print("DONE!!!!!!!!!!!!!!!!!!!!")
        self.increment_progress(self.total_relative_time_estimated)    # finish to 100%, in case accounting was a bit off
        self.signals.message.emit("Done!")
        self.signals.done.emit()
    

    def get_image_filepaths(self, deployment):
        # useful for filtering for images only, and counting # images
        image_filepaths = []
        for file in os.listdir(os.path.join(self.deployments_dir, deployment)):
            ext = os.path.splitext(file)[1].lower()
            if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                filepath = os.path.join(self.deployments_dir, deployment, file)
                image_filepaths.append(filepath)
        return image_filepaths


    def increment_progress(self, relative_time_increment):
        self.progress += 100 * relative_time_increment / self.total_relative_time_estimated
        self.progress = min(100, self.progress)
        self.signals.progress.emit(self.progress)


    def get_segmentation_sets(self, deployment):
        # RPCA segmentation runs best on sets of images where for each image there are at least several others with a very similar background
        # we cannot run RPCA on the whole deployment at once because that would take way too much memory
        # so, organize deployment images into sets, where each set has a similar background
        # images with less time than self.MAX_SEC_BETWEEN_SET_IMAGES between when they were taken go to the same set, because they're from the same burst / a close one
        # images without JPG date-time-original metadata get sorted into sets based on whether they are a day or night image

        # create two indices to convert back and forth between set ID and image filenames
        basename_to_set = {}
        set_to_basenames = {}

        basename_date_tuples = []
        basenames_without_metadata = []
        directory = os.path.join(self.deployments_dir, deployment)
        
        # process images with metadata first

        for filepath in self.get_image_filepaths(deployment):
            basename = os.path.basename(filepath)
            # only jpg images have date time original info
            if os.path.splitext(basename)[1].lower() != ".jpg":
                basenames_without_metadata.append(basename)
                continue
            with Image.open(os.path.join(directory, basename)) as image:
                exif = image._getexif()
                if not exif or 36867 not in exif:
                    basenames_without_metadata.append(basename)
                    continue
                datestring = exif[36867] # "date time original" tag id
                date = datetime.strptime(datestring, "%Y:%m:%d %H:%M:%S")
                basename_date_tuples.append((basename, date))
        
        basename_date_tuples.sort(key=lambda item: item[1])
        
        current_set_id = 0
        current_set_files = []
        for i, item in enumerate(basename_date_tuples):
            basename_to_set[item[0]] = current_set_id
            current_set_files.append(item[0])

            # if last item, or enough time between this one and the next, save this set and start a new one
            if i+1 == len(basename_date_tuples) or basename_date_tuples[i+1][1] - item[1] > timedelta(seconds=self.MAX_SEC_BETWEEN_SET_IMAGES):
                set_to_basenames[current_set_id] = current_set_files
                current_set_files = []
                current_set_id += 1

        # process images without metadata, sorting by day/night

        # sort into day/night
        day_basenames_without_metadata = []
        night_basenames_without_metadata = []

        for basename in basenames_without_metadata:
            filepath = os.path.join(directory, basename)
            with Image.open(filepath) as pil_image:
                if is_night(pil_image):
                    night_basenames_without_metadata.append(basename)
                else:
                    day_basenames_without_metadata.append(basename)

        target_set_size = 50    # aim on the big side because these images could vary a lot

        # day sets
        if len(day_basenames_without_metadata) > 0:
            # divide up roughly equally, trying to get close to the target set size
            n_day_sets = math.ceil(len(day_basenames_without_metadata) / target_set_size)   # round up so we don't get zero sets
            day_set_size = math.ceil(len(day_basenames_without_metadata) / n_day_sets)  # round up so that the last set will just be slightly small, instead of having a tiny remainder set
            for i in range(n_day_sets):
                this_set = day_basenames_without_metadata[i*day_set_size:(i+1)*day_set_size]
                set_to_basenames[current_set_id] = this_set
                for basename in this_set:
                    basename_to_set[basename] = current_set_id
                current_set_id += 1
        
        # night sets
        if len(night_basenames_without_metadata) > 0:
            n_night_sets = math.ceil(len(night_basenames_without_metadata) / target_set_size)
            night_set_size = math.ceil(len(night_basenames_without_metadata) / n_night_sets)
            for i in range(n_night_sets):
                this_set = night_basenames_without_metadata[i*night_set_size:(i+1)*night_set_size]
                set_to_basenames[current_set_id] = this_set
                for basename in this_set:
                    basename_to_set[basename] = current_set_id
                current_set_id += 1
        
        return basename_to_set, set_to_basenames


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
    


    def run_segmentation(self, file_basename_list, deployment, resize_factor=4, files_to_save=None):
        # resize_factor: shrink images by a factor of resize_factor
        # files_to_save: if specified, only save these files' segmentation masks (default is to save all of them)

        output_dir = os.path.join(self.root_path, "segmentation", deployment)
        os.makedirs(output_dir, exist_ok=True)
        
        vectors = []
        gray_image_shape = None

        # Load and preprocess images
        print("Preprocessing images")

        for basename in file_basename_list:
            if self.stop_requested:
                return

            with Image.open(os.path.join(self.deployments_dir, deployment, basename)) as image:
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
            if files_to_save and file_basename_list[i] not in files_to_save:
                continue

            sparse_img = np.reshape(S[:,i], gray_image_shape)

            threshold = np.std(sparse_img, ddof=1)
            thresholded = (np.multiply(sparse_img, sparse_img) > threshold) * 255

            filtered = filters.median(thresholded, median_footprint)
            filtered = opening(filtered, morphological_footprint)

            binary = filtered.astype(bool)
            save_basename = os.path.splitext(file_basename_list[i])[0] + ".png"
            save_path = os.path.join(output_dir, save_basename)
            Image.fromarray(binary).save(save_path)
        
        print("Segmentation done")




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