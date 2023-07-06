import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
from zoedepth.utils.misc import colorize
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

import sys
import run_detector_batch

import matplotlib.pyplot as plt

# get input dir
parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("calibration_file")
parser.add_argument("-d", "--depth_dir")
parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default=None, help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
args = parser.parse_args()


os.makedirs("./output", exist_ok=True)


# Load calibration

with open(args.calibration_file) as calib_file:
    calib_json = json.load(calib_file)
if args.input_dir not in calib_json:
    print("Couldn't find calibration for " + args.input_dir)
    sys.exit()


# MegaDetector

print("\nRunning MegaDetector\n")
if os.path.exists("./output/detections.json"):
    print("Found detections.json file, using that instead of recalculating")
else:
    base_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(base_dir, "MegaDetector"))
    sys.path.append(os.path.join(base_dir, "yolov5"))
    arg_string = f"megadetector_weights/md_v5a.0.0.pt {args.input_dir} output/detections.json --recursive --threshold 0.2"
    run_detector_batch.main(arg_string)

with open("./output/detections.json") as json_data:
    bbox_json = json.load(json_data)



print("\nRunning ZoeDepth\n")

# Load Zoe_NK
if args.pretrained_resource:
    overwrite = {"pretrained_resource": args.pretrained_resource}
    config = get_config("zoedepth_nk", "infer", None, **overwrite)
    model_zoe_nk = build_model(config)
else:
    model_zoe_nk = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True, config_mode="eval")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)


# Do depth inference

for file in os.listdir(args.input_dir):
    if not os.path.isfile(args.input_dir + "/" + file):
        continue

    print("\nDoing inference on " + file)

    with Image.open(args.input_dir + "/" + file).convert("RGB") as img:

        # get bounding box of animal
        bboxs = None
        for img_data in bbox_json['images']:
            if file != os.path.basename(img_data['file']):
                continue
            for detection in img_data['detections']:
                if detection['conf'] > 0.2:
                    print("conf", detection['conf'])
                    if not bboxs:
                        bboxs = []
                    bboxs.append(detection['bbox'])
            print("bboxs", bboxs)
            break
        if not bboxs:
            print("Couldn't find megadetector detections")
            continue

        
        print("Processing whole image")
        if args.depth_dir:
            for f in os.listdir(args.depth_dir):
                if not os.path.splitext(file)[0] + "_raw.png" == os.path.basename(f):
                    continue
                with Image.open(os.path.join(args.depth_dir, f)) as depth_img:
                    depth = np.asarray(depth_img) / 256
                break
        else:
            depth = zoe.infer_pil(img)  # as numpy
        
        # calibrate depth
        # TODO exclude bounding box regions
        calib_obj = calib_json[args.input_dir]
        with Image.open(calib_obj["relative_depth_map"]) as ref_img:
            ref_data = np.asarray(ref_img) / 256
            depth = (depth - np.mean(depth)) / np.std(depth)
            depth = depth * np.std(ref_data) + np.mean(ref_data)
            depth = depth * calib_obj["slope"] + calib_obj["intercept"]
        

        print("Processing animal area")
        
        for i in range(len(bboxs)):
            print(f"Animal {i}")

            # resize bounding box
            w, h = img.size
            b = {
                'x': int(bboxs[i][0] * w),
                'y': int(bboxs[i][1] * h),
                'width': int(bboxs[i][2] * w),
                'height': int(bboxs[i][3] * h)
            }
            padding = 150
            padded_box = (
                max(0, b['x'] - padding),
                max(0, b['y'] - padding),
                min(w, b['x'] + b['width'] + padding),
                min(h, b['y'] + b['height'] + padding)
            )
            local_img = img.crop(padded_box)

            # do inference
            local_depth = zoe.infer_pil(local_img)

            # get copy of global depth
            depth_copy = depth.copy()

            # align the depths using the padding area
            local_mask = np.zeros(local_depth.shape, dtype='bool')
            b_offset = {
                'top': b['y'] - padded_box[1],
                'bottom': b['y'] + b['height'] - padded_box[1],
                'left': b['x'] - padded_box[0],
                'right': b['x'] + b['width'] - padded_box[0]
            }
            local_mask[b_offset['top']:b_offset['bottom'], b_offset['left']:b_offset['right']] = True
            global_mask = np.ones(depth.shape, dtype='bool')
            global_mask[padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]] = local_mask
            L = local_depth[~local_mask]
            D = depth_copy[~global_mask]

            local_depth = ((local_depth - L.mean()) / L.std()) * D.std() + D.mean()
            local_depth = np.maximum(local_depth, 0)

            # paste local depth onto bigger depth map
            depth_copy[padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]] = local_depth

            # save
            save_filename_base = "./output/" + os.path.splitext(file)[0] + ("" if i==0 else f"_{i}")

            # save raw
            from zoedepth.utils.misc import save_raw_16bit
            fpath = save_filename_base + "_raw.png"
            save_raw_16bit(depth_copy, fpath)

            # save colored output
            colored = colorize(depth_copy)
            fpath_colored = save_filename_base + "_colored.png"
            Image.fromarray(colored).save(fpath_colored)

print("\nDone")