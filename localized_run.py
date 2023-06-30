import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
from zoedepth.utils.misc import colorize
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

import matplotlib.pyplot as plt

# get input dir
parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("bbox_file")
parser.add_argument("-d", "--depth_dir")
parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default=None, help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
args = parser.parse_args()

with open(args.bbox_file) as json_data:
    bbox_json = json.load(json_data)

# Zoe_NK
if args.pretrained_resource:
    overwrite = {"pretrained_resource": args.pretrained_resource}
    config = get_config("zoedepth_nk", "infer", None, **overwrite)
    model_zoe_nk = build_model(config)
else:
    model_zoe_nk = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True, config_mode="eval")


##### prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)


for file in os.listdir(args.input_dir):
    if not os.path.isfile(args.input_dir + "/" + file):
        continue

    print("\nDoing inference on " + file)

    with Image.open(args.input_dir + "/" + file).convert("RGB") as img:

        # get bounding box of animal
        bboxs = None
        for img_data in bbox_json['images']:
            if file != img_data['file']:
                continue
            for detection in img_data['detections']:
                if detection['conf'] > 0.2:
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
                if not os.path.splitext(file)[0] + "_raw" in f:
                    continue
                with Image.open(os.path.join(args.depth_dir, f)) as depth_img:
                    depth = np.asarray(depth_img) / 256
                break
        else:
            depth = zoe.infer_pil(img)  # as numpy
        

        print("Processing animal area")
        # TODO support multiple animals (don't just use index 0 of bboxs)
        # TODO don't assume padding not cut off when doing mask
        

        # resize bounding box
        w, h = img.size
        b = {
            'x': int(bboxs[0][0] * w),
            'y': int(bboxs[0][1] * h),
            'width': int(bboxs[0][2] * w),
            'height': int(bboxs[0][3] * h)
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

        # align the depths using the padding area
        local_mask = np.zeros(local_depth.shape, dtype='bool')
        local_mask[padding:padding+b['height'], padding:padding+b['width']] = True
        global_mask = np.ones(depth.shape, dtype='bool')
        global_mask[padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]] = local_mask
        L = local_depth[~local_mask]
        D = depth[~global_mask]

        local_depth = ((local_depth - L.mean()) / L.std()) * D.std() + D.mean()
        local_depth = np.maximum(local_depth, 0)

        # paste local depth onto bigger depth map
        depth[padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]] = local_depth


        print("inference done, saving")

        # create output dir if necessary
        if not os.path.exists("./output"):
            os.makedirs("./output")

        # save raw
        from zoedepth.utils.misc import save_raw_16bit
        fpath = "./output/" + os.path.splitext(file)[0] + "_raw.png"
        save_raw_16bit(depth, fpath)

        # save colored output
        colored = colorize(depth)
        fpath_colored = "./output/" + os.path.splitext(file)[0] + "_colored.png"
        Image.fromarray(colored).save(fpath_colored)

print("Done")