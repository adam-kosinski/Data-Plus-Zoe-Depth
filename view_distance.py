from PIL import Image
import numpy as np
import cv2
import argparse
import os
import json

def putDist(data, img, x, y, index=0, label=""):
    bgr_color=(0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = " " + str(round(data[y][x], 2)) + "m" + "  " + label
    cv2.rectangle(img, (x + 15, y - 10 + 30*index), (x + 140, y + 30 + 30*index), (255, 255, 255), -1)
    cv2.putText(img, text, (x,y + 20 + 30*index), font, 1, bgr_color, 2)
    cv2.circle(img, (x,y), 6, bgr_color, 2)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # display distance on the image window
        img_copy = colored_depth_map.copy()
        if args.all:
            for i, (filename, depths) in enumerate(all_data.items()):
                putDist(depths, img_copy, x, y, i, filename)
        else:
            putDist(data, img_copy, x, y)
        cv2.imshow('display', img_copy)




# get filename
parser = argparse.ArgumentParser()
parser.add_argument('raw_filename')
parser.add_argument('-f', '--colored_filename')
parser.add_argument('-c', '--calib_json_file')
parser.add_argument('-a', '--all', action='store_true')
parser.add_argument('-r', '--round', action='store_true')
parser.add_argument('-u', '--unscaled_depth', action='store_true')
args = parser.parse_args()

colored_filename = None
if args.colored_filename:
    colored_filename = args.colored_filename

print("colored_filename", colored_filename)

# load depth data
with Image.open(args.raw_filename) as img:
    data = np.asarray(img)

# process depth data
if not args.unscaled_depth:
    data = data / 256   # undo the scaling that was used when storing the image

# calibration
calib_depth = None
if args.calib_json_file and os.path.exists(args.calib_json_file):
    with open(args.calib_json_file) as json_file:
        calib_json = json.load(json_file)
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(args.raw_filename)))
    if parent_dir in calib_json:
        print("deployment", parent_dir)
        calib = calib_json[parent_dir]
        root_dir = os.path.join(os.path.dirname(args.calib_json_file), "..")
        rel_depth_path = os.path.join(root_dir, calib["rel_depth_path"])
        print("rel depth path", rel_depth_path)
        with Image.open(rel_depth_path) as calib_depth_img:
            calib_depth = np.asarray(calib_depth_img) / 256
        slope = calib["slope"]
        intercept = calib["intercept"]
        calib_depth = calib_depth * slope + intercept

        # lazy calibration
        norm_depth = (data - np.mean(data)) / np.std(data)
        data = np.maximum(0, norm_depth * np.std(calib_depth) + np.mean(calib_depth))

if args.round:
    data = np.round(data)
    
if args.all:
    all_data = {}
    dir = os.path.dirname(args.raw_filename)
    for file in os.listdir(dir):
        if not file.endswith("_raw.png"):
            continue
        with Image.open(dir + "/" + file) as img:
            all_data[file] = np.asarray(img) / 256


# display the color depth map
if colored_filename:
    colored_depth_map = cv2.imread(colored_filename, cv2.IMREAD_COLOR)
else:
    rescaled = (data - np.min(data)) / (np.max(data) - np.min(data))
    colored_depth_map = 1 - rescaled

cv2.imshow('display', colored_depth_map)

# click event handler
cv2.setMouseCallback('display', click_event)

# wait for a key to be pressed to exit
cv2.waitKey(0)
cv2.destroyAllWindows()