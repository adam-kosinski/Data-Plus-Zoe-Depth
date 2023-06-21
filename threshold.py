import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("raw")
parser.add_argument("--min")
parser.add_argument("--max")
parser.add_argument("-d", "--depth")
parser.add_argument("-m", "--margin")
args = parser.parse_args()

calib_filepath = "./third_results/RCNX0367_raw.png"

def show_big():
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


with Image.open(args.raw) as raw_img:
    data = np.asarray(raw_img) / 256
with Image.open(calib_filepath) as calib_img:
    calib_data = np.asarray(calib_img) / 256
min_height = min(data.shape[0], calib_data.shape[0])
data = data[:min_height,:]
calib_data = calib_data[:min_height,:]
calib_data = normalize(calib_data) * np.std(data) + np.mean(data)

with open("./bboxs.json") as json_data:
    bboxs = json.load(json_data)

bbox_mask = np.zeros(data.shape, dtype=bool)
bbox = bboxs["RCNX0367"]
bbox_mask[bbox['y']:bbox['y']+bbox['height'], bbox['x']:bbox['x']+bbox['height']] = True


if args.min:
    low = float(args.min)
else:
    low = np.min(data)

if args.max:
    high = float(args.max)
else:
    high = np.max(data)

if args.depth:
    mid = float(args.depth)
    margin = float(args.margin) if args.margin else 0.5
    low = mid - margin
    high = mid + margin

print(low, high)

threshold_mask = (data < low) | (data > high)
dist_range = np.ma.masked_array(np.zeros(data.shape), mask=threshold_mask)

print(np.average(np.ma.masked_array(data, mask=threshold_mask)))

plt.imshow(data*(2**15), interpolation='nearest', cmap='binary')
plt.imshow(dist_range, interpolation='nearest', cmap='autumn', alpha=0.2)
show_big()

no_bbox_mask = threshold_mask | bbox_mask
dist_range_no_bbox = np.ma.masked_array(np.zeros(data.shape), mask=no_bbox_mask)

print(np.average(np.ma.masked_array(calib_data, mask=no_bbox_mask)))

plt.imshow(calib_data*(2**15), interpolation='nearest', cmap='binary')
plt.imshow(dist_range_no_bbox, interpolation='nearest', cmap='autumn', alpha=0.2)
show_big()