import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import math


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std



def get_bbox_mask(bbox_file, filename, data):
    if not bbox_file:
        return np.ma.nomask
    
    with open(bbox_file) as json_data:
        bboxs = json.load(json_data)
        for key in bboxs.keys():
            if key in filename:
                b = bboxs[key]
                bbox_mask = np.zeros(data.shape, dtype=bool)
                bbox_mask[b['y']:b['y']+b['height'], b['x']:b['x']+b['height']] = True
                break

    return bbox_mask



def show(data):
    plt.imshow(data, interpolation='nearest', cmap='binary')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


# returns average (or median), and also the cropped, normalized, masked image arrays
def get_average(files, bbox_file, median=False):
    print("Getting average for these files:")

    # open files and get data, apply bbox masks
    uncropped_data = []
    heights = []
    for file in files:
        if not file.endswith("_raw.png"):
            continue
        print(file)
        with Image.open(file) as raw_img:
            # raw_img = raw_img.resize((math.floor(raw_img.width * 8.192), math.floor(raw_img.height * 8.192)))
            img_data = np.asarray(raw_img)[10:,:] / 256
            masked = np.ma.masked_array(img_data, mask=get_bbox_mask(bbox_file, file, img_data))
            uncropped_data.append(masked)
            heights.append(masked.shape[0])

    # crop and normalize
    data = []
    for arr in uncropped_data:
        data.append(arr[:min(heights),:])
    norm = []
    for arr in data:
        norm.append(normalize(arr) * np.std(data[0]) + np.mean(data[0]))


    # do average
    if median:
        avg = np.ma.median(norm, axis=0)
        print("mode median")
    else:
        # avg = np.round(np.ma.min(norm, axis=0))
        avg = np.ma.average(norm, axis=0)
        print("mode average")
    
    return avg, norm

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_files", nargs='+')
    parser.add_argument("-m","--median", action='store_true')
    parser.add_argument("-b","--bbox_file")
    parser.add_argument("-s","--save", action='store_true')
    args = parser.parse_args()

    if os.path.isdir(args.raw_files[0]):
        dir = args.raw_files[0]
        files = os.listdir(dir)
        for i, file in enumerate(files):
            files[i] = dir + "/" + file
    else:
        dir = "."
        files = args.raw_files

    bbox_file = args.bbox_file if args.bbox_file else None

    avg, norm = get_average(files, bbox_file, args.median)

    show(np.round(avg))


    # plt.axis('off')
    # show(avg)

    if args.save:
        avg_scaled = 256 * avg
        fpath = "./average.png"
        cv2.imwrite(fpath, avg_scaled.astype("uint16"))
        print("Saved depth map to ./average.png")

