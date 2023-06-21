import numpy as np
import argparse
import os
from PIL import Image
from zoedepth.utils.misc import colorize
import cv2
import matplotlib.pyplot as plt


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("--align_to")
parser.add_argument("-n", "--normalize", action="store_true")
parser.add_argument("-s", "--show_orig", action="store_true")
parser.add_argument("--bboxes")
args = parser.parse_args()

assert os.path.exists(args.directory)
assert os.listdir(args.directory) != []
if args.align_to:
    assert os.path.exists(args.align_to)
    align_to = args.align_to
else:
    align_to = args.directory + "/" + os.listdir(args.directory)[0]
if args.bboxes:
    assert os.path.exists(args.align_to)

print("Aligning to " + align_to)

if not os.path.exists("./aligned_output"):
    os.makedirs("./aligned_output")


with Image.open(align_to) as align_img:
    align_data_orig = np.asarray(align_img)
    if args.normalize:
        align_data_orig = normalize(align_data_orig)
    align_height = align_data_orig.shape[0]

    for file in os.listdir(args.directory):
        if not file.endswith("_raw.png"):
            continue
        if align_to.endswith(file):
            continue

        print(file)

        with Image.open(args.directory + "/" + file) as img:
            data_orig = np.asarray(img)
            if args.normalize:
                data_orig = normalize(data_orig)
            data_height = data_orig.shape[0]

            crop_height = min(data_height, align_height)
            data = data_orig[0:crop_height,:].flatten() / 256
            align_data = align_data_orig[0:crop_height,:].flatten() / 256

            A = np.ones((data.size, 2))
            A[:,0] = data
            slope, intercept = np.linalg.lstsq(A, align_data, rcond=None)[0]
            print("m = ", slope)
            print("b = ", intercept)
            out = slope * data_orig + intercept

            if args.normalize or args.show_orig:
                out_values = data
            else:
                out_values = slope * data + intercept
            
            plt.scatter(out_values[::231], align_data[::231])
            low = np.min(out_values)
            high = np.max(out_values)
            plt.plot([low,high],[low,high], color="red")
            plt.show()

            continue

            fpath = "./aligned_output/" + file.split("_")[0] + "_raw.png"
            cv2.imwrite(fpath, out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])

            colored = colorize(out)
            fpath_colored = "./aligned_output/" + file.split("_")[0] + "_colored.png"
            Image.fromarray(colored).save(fpath_colored)
