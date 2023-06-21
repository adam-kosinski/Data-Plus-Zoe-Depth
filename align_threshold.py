import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
from average_depths import get_average, show

# NOTE: assuming one bbox per image in average_depths


parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("--bbox_file")
parser.add_argument("-b", "--band_width")
args = parser.parse_args()

band_width = 1
if args.band_width:
    band_width = float(args.band_width)

# get list of files to use, with directory path
files = []
for file in os.listdir(args.directory):
    if file.endswith("_raw.png"):
        files.append(args.directory + "/" + file)

# get average depth map, and bbox-masked, cropped, normalized images
if args.bbox_file:
    avg, norm = get_average(files, args.bbox_file, median=True)
else:
    avg, norm = get_average(files, None, median=True)


# process each image
for i, img in enumerate(norm):
    print("processing " + files[i])
    img_nomask = np.array(img)

    out = np.zeros(img.shape)

    # iterate through depth bands, each band is 1 wide
    close = np.min(img_nomask)
    far = np.max(img_nomask)
    print("range", close, far)
    for band_start in np.arange(math.floor(close), far, band_width):
        # create mask for that depth band (true everywhere not the band)
        bbox_mask = np.ma.getmaskarray(img)
        band_mask = (img_nomask < band_start) | (img_nomask > band_start + band_width)
        
        # get pixels from the average image that are in this band, and find the average
        masked_avg = np.ma.array(avg, mask=(band_mask | bbox_mask))
        depth_val = np.ma.average(masked_avg)
        print(band_start, " ", round(float(depth_val), 1))

        # assign that depth to this band
        out[~band_mask] = depth_val
    
    show(out)

    if not os.path.exists("./aligned_output"):
        os.makedirs("./aligned_output")
    fpath = "./aligned_output/" + os.path.basename(files[i])
    cv2.imwrite(fpath, (256 * out).astype("uint16"))
    print("Saved depth map to " + fpath)

# create mask for that depth band, combine with the bbox mask for this image (np.ma.getmaskarray(arr))
#   - actually not necessary, bbox is already masked out, depth query shouldn't fetch those pixels
# get average (or median) from average image, make sure to use ma.average or ma.median
# assign the value to the corresponding pixels in the output array
# display the output array
# save the output array


'''
4.8 16.4
4 16
16 - 4 = 12
12 / 2 = 6
ceil -> 6 bands

4, 6, 8, 10, 12, 14, 16

'''