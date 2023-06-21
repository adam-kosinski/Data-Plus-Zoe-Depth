from PIL import Image
import numpy as np
import cv2
import argparse
import os

def putDist(data, img, x, y, index=0, label=""):
    bgr_color=(0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = " " + str(round(data[y][x], 2)) + "m" + "  " + label
    cv2.rectangle(img, (x + 15, y - 10 + 30*index), (x + 120, y + 30 + 30*index), (255, 255, 255), -1)
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
parser.add_argument('-c', '--colored_filename')
parser.add_argument('-a', '--all', action='store_true')
parser.add_argument('-r', '--round', action='store_true')
args = parser.parse_args()

colored_filename = None
if args.colored_filename:
    colored_filename = args.colored_filename

print("colored_filename", colored_filename)

with Image.open(args.raw_filename) as img:
    data = np.asarray(img)
    data = data / 256   # undo the scaling that was used when storing the image
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