from PIL import Image
import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt


coords = None
input_string = ""
# matrix and vector for linear regression
A = []
b = []


def updateDisplay():
    img_copy = colored_depth_map.copy()
    if coords:
        bgr_color=(0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_copy, input_string, (coords[0] + 20, coords[1] + 20), font, 1, bgr_color, 2)
        cv2.circle(img_copy, coords, 6, bgr_color, 2)
    cv2.imshow('display', img_copy)

def click_event(event, x, y, flags, params):
    global coords
    global input_string
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset input
        coords = (x,y)
        input_string = ""
        updateDisplay()




# get filename
parser = argparse.ArgumentParser()
parser.add_argument('raw_filename')
parser.add_argument('-c', '--colored_filename')
args = parser.parse_args()

colored_filename = args.raw_filename.split("_raw")[0] + "_colored.png"
if not os.path.exists(colored_filename):
    colored_filename = None
if args.colored_filename:
    colored_filename = args.colored_filename

print("colored_filename", colored_filename)
print("")

with Image.open(args.raw_filename) as img:
    data = np.asarray(img)
    data = data / 256   # undo the scaling that was used when storing the image
 
# display the color depth map
if colored_filename:
    colored_depth_map = cv2.imread(colored_filename, cv2.IMREAD_COLOR)
else:
    rescaled = (data - np.min(data)) / (np.max(data) - np.min(data))
    colored_depth_map = 1 - rescaled

cv2.imshow('display', colored_depth_map)

# click event handler
cv2.setMouseCallback('display', click_event)


# read keystrokes until q is pressed, then close window and save calibration
key = None
while key != 113 and cv2.getWindowProperty('display', 0) >= 0:
    key = cv2.waitKeyEx(0)
    if coords and (key >= 49 and key <= 57 or key == 46):
        input_string += chr(key)
        updateDisplay()
    elif coords and key == 8:
        # backspace key
        input_string = input_string[:-1]
        updateDisplay()
    elif coords and key == 13 and len(input_string) > 0:
        # enter key was pressed
        estim = data[coords[1]][coords[0]]
        truth = float(input_string)
        A.append([estim, 1])
        b.append(truth)

        print(coords)
        print("   ", f"estim {round(estim, 2)}", f"truth {round(truth, 2)}\n")
        
        input_string = ""
        coords = None
        updateDisplay()

cv2.destroyAllWindows()


# do linear regression to find scale and shift
print(A)
print(b)
slope, intercept = np.linalg.lstsq(A, b, rcond=None)[0]
print(slope, intercept)

# save calibrated image
calib_data = data * slope + intercept
calib_data = np.maximum(np.zeros(calib_data.shape), calib_data)
os.makedirs("./calibrated", exist_ok=True)
Image.fromarray((calib_data * 256).astype(np.uint16)).save("./calibrated/" + os.path.basename(args.raw_filename))