import numpy as np
from PIL import Image
import argparse
import random
import os

def is_night(img_path):
    with Image.open(img_path).convert("RGB") as img:
        np_img = np.asarray(img)
    for i in range(5):
        x = random.randint(0, np_img.shape[1])
        y = random.randint(0, np_img.shape[0])
        equal = np_img[y,x,0] == np_img[y,x,1] and np_img[y,x,1] == np_img[y,x,2]
        if not equal:
            return False
    return True
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-d", "--is_dir", action='store_true')
    args = parser.parse_args()

    if args.is_dir:
        for img in os.listdir(args.path):
            print(img, is_night(os.path.join(args.path, img)))
    else:
        print(is_night(args.path))