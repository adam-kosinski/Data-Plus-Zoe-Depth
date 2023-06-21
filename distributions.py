import numpy as np
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()

for file in os.listdir(args.directory):
    if not file.endswith("_raw.png"):
        continue

    print(file)

    with Image.open(args.directory + "/" + file) as img:
        data = np.asarray(img)
        data = data / 256
        print("     ", np.mean(data), np.std(data), np.max(data) - np.min(data))
