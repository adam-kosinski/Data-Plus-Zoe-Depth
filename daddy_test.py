import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file1")
parser.add_argument("file2")
args = parser.parse_args()

with Image.open(args.file1) as img1:
    data1 = np.asarray(img1).flatten() / 256

with Image.open(args.file2) as img2:
    data2 = np.asarray(img2).flatten() / 256

plt.scatter(data1[::356], data2[::356])
plt.show()