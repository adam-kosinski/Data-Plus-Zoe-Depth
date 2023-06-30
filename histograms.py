import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('-n', '--normalize', action='store_true')
args = parser.parse_args()

if not os.path.exists("./histograms"):
    os.makedirs("./histograms")

for file in os.listdir(args.directory):
    # if not file.endswith("_raw.png"):
    #     continue

    print(file)

    with Image.open(args.directory + "/" + file) as img:
        data = np.asarray(img)
        data = data / 256
        
        plt.clf()

        if args.normalize:
            data = data - np.mean(data)
            data = data / np.std(data)
        else:
            plt.xlim(0, 20)

        plt.hist(data.flatten(), bins=20)
        plt.savefig("./histograms/" + os.path.splitext(file)[0] + "_hist.png")

        
