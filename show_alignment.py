import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

parser = argparse.ArgumentParser()
parser.add_argument("raw1")
parser.add_argument("raw2")
parser.add_argument("-b", "--bin_size")
args = parser.parse_args()

if args.bin_size:
    bin_size = float(args.bin_size)
else:
    bin_size = 1    # meters

# load image data
with Image.open(args.raw1) as img1:
    data1 = np.asarray(img1) / 256
with Image.open(args.raw2) as img2:
    data2 = np.asarray(img2) / 256

# crop to same height because my manual crop was imperfect
min_height = min(data1.shape[0], data2.shape[0])
data1 = data1[:min_height,:]
data2 = data2[:min_height,:]

# normalize to data1
norm1 = normalize(data1) * 2 + 4
norm2 = normalize(data2) * 2 + 4

# get binned image
diff = norm1 - norm2
binned = np.fix(diff / bin_size).astype(int)
binned_masked = np.ma.masked_array(binned, mask=(binned == 0))
bin0_only1 = np.ma.masked_array(norm1, mask=(binned != 0))
bin0_only2 = np.ma.masked_array(norm2, mask=(binned != 0))
n_bins = np.max(binned) - np.min(binned)
cmap = plt.get_cmap("Spectral", n_bins)


fig = plt.figure()

# plot scatter plot
fig.add_subplot(2,3,(1,4))
plt.gca().set_title("Bin size = " + str(bin_size) + "m difference")
plt.scatter(bin0_only1.flatten()[::567], bin0_only2.flatten()[::567], c='lightgray')
plt.scatter(norm1.flatten()[::567], norm2.flatten()[::567], c=binned_masked.flatten()[::567], cmap=cmap)
ticks = np.linspace(np.min(binned), np.max(binned), n_bins+1)
plt.colorbar(ticks=ticks)

min1 = np.min(norm1)
max1 = np.max(norm1)
plt.plot([min1, max1], [min1, max1], color="black")
plt.xlabel(os.path.split(args.raw1)[1])
plt.ylabel(os.path.split(args.raw2)[1] + " (normalized)")


# original depth images
fig.add_subplot(2,3,2)
plt.axis('off')
plt.gca().set_title(os.path.split(args.raw1)[1])
plt.imshow(norm1, interpolation='nearest', cmap='binary')

fig.add_subplot(2,3,5)
plt.axis('off')
plt.gca().set_title(os.path.split(args.raw2)[1])
plt.imshow(norm2, interpolation='nearest', cmap='binary')


# binned image
fig.add_subplot(2,3,3)
plt.axis('off')
plt.gca().set_title("Redder = estimated closer")
plt.imshow(norm1*(2**15), interpolation='nearest', cmap='binary')
plt.imshow(binned_masked, interpolation='nearest', cmap=cmap, alpha=0.8)

fig.add_subplot(2,3,6)
plt.axis('off')
plt.gca().set_title("Bluer = estimated closer")
plt.imshow(norm2, interpolation='nearest', cmap='binary')
plt.imshow(binned_masked, interpolation='nearest', cmap=cmap, alpha=0.8)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

