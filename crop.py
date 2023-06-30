from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("--left", type=int, default=0)
parser.add_argument("--top", type=int, default=0)
parser.add_argument("--right", type=int, default=0)
parser.add_argument("--bottom", type=int, default=0)
args = parser.parse_args()

save_dir = os.path.join(args.input_dir, "cropped")
os.makedirs(save_dir, exist_ok=True)

for file in os.listdir(args.input_dir):
    if not os.path.isfile(os.path.join(args.input_dir, file)):
        continue
    print(file)

    with Image.open(os.path.join(args.input_dir, file)) as img:
        w, h = img.size
        box = (args.left, args.top, w - args.right, h - args.bottom)
        cropped = img.crop(box)
        cropped.save(os.path.join(save_dir, file))
