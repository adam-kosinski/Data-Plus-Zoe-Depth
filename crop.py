from PIL import Image
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("--left", type=int, default=0)
parser.add_argument("--top", type=int, default=0)
parser.add_argument("--right", type=int, default=0)
parser.add_argument("--bottom", type=int, default=0)
args = parser.parse_args()


os.makedirs(os.path.join(args.input_dir, "uncropped"), exist_ok=True)

for file in os.listdir(args.input_dir):
    fpath = os.path.join(args.input_dir, file)
    ext = os.path.splitext(file)[1].lower()
    if not(ext == ".png" or ext == ".jpg" or ext == ".jpeg") or os.path.isdir(file):
            continue
    print(file)

    with Image.open(fpath) as img:
        w, h = img.size
        box = (args.left, args.top, w - args.right, h - args.bottom)
        cropped = img.crop(box)
    
    shutil.move(fpath, os.path.join(args.input_dir, "uncropped"))
    cropped.save(os.path.join(args.input_dir, file))
