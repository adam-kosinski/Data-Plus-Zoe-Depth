import torch
import numpy as np
import argparse
import os
from PIL import Image
from zoedepth.utils.misc import colorize

# get input dir
parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
args = parser.parse_args()

# Zoe_NK
model_zoe_nk = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True, config_mode="eval")

##### prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)

for file in os.listdir(args.input_dir):
    print("Doing inference on " + file)

    with Image.open(args.input_dir + "/" + file).convert("RGB") as image:

        depth = zoe.infer_pil(image)  # as numpy

        print("inference done, saving")

        # create output dir if necessary
        if not os.path.exists("./output"):
            os.makedirs("./output")

        # save raw
        from zoedepth.utils.misc import save_raw_16bit
        fpath = "./output/" + os.path.splitext(file)[0] + "_raw.png"
        save_raw_16bit(depth, fpath)

        # save colored output
        colored = colorize(depth)
        fpath_colored = "./output/" + os.path.splitext(file)[0] + "_colored.png"
        Image.fromarray(colored).save(fpath_colored)

print("Done")