import torch
import numpy as np
import os
from PIL import Image
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool



def build_zoedepth_model():
    model_zoe_nk = torch.hub.load(os.path.dirname(__file__), "ZoeD_NK", source="local", pretrained=True, config_mode="eval")
    # first arg is repository root, which because of copying datas in the spec file will be the folder where the executable runs
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)
    return zoe


class ZoeDepthResultSignal(QObject):
    result = pyqtSignal(object, str) # depth, filename absolute path


class ZoeWorker(QRunnable):
    # convenience runnable, if code is already in a separate thread it can just run zoedepth itself
    def __init__(self, main_window, rgb_filename, deployment):
        super().__init__()
        self.main_window = main_window
        self.rgb_filename = rgb_filename
        self.deployment = deployment
        self.signals = ZoeDepthResultSignal()

    def run(self):
        if not self.main_window.zoedepth_model:
            self.main_window.zoedepth_model = build_zoedepth_model()

        print("Running ZoeDepth on " + self.rgb_filename)
        with Image.open(self.rgb_filename).convert("RGB") as image:
            cropped_image = self.main_window.crop_manager.crop(image, self.deployment)
            depth = self.main_window.zoedepth_model.infer_pil(cropped_image)  # as numpy
            print("zoe worker finished inference")
            self.signals.result.emit(depth, self.rgb_filename)