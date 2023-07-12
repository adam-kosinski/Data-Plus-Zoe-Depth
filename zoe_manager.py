import torch
import numpy as np
import os
from PIL import Image
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal



class ZoeManager(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # might be useful who knows

        self.zoe = None
        self.threadpool = QThreadPool()
        self.queue = [] # see self.infer
        
        # start building the model
        worker = BuildZoeModel()
        worker.signals.result.connect(self.model_built_callback)
        self.threadpool.start(worker)
    
    def model_built_callback(self, zoe):
        print("Model ready for use")
        self.zoe = zoe
        for job in self.queue:
            self.infer(job["rgb_filename"], job["callback"])
    
    def infer(self, abs_fpath, callback):
        # abs_fpath is ABSOLUTE path to the rgb image to infer depth on
        # callback is a function that takes one argument - the outputted depth in np array form
        
        if not self.zoe:
            print("Queueing request for " + abs_fpath)
            # jobs submitted before the model is built are stored in the queue
            # once the model is built, the queue is processed and not used again (threadpool will handle queueing)
            self.queue.append({
                "rgb_filename": abs_fpath,
                "callback": callback
            })
            return
        
        print("Running ZoeDepth on " + abs_fpath)

        worker = ZoeWorker(self.zoe, abs_fpath)
        worker.signals.result.connect(callback)
        self.threadpool.start(worker)



class ResultSignal(QObject):
    result = pyqtSignal(object)


class BuildZoeModel(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = ResultSignal()
    def run(self):
        try:
            # first arg is repository root, which because of copying datas in the spec file will be the folder where the executable runs
            model_zoe_nk = torch.hub.load(os.path.dirname(__file__), "ZoeD_NK", source="local", pretrained=True, config_mode="eval")
            
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            zoe = model_zoe_nk.to(DEVICE)
            self.signals.result.emit(zoe)
        except:
            import traceback
            traceback.print_exc()


class ZoeWorker(QRunnable):
    def __init__(self, zoe, rgb_filename):
        super().__init__()
        self.zoe = zoe  # model object, pre-built (so we don't do it multiple times)
        self.rgb_filename = rgb_filename
        self.signals = ResultSignal()

    def run(self):
        # Do inference
        with Image.open(self.rgb_filename).convert("RGB") as image:
            depth = self.zoe.infer_pil(image)  # as numpy
            self.signals.result.emit(depth)