import torch
import numpy as np
import os
from PIL import Image
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool


class ResultSignal(QObject):
    result = pyqtSignal(object)


class ZoeManager(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # might be useful who knows

        self.zoe = None
        self.job_running = False
        self.threadpool = QThreadPool()
        self.queue = [] # see self.infer
        
        # start building the model
        worker = BuildZoeModel()
        worker.signals.result.connect(self.model_built_callback)
        self.threadpool.start(worker)
    
    def model_built_callback(self, zoe):
        print("Model ready for use")
        self.zoe = zoe
        if len(self.queue) > 0:
            job = self.queue.pop(0)
            self.infer(job["rgb_filename"], job["callback"])
    
    def infer(self, abs_fpath, callback):
        # abs_fpath is absolute path to the rgb image to infer depth on
        # callback is a function that takes args: depth (numpy array), abs_fpath
        
        if not self.zoe or self.job_running:
            print("Queueing request for " + abs_fpath)
            self.queue.append({
                "rgb_filename": abs_fpath,
                "callback": callback
            })
            return
        
        print("Running ZoeDepth on " + abs_fpath)
        self.job_running = True

        worker = ZoeWorker(self.zoe, abs_fpath)
        worker.signals.result.connect(lambda depth: callback(depth, abs_fpath))
        worker.signals.result.connect(self.worker_finished)
        self.threadpool.start(worker)
    
    def worker_finished(self, depth):
        self.job_running = False
        if len(self.queue) > 0:
            job = self.queue.pop(0)
            self.infer(job["rgb_filename"], job["callback"])



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