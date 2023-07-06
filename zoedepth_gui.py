import torch
import numpy as np
import os
from PIL import Image
from PIL.ImageQt import ImageQt
from zoedepth.utils.misc import colorize
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow,
    QVBoxLayout, QHBoxLayout, QStackedLayout,
    QWidget, QPushButton, QLabel, QStackedWidget, QDialog, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap

base_dir = os.path.dirname(__file__)

class ResultSignal(QObject):
    result = pyqtSignal(object)

class BuildZoeModel(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = ResultSignal()
    def run(self):
        try:
            model_zoe_nk = torch.hub.load(base_dir, "ZoeD_NK", source="local", pretrained=True, config_mode="eval")
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
            


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.zoe = None
        self.image_open = False
        self.threadpool = QThreadPool()

        self.setWindowTitle("ZoeDepth NK GUI")
        layout = QVBoxLayout()


        control_buttons = QHBoxLayout()

        self.open_image_button = QPushButton("Select Image")
        self.open_image_button.clicked.connect(self.open_image)
        self.open_image_button.setMaximumWidth(100)
        control_buttons.addWidget(self.open_image_button)

        self.run_button = QPushButton("Run ZoeDepth")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_zoedepth)
        self.run_button.setMaximumWidth(100)
        control_buttons.addWidget(self.run_button)

        self.info = QLabel("Building model...")
        self.info.setMaximumHeight(50)
        control_buttons.addWidget(self.info)


        images = QHBoxLayout()

        self.rgb_image = QLabel()
        self.rgb_filename = None
        images.addWidget(self.rgb_image)

        self.depth_image = QLabel()
        images.addWidget(self.depth_image)
        

        layout.addLayout(control_buttons)
        layout.addLayout(images)
        widget = QWidget()
        widget.setLayout(layout)
        self.resize(QSize(800, 400))
        self.setCentralWidget(widget)

        # start building the model
        worker = BuildZoeModel()
        worker.signals.result.connect(self.store_zoe_model)
        self.threadpool.start(worker)

    def store_zoe_model(self, zoe):
        self.zoe = zoe
        self.info.setText("Model ready")
        if self.image_open:
            self.run_button.setEnabled(True)
    
    def open_image(self):
        # open image with static function - args are: parent, caption, directory, filter. Returns: (filename, filter)
        self.rgb_filename = QFileDialog.getOpenFileName(self, "Open Image", ".", "Image Files (*.png *.jpg)")[0]
        if self.rgb_filename:
            print(self.rgb_filename)
            self.image_open = True
            self.rgb_image.setPixmap(QPixmap(self.rgb_filename).scaledToWidth(400))
            self.depth_image.setPixmap(QPixmap())
            if self.zoe:
                self.run_button.setEnabled(True)

    
    def run_zoedepth(self):
        # Make sure model was built
        if not self.zoe:
            return
        
        print("Running ZoeDepth on " + self.rgb_filename)
        self.open_image_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.info.setText("Running...")

        worker = ZoeWorker(self.zoe, self.rgb_filename)
        worker.signals.result.connect(self.process_zoe_results)
        self.threadpool.start(worker)
        
    def process_zoe_results(self, depth):
        colored = colorize(depth)
        qt_depthmap = ImageQt(Image.fromarray(colored))
        pixmap = QPixmap.fromImage(qt_depthmap).scaledToWidth(400)
        self.depth_image.setPixmap(pixmap)
        
        self.open_image_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.info.setText("Done")




if __name__ == '__main__':
    # create application
    app = QApplication([]) # replace [] with sys.argv if want to use command line args

    # create window
    window = MainWindow()
    window.show()

    # allow ctrl + C to work
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # start the event loop
    app.exec()