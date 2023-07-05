import torch
import numpy as np
import os
from PIL import Image
from PIL.ImageQt import ImageQt
from zoedepth.utils.misc import colorize
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow,
    QVBoxLayout, QHBoxLayout, QStackedLayout,
    QWidget, QPushButton, QLabel, QStackedWidget, QDialog, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.zoe = None

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

  
    
    def open_image(self):
        # open image with static function - args are: parent, caption, directory, filter. Returns: (filename, filter)
        self.rgb_filename = QFileDialog.getOpenFileName(self, "Open Image", ".", "Image Files (*.png *.jpg)")[0]
        if self.rgb_filename:
            print(self.rgb_filename)
            self.rgb_image.setPixmap(QPixmap(self.rgb_filename).scaledToWidth(400))
            self.run_button.setEnabled(True)
            self.depth_image.setPixmap(QPixmap())
    
    def run_zoedepth(self):
        print("Running ZoeDepth on " + self.rgb_filename)
        self.open_image_button.setEnabled(False)
        self.run_button.setEnabled(False)

        # Prepare model if not done
        if not self.zoe:
            model_zoe_nk = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True, config_mode="eval")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.zoe = model_zoe_nk.to(DEVICE)

        # Do inference
        with Image.open(self.rgb_filename).convert("RGB") as image:
            depth = self.zoe.infer_pil(image)  # as numpy
            colored = colorize(depth)
            qt_depthmap = ImageQt(Image.fromarray(colored))
            pixmap = QPixmap.fromImage(qt_depthmap).scaledToWidth(400)
            self.depth_image.setPixmap(pixmap)

        self.open_image_button.setEnabled(True)
        self.run_button.setEnabled(True)





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