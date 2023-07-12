import numpy as np
from PIL import Image
from zoedepth.utils.misc import colorize
from PIL.ImageQt import ImageQt

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

def clear_layout_contents(layout):
    for i in reversed(range(layout.count())):
        widgetToRemove = layout.takeAt(i).widget()
        layout.removeWidget(widgetToRemove)
        widgetToRemove.deleteLater()

def depth_to_pixmap(depth, rescale_width = None):
    # converts numpy depth map to a QPixmap, rounding the depths and rescaling the pixmap
    colored = colorize(np.floor(depth), vmin=0)
    img = Image.fromarray(colored)
    pixmap = QPixmap.fromImage(ImageQt(img))
    return pixmap if rescale_width is None else pixmap.scaledToWidth(400, mode=Qt.TransformationMode.SmoothTransformation)
    