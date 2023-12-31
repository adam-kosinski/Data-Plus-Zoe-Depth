import numpy as np
import PyQt5.QtGui

from zoedepth.utils.misc import colorize

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
from PIL.ImageQt import ImageQt

def clear_layout_contents(layout):
    # https://stackoverflow.com/questions/9374063/remove-all-items-from-a-layout/9383780#9383780
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clear_layout_contents(item.layout())

def depth_to_pixmap(depth, rescale_width = None):
    # converts numpy depth map to a QPixmap, rounding the depths and rescaling the pixmap
    colored = colorize(np.floor(depth), vmin=0)
    img = Image.fromarray(colored)
    #pixmap = QPixmap.fromImage(imgg)
    pixmap = QPixmap.fromImage(ImageQt(img))
    return pixmap if rescale_width is None else pixmap.scaledToWidth(400, mode=Qt.TransformationMode.SmoothTransformation)