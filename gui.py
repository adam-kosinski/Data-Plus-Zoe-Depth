import numpy as np
import os
import functools
import math
from PIL import Image
from PIL.ImageQt import ImageQt

from zoedepth.utils.misc import colorize


from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QPoint
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QLineEdit
)
from PyQt6.QtGui import QPixmap, QPen, QPainter
from PyQt6 import uic


# TODO
# when pop widget from calibration lists, delete the widget too, maybe something like this:
# import sip
# layout.removeWidget(self.widget_name)
# sip.delete(self.widget_name)

# implement itemChange event for points, so they drag together, and to avoid dragging out of bounds
# https://stackoverflow.com/questions/3548254/restrict-movable-area-of-qgraphicsitem



def clear_layout_contents(layout):
    for i in reversed(range(layout.count())): 
        widgetToRemove = layout.itemAt(i).widget()
        layout.removeWidget(widgetToRemove)
        widgetToRemove.deleteLater()

def depth_to_pixmap(depth, rescale_width = None):
    # converts numpy depth map to a QPixmap, rounding the depths and rescaling the pixmap
    colored = colorize(np.floor(depth), vmin=0)
    img = Image.fromarray(colored)
    pixmap = QPixmap.fromImage(ImageQt(img))
    return pixmap if rescale_width is None else pixmap.scaledToWidth(400, mode=Qt.TransformationMode.SmoothTransformation)
    



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)

        self.openRootFolder.clicked.connect(self.open_root_folder)

        self.referenceGraphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.deploymentGrid.setColumnStretch(1, 1)
        self.calibrationGrid.setColumnStretch(0, 1)

        # temp
        self.open_root_folder()
        # with Image.open("second_results/calibrated/RCNX0332_raw.png") as raw_img:
        #     self.data = np.asarray(raw_img) / 256
        #     self.pixmap = depth_to_pixmap(self.data, rescale_width=400)
        #     self.imgDisplay.setPixmap(self.pixmap)
        
        # self.imgDisplay.setMouseTracking(True)
        # self.imgDisplay.mouseMoveEvent = self.update_dist

        self.init_calibration_image("second_cropped/RCNX0332.JPG")
        self.resize(QSize(800, 600))
    
    # def update_dist(self, e):
    #     rect = self.pixmap.rect()
    #     x = math.floor(self.data.shape[1] * e.position().x() / rect.width())
    #     y = math.floor(self.data.shape[0] * e.position().y() / rect.height())
    #     depth_val = round(self.data[y][x], 1)

    #     box = self.meterContainer
    #     dest = box.parentWidget().mapFromGlobal(e.globalPosition())
    #     box.move(int(dest.x()) + 20, int(dest.y()) + 10)
    #     self.meterDisplay.setText(f"{depth_val} m ")
        

    def open_root_folder(self):
        # dialog = QFileDialog(self)
        # dialog.setFileMode(QFileDialog.FileMode.Directory)
        # if dialog.exec():
        #     self.root_path = dialog.selectedFiles()[0]
        #     self.rootFolderLabel.setText(self.root_path)
        self.root_path = "."
        if True:

            # display deployments
            grid = self.deploymentGrid
            clear_layout_contents(grid)
            for item in os.listdir(self.root_path):
                if not os.path.isdir(item):
                    continue
                row_idx = grid.rowCount()
                button = QPushButton("Calibrate")
                button.clicked.connect(functools.partial(self.open_calibration_screen, item))   # functools for using the current value of item, not whatever it ends up being
                grid.addWidget(button, row_idx, 0)
                grid.addWidget(QLabel(item), row_idx, 1)
    
    def open_calibration_screen(self, deployment):
        print(deployment)
        self.screens.setCurrentWidget(self.calibrationScreen)

    def init_calibration_image(self, fpath):
        clear_layout_contents(self.calibrationGrid)
        self.calibration_img_points = []
        self.calibration_depth_points = []
        self.calibration_line_edits = []
        self.prev_calibration_point_valid = True

        # make scenes and assign to views
        ref_scene = QGraphicsScene(0,0,400,200)
        self.referenceGraphicsScene = ref_scene
        ref_view = self.referenceGraphicsView
        ref_view.setScene(ref_scene)

        depth_scene = QGraphicsScene(0,0,400,200)
        self.depthGraphicsScene = depth_scene
        depth_view = self.depthGraphicsView
        depth_view.setScene(depth_scene)

        # display pixmaps and set up event listeners
        pixmap = QPixmap(fpath).scaledToWidth(400, mode=Qt.TransformationMode.SmoothTransformation)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        pixmap_item.mousePressEvent = lambda e: self.add_calibration_point(e.pos().x(), e.pos().y())
        ref_scene.addItem(pixmap_item)

        ref_view.setMinimumSize(pixmap.width(), pixmap.height())
        depth_view.setMinimumSize(pixmap.width(), pixmap.height())


    def add_calibration_point(self, x, y, distance=None):
        print(x,y)
        if not self.prev_calibration_point_valid:
            self.calibration_img_points.pop()
            self.calibration_depth_points.pop()
            self.calibration_line_edits.pop()

        idx = len(self.calibration_img_points)

        # add image point
        img_point = QGraphicsEllipseItem(x-5, y-5, 10, 10)
        img_point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.referenceGraphicsScene.addItem(img_point)
        self.calibration_img_points.append(img_point)
        img_point.mousePressEvent = lambda e: self.focus_calibration_point(idx)

        # add depth map point
        depth_point = QGraphicsEllipseItem(x-5, y-5, 10, 10)
        depth_point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.depthGraphicsScene.addItem(depth_point)
        self.calibration_depth_points.append(depth_point)
        depth_point.mousePressEvent = lambda e: self.focus_calibration_point(idx)

        # add list entry
        grid = self.calibrationGrid
        row_idx = grid.rowCount()
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("enter distance (m)")
        delete_button = QPushButton("Remove")
        grid.addWidget(line_edit, row_idx, 0)
        grid.addWidget(delete_button, row_idx, 1)
        self.calibration_line_edits.append(line_edit)
        line_edit.focusInEvent = lambda e: self.line_edit_focused(e, line_edit, idx)

        # focus point if it's a new one
        if distance:
            line_edit.setText(str(distance))
        else:
            self.focus_calibration_point(idx)
        
        # set previous point false because no value typed in yet for this point
        self.prev_calibration_point_valid = False
    

    def line_edit_focused(self, e, line_edit, idx):
        self.focus_calibration_point(idx, line_edit_triggered=True)
        QLineEdit.focusInEvent(line_edit, e)
    

    def focus_calibration_point(self, idx, line_edit_triggered=False):
        items = self.referenceGraphicsScene.items()

        # update outline bolding
        thin_pen = QPen(Qt.GlobalColor.red)
        for item in self.calibration_img_points:
            item.setPen(thin_pen)
        for item in self.calibration_depth_points:
            item.setPen(thin_pen)

        fat_pen = QPen(Qt.GlobalColor.red)
        fat_pen.setWidth(3)
        self.calibration_img_points[idx].setPen(fat_pen)
        self.calibration_depth_points[idx].setPen(fat_pen)

        if not line_edit_triggered: # needed to avoid infinite loop
            self.calibration_line_edits[idx].setFocus()




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