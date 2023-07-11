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
    QHBoxLayout,
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
from PyQt6.QtGui import QPixmap, QPen, QPainter, QDoubleValidator
from PyQt6 import uic



# TODO
# implement itemChange event for points, to avoid dragging out of bounds
# https://stackoverflow.com/questions/3548254/restrict-movable-area-of-qgraphicsitem

# figure out flow for opening new image and loading the depth display, pixmap stuff, label saying loading

# abstract all the calibration stuff to a class in a different file (class CalibrationManager)



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
    


class CalibrationEntry:
    def __init__(self, x, y, distance, scene, vbox, entry_list):
        # note that some signals (remove button, finished editing, etc will be dealt with by the code using this class)

        # references
        self.scene = scene
        self.entry_list = entry_list

        # point
        self.point = QGraphicsEllipseItem(-5, -5, 10, 10)   # scene coords, -5 offset so center is at item coords 0,0
        self.point.setPos(x, y)
        self.point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.point.mousePressEvent = lambda e: self.focus()
        scene.addItem(self.point)

        # add list entry
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("enter distance (m)")
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.line_edit.setValidator(validator)
        self.line_edit.focusInEvent = self.line_edit_focused
        self.line_edit.focusOutEvent = self.line_edit_unfocused
        self.line_edit.editingFinished.connect(self.unfocus)
        self.delete_button = QPushButton("Remove")
        
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.line_edit)
        self.hbox.addWidget(self.delete_button)
        vbox.addLayout(self.hbox)

        if distance:
            self.line_edit.setText(distance)
            self.unfocus()
        else:
            self.focus()
    
    def focus(self):
        self.line_edit.setFocus()   # this will trigger the point being highlighted too

    def line_edit_focused(self, e):
        QLineEdit.focusInEvent(self.line_edit, e)
        self.highlight_point()
    
    def line_edit_unfocused(self, e):
        QLineEdit.focusOutEvent(self.line_edit, e)
        self.unfocus()
    
    def highlight_point(self):
        for entry in self.entry_list:
            if entry != self:
                entry.unfocus()
        fat_pen = QPen(Qt.GlobalColor.red)
        fat_pen.setWidth(3)
        self.point.setPen(fat_pen)
    
    def unfocus(self):
        thin_pen = QPen(Qt.GlobalColor.red)
        self.point.setPen(thin_pen)
        self.line_edit.clearFocus()
    
    def remove(self):
        # remove point
        self.point.scene().removeItem(self.point)

        # remove list hbox
        clear_layout_contents(self.hbox)
        self.hbox.parent().removeItem(self.hbox)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)

        self.openRootFolder.clicked.connect(self.open_root_folder)

        self.referenceGraphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.deploymentGrid.setColumnStretch(1, 1)

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
        clear_layout_contents(self.calibrationList)
        self.calibration_entries = []
        
        # make scenes and assign to views
        ref_scene = QGraphicsScene(0,0,400,200)
        self.referenceGraphicsScene = ref_scene
        ref_view = self.referenceGraphicsView
        ref_view.setScene(ref_scene)

        depth_scene = QGraphicsScene(0,0,400,200)
        self.depthGraphicsScene = depth_scene
        depth_view = self.depthGraphicsView
        depth_view.setScene(ref_scene)

        # display pixmap background
        ref_pixmap = QPixmap(fpath).scaledToWidth(400, mode=Qt.TransformationMode.SmoothTransformation)
        self.ref_pixmap = ref_pixmap
        with Image.open("second_results/RCNX0332_raw.png") as raw_img:
            self.rel_depth = np.asarray(raw_img) / 256
        depth_pixmap = depth_to_pixmap(self.rel_depth, rescale_width=400)

        self.set_background(ref_view, ref_pixmap)
        self.set_background(depth_view, depth_pixmap)

        # sizing
        ref_view.setMinimumSize(ref_pixmap.width(), ref_pixmap.height())
        depth_view.setMinimumSize(ref_pixmap.width(), ref_pixmap.height())

        # event listeners
        ref_scene.mousePressEvent = self.scene_mouse_press

    def set_background(self, graphics_view, pixmap):
        graphics_view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate) # this gets rid of weird artifacts when items move around, due to update region calculations missing like half a pixel
        graphics_view.drawBackground = lambda painter, rect: painter.drawPixmap(rect, pixmap, rect)
        graphics_view.scene().update()

    def scene_mouse_press(self, e):
        pos = e.scenePos()
        QGraphicsScene.mousePressEvent(self.referenceGraphicsScene , e)
        if e.isAccepted():
            return
        self.add_calibration_entry(pos.x(), pos.y())


    def add_calibration_entry(self, x, y, distance=None):
        entry = CalibrationEntry(x, y, distance, self.referenceGraphicsScene, self.calibrationList, self.calibration_entries)
        self.calibration_entries.append(entry)
        entry.delete_button.clicked.connect(lambda: self.remove_calibration_entry(entry))
        entry.line_edit.textChanged.connect(self.update_calibration)
        entry.point.mouseReleaseEvent = lambda e: self.point_mouserelease(e, entry.point)

    def remove_calibration_entry(self, entry):
        entry.remove()
        self.calibration_entries.remove(entry)

    def point_mouserelease(self, e, point):
        # detect when we stop dragging, so we can update the calibration
        QGraphicsEllipseItem.mouseReleaseEvent(point, e)
        self.update_calibration()

    def update_calibration(self):
        print("update calibration")

        A = []
        b = []

        # get gt, estimated pairs
        for entry in self.calibration_entries:
            text = entry.line_edit.text()
            if len(text) == 0:
                continue
            truth = float(text)
            pos = entry.point.pos()
            x = math.floor(self.rel_depth.shape[1] * pos.x() / self.ref_pixmap.width())
            y = math.floor(self.rel_depth.shape[0] * pos.y() / self.ref_pixmap.height())
            estim = self.rel_depth[y][x]

            A.append([estim, 1])
            b.append(truth)
        
        if len(b) < 2:
            return
        
        # do linear regression
        self.slope, self.intercept = np.linalg.lstsq(A, b, rcond=None)[0]
        print(self.slope, self.intercept)

        # correct the depth map
        corrected = np.maximum(0.0, self.rel_depth * self.slope + self.intercept)
        self.set_background(self.depthGraphicsView, depth_to_pixmap(corrected, rescale_width=400))


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