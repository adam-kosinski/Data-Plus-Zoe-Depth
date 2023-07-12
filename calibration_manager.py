import numpy as np
import os
import math
from PIL import Image

from gui_utils import clear_layout_contents, depth_to_pixmap


from PyQt6.QtCore import QSize, Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QPoint
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QLineEdit
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QDoubleValidator, QFont


class CalibrationManager:
    def __init__(self, main_window):

        self.main_window = main_window
        
        # make scenes and init views
        self.scene = QGraphicsScene(0,0,400,200)
        font = QFont("Times", 12)
        self.choose_image_scene = QGraphicsScene()
        self.choose_image_scene.addText("Please choose a reference image", font)
        self.loading_depth_scene = QGraphicsScene()
        self.loading_depth_scene.addText("Loading depth preview...", font)

        self.ref_view = main_window.referenceGraphicsView
        self.depth_view = main_window.depthGraphicsView
        

        # set full viewport update (for background) - gets rid of weird artifacts when items move around, due to update region calculations missing like half a pixel
        self.ref_view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.depth_view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # antialiasing
        self.ref_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.depth_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        # widget references
        self.vbox = main_window.calibrationList

        # event listeners
        main_window.openCalibrationImageButton.clicked.connect(self.choose_ref_image)
        self.scene.mousePressEvent = self.scene_mouse_press
        main_window.backToMainButton.clicked.connect(self.exit_calibration)
        main_window.testReset.clicked.connect(self.reset)

        # set up other stuff and state via the reset function
        self.calibration_entries = []
        self.reset()
    
    def reset(self):
        clear_layout_contents(self.vbox)
        for entry in self.calibration_entries:
            entry.remove()
        self.calibration_entries = []

        self.saved = True
        self.rel_depth = None
        self.ref_image_path = None
        self.slope = None
        self.intercept = None
        self.ref_pixmap = None

        self.ref_view.setScene(self.choose_image_scene)
        self.depth_view.setScene(self.choose_image_scene)

        # prime backgrounds for pixmaps, no idea why we need this at the beginning but it doesn't work otherwise
        self.set_background(self.ref_view, QPixmap())
        self.set_background(self.depth_view, QPixmap())
    
    def init_calibration(self, deployment):
        print(deployment)
        self.main_window.screens.setCurrentWidget(self.main_window.calibrationScreen)
        self.main_window.calibrationTitle.setText("Deployment: " + deployment)

    def choose_ref_image(self):
        dialog = QFileDialog(parent=self.main_window, caption="Choose Reference Image", directory=self.main_window.root_path)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("*.jpg *.jpeg *.png")
        if not dialog.exec():
            return
        abs_path = dialog.selectedFiles()[0]
        self.ref_image_path = os.path.relpath(abs_path, self.main_window.root_path)
        print(self.ref_image_path)
        self.init_calibration_image(self.ref_image_path)

        # TEMP
        with Image.open("second_results/RCNX0332_raw.png") as raw_img:
            self.rel_depth = np.asarray(raw_img) / 256
        self.set_background(self.depth_view, depth_to_pixmap(self.rel_depth, rescale_width=400))
        self.depth_view.setScene(self.scene)

    def init_calibration_image(self, fpath):
        # update scenes
        self.ref_view.setScene(self.scene)
        self.depth_view.setScene(self.loading_depth_scene)

        # display pixmap background
        self.ref_pixmap = QPixmap(fpath).scaledToWidth(400, mode=Qt.TransformationMode.SmoothTransformation)
        self.set_background(self.ref_view, self.ref_pixmap)
        # sizing
        self.ref_view.setMinimumSize(self.ref_pixmap.width(), self.ref_pixmap.height())
        self.depth_view.setMinimumSize(self.ref_pixmap.width(), self.ref_pixmap.height())


    def set_background(self, graphics_view, pixmap):
        graphics_view.drawBackground = lambda painter, rect: painter.drawPixmap(rect, pixmap, rect)
        graphics_view.scene().update()

    def scene_mouse_press(self, e):
        pos = e.scenePos()
        QGraphicsScene.mousePressEvent(self.scene, e)
        if e.isAccepted():
            return
        self.add_calibration_entry(pos.x(), pos.y())

    def add_calibration_entry(self, x, y, distance=None):
        print(x,y)
        if not self.ref_image_path:
            return
        entry = CalibrationEntry(x, y, distance, self.scene, self.vbox, self.calibration_entries)
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

        # Ax = b for linear regression
        A = []
        b = []

        # get pairs of truth, estimated
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
        
        # need at least 2 points for linear regression
        if len(b) < 2:
            return
        
        print("update calibration")
        
        # do linear regression
        self.slope, self.intercept = np.linalg.lstsq(A, b, rcond=None)[0]
        self.saved = False
        print(self.slope, self.intercept)

        # correct the depth map
        corrected = np.maximum(0.0, self.rel_depth * self.slope + self.intercept)
        self.set_background(self.depth_view, depth_to_pixmap(corrected, rescale_width=400))

    def exit_calibration(self):
        if not self.saved:
            button = QMessageBox.question(self.main_window, "Calibration Not Saved", "Are you sure you want to exit? The changes you made to this calibration aren't saved.")
            if button != QMessageBox.StandardButton.Yes:
                return
        self.main_window.screens.setCurrentWidget(self.main_window.mainScreen)
    
    





class CalibrationEntry:
    def __init__(self, x, y, distance, scene, vbox, calibration_entries):
        # note that some signals (remove button, text changed, etc will be handled by the code using this class)

        # references
        self.scene = scene
        self.calibration_entries = calibration_entries

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
        for entry in self.calibration_entries:
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
        del self.point  # necessary? not sure

        # remove list hbox
        clear_layout_contents(self.hbox)
        self.hbox.parent().removeItem(self.hbox)
