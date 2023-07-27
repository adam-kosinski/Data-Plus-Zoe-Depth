import numpy as np
import os
from PIL import Image
import csv
import math

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsPathItem,
    QLineEdit
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QDoubleValidator, QFont, QPalette, QColor, QPainterPath


CROP_PIXMAP_WIDTH = 700
CROP_HANDLE_SIZE = 30
CROP_HANDLE_WIDTH = 4
MIN_CROP_RECT_WIDTH = CROP_HANDLE_SIZE * 3 + 10
MIN_CROP_RECT_HEIGHT = CROP_HANDLE_SIZE * 3 + 10


class CropManager:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.scene = QGraphicsScene()
        self.view = main_window.cropGraphicsView
        self.view.setScene(self.scene)

        self.crop_rect = None   # initialized when image is opened

        self.crop_image_abspath = None

        # units of the original image's pixels
        self.image_width = 0
        self.image_height = 0
        self.crop_top = 0
        self.crop_bottom = 0
        self.crop_left = 0
        self.crop_right = 0

        # event handlers
        main_window.openCropScreenButton.clicked.connect(lambda: main_window.screens.setCurrentWidget(main_window.cropScreen))
        main_window.openCropImageButton.clicked.connect(self.open_image)
        main_window.cancelCropButton.clicked.connect(lambda: main_window.screens.setCurrentWidget(main_window.mainScreen))

    def open_image(self):
        open_directory = self.main_window.deployments_dir
        dialog = QFileDialog(parent=self.main_window, caption="Choose Image", directory=open_directory)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("*.jpg *.jpeg *.png")
        if not dialog.exec():
            return

        abspath = dialog.selectedFiles()[0]
        self.crop_image_abspath = abspath
        pixmap = QPixmap(abspath)
        self.image_width = pixmap.width()
        self.image_height = pixmap.height()
        self.main_window.cropImageDimensionsLabel.setText(f"{self.image_width} x {self.image_height}")
        
        self.main_window.cropTopSpinBox.setMaximum(self.image_height)
        self.main_window.cropTopSpinBox.setValue(0)
        self.main_window.cropBottomSpinBox.setMaximum(self.image_height)
        self.main_window.cropBottomSpinBox.setValue(0)
        self.main_window.cropLeftSpinBox.setMaximum(self.image_width)
        self.main_window.cropLeftSpinBox.setValue(0)
        self.main_window.cropRightSpinBox.setMaximum(self.image_width)
        self.main_window.cropRightSpinBox.setValue(0)

        resized_pixmap = pixmap.scaledToWidth(CROP_PIXMAP_WIDTH, mode=Qt.TransformationMode.SmoothTransformation)
        self.scene.setSceneRect(resized_pixmap.rect().toRectF())   # will be used to limit drag range
        
        background_pixmap_item = QGraphicsPixmapItem(resized_pixmap)
        background_pixmap_item.setOpacity(0.5)
        self.scene.addItem(background_pixmap_item)

        self.crop_rect = CropRect(self, self.scene, resized_pixmap.rect().toRectF(), resized_pixmap)

    def crop_rect_changed(self):
        rect = self.crop_rect.rect()
        resize_factor = CROP_PIXMAP_WIDTH / self.image_width

        self.crop_top = round(rect.top() / resize_factor)
        self.crop_bottom = self.image_height - min(self.image_height, round(rect.bottom() / resize_factor))
        self.crop_left = round(rect.left() / resize_factor)
        self.crop_right = self.image_width - min(self.image_width, round(rect.right() / resize_factor))

        self.main_window.cropTopSpinBox.setValue(self.crop_top)
        self.main_window.cropBottomSpinBox.setValue(self.crop_bottom)
        self.main_window.cropLeftSpinBox.setValue(self.crop_left)
        self.main_window.cropRightSpinBox.setValue(self.crop_right)



class CropRect(QGraphicsRectItem):
    def __init__(self, crop_manager, scene, rectF, pixmap):
        super().__init__(rectF)

        self.crop_manager = crop_manager

        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(2)
        self.setPen(pen)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape)
        self.pixmap_item = QGraphicsPixmapItem(pixmap, self)

        self.handles = [
            CropHandle(self, for_top=True),
            CropHandle(self, for_bottom=True),
            CropHandle(self, for_left=True),
            CropHandle(self, for_right=True),
            CropHandle(self, for_top=True, for_left=True),
            CropHandle(self, for_top=True, for_right=True),
            CropHandle(self, for_bottom=True, for_left=True),
            CropHandle(self, for_bottom=True, for_right=True)
        ]

        scene.addItem(self)
        for handle in self.handles:
            scene.addItem(handle)

    # override - treat the content box rect as the shape to clip the image to, instead of the bounding rect, allows the border to show up
    def shape(self):
        path = QPainterPath()
        path.addRect(self.rect())
        return path
    
    def update_rect(self, new_rectF):
        self.setRect(new_rectF)
        for handle in self.handles:
            handle.ignore_position_changes = True
            handle.update_to_match_crop_rect()
            handle.ignore_position_changes = False
        
        self.crop_manager.crop_rect_changed()
        



class CropHandle(QGraphicsPathItem):

    def __init__(self, crop_rect, for_top=False, for_bottom=False, for_left=False, for_right=False):
        self.for_top = for_top
        self.for_bottom = for_bottom
        self.for_left = for_left
        self.for_right = for_right
        self.horizontal_movement_only = (for_left or for_right) and (not for_top) and (not for_bottom)
        self.vertical_movement_only = (for_top or for_bottom) and (not for_left) and (not for_right)

        # set path
        path = QPainterPath()
        offset = CROP_HANDLE_WIDTH/2

        if for_top and for_left:
            path.moveTo(-offset, CROP_HANDLE_SIZE)
            path.lineTo(-offset, -offset)
            path.lineTo(CROP_HANDLE_SIZE, -offset)
        elif for_top and for_right:
            path.moveTo(offset, CROP_HANDLE_SIZE)
            path.lineTo(offset, -offset)
            path.lineTo(-CROP_HANDLE_SIZE, -offset)
        elif for_bottom and for_left:
            path.moveTo(-offset, -CROP_HANDLE_SIZE)
            path.lineTo(-offset, offset)
            path.lineTo(CROP_HANDLE_SIZE, offset)
        elif for_bottom and for_right:
            path.moveTo(offset, -CROP_HANDLE_SIZE)
            path.lineTo(offset, offset)
            path.lineTo(-CROP_HANDLE_SIZE, offset)
        elif for_top:
            path.moveTo(-CROP_HANDLE_SIZE/2, -offset)
            path.lineTo(CROP_HANDLE_SIZE/2, -offset)
        elif for_bottom:
            path.moveTo(-CROP_HANDLE_SIZE/2, offset)
            path.lineTo(CROP_HANDLE_SIZE/2, offset)
        elif for_left:
            path.moveTo(-offset, -CROP_HANDLE_SIZE/2)
            path.lineTo(-offset, CROP_HANDLE_SIZE/2)
        elif for_right:
            path.moveTo(offset, -CROP_HANDLE_SIZE/2)
            path.lineTo(offset, CROP_HANDLE_SIZE/2)

        super().__init__(path)

        self.crop_rect = crop_rect

        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(CROP_HANDLE_WIDTH)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.setPen(pen)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)      
        
        self.ignore_position_changes = False    # used by crop rect when it's updating the crop handle locations

        # cursor
        if self.horizontal_movement_only:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self.vertical_movement_only:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif (for_top and for_left) or (for_bottom and for_right):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)

        # set position
        self.update_to_match_crop_rect()


    def shape(self):
        shape_rect = QRectF(-CROP_HANDLE_SIZE, -CROP_HANDLE_SIZE, 2*CROP_HANDLE_SIZE, 2*CROP_HANDLE_SIZE)
        
        if self.horizontal_movement_only:
            half_height = (self.crop_rect.rect().height() / 2) - CROP_HANDLE_SIZE
            shape_rect.setTop(-half_height)
            shape_rect.setBottom(half_height)
        
        if self.vertical_movement_only:
            half_width = (self.crop_rect.rect().width() / 2) - CROP_HANDLE_SIZE
            shape_rect.setLeft(-half_width)
            shape_rect.setRight(half_width)

        path = QPainterPath()
        path.addRect(shape_rect)
        return path
    
    
    def update_to_match_crop_rect(self):
        
        rect = self.crop_rect.rect()

        # x
        if self.vertical_movement_only:
            self.setX(rect.left() + rect.width()/2)
        elif self.for_left:
            self.setX(rect.left())
        else:
            self.setX(rect.right())

        # y
        if self.horizontal_movement_only:
            self.setY(rect.top() + rect.height()/2)
        elif self.for_top:
            self.setY(rect.top())
        else:
            self.setY(rect.bottom())

        
    
    def itemChange(self, change, value):
        if self.scene() and change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and not self.ignore_position_changes:

            if self.horizontal_movement_only:
                value.setY(self.pos().y())
            if self.vertical_movement_only:
                value.setX(self.pos().x())
            
            # keep within scene
            scene_rect = self.scene().sceneRect()
            value.setX(max(scene_rect.left(), min(scene_rect.right(), value.x())))
            value.setY(max(scene_rect.top(), min(scene_rect.bottom(), value.y())))

            rect = self.crop_rect.rect()

            # enforce min crop rect size
            max_top = rect.bottom() - MIN_CROP_RECT_HEIGHT
            min_bottom = rect.top() + MIN_CROP_RECT_HEIGHT
            max_left = rect.right() - MIN_CROP_RECT_WIDTH
            min_right = rect.left() + MIN_CROP_RECT_WIDTH
            if self.for_top:
                value.setY(min(max_top, value.y()))
            if self.for_bottom:
                value.setY(max(min_bottom, value.y()))
            if self.for_left:
                value.setX(min(max_left, value.x()))
            if self.for_right:
                value.setX(max(min_right, value.x()))


            # update crop rect
            if self.for_top and self.for_left:
                rect.setTopLeft(value)
            elif self.for_top and self.for_right:
                rect.setTopRight(value)
            elif self.for_bottom and self.for_left:
                rect.setBottomLeft(value)
            elif self.for_bottom and self.for_right:
                rect.setBottomRight(value)
            elif self.for_top:
                rect.setTop(value.y())
            elif self.for_bottom:
                rect.setBottom(value.y())
            elif self.for_left:
                rect.setLeft(value.x())
            elif self.for_right:
                rect.setRight(value.x())
            
            self.crop_rect.update_rect(rect)

            return value

        return QGraphicsItem.itemChange(self, change, value)