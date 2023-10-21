import sys
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem
from PyQt5.QtGui import QBrush, QPen, QPainter, QPixmap, QPolygonF
from PyQt5.QtCore import Qt, QPointF
import torch


app = QApplication(sys.argv)

# scene 400x200, origin at 0,0
scene = QGraphicsScene(0,0,400,200)

# draw rect
rect = QGraphicsRectItem(0,0,200,50)
rect.setPos(50,20)
brush = QBrush(Qt.GlobalColor.red)  # fill
rect.setBrush(brush)
pen = QPen(Qt.GlobalColor.cyan) # stroke
pen.setWidth(10)
rect.setPen(pen)
# rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
scene.addItem(rect)

# rect.mousePressEvent = lambda e: print(e)


scene.addPolygon(
    QPolygonF(
        [
            QPointF(30, 60),
            QPointF(270, 40),
            QPointF(400, 200),
            QPointF(20, 150),
        ]),
    QPen(Qt.GlobalColor.darkGreen),
)


# pixmap_item = scene.addPixmap(QPixmap("C:/Users/AdamK/Documents/Birds/images/birdhouse.png"))
# pixmap_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
# print(pixmap_item.boundingRect())
# pixmap_item.mousePressEvent = lambda e: print(e)

pixmap_item = QGraphicsPixmapItem(QPixmap("C:/Users/AdamK/Documents/Birds/images/birdhouse.png"))
scene.addItem(pixmap_item)
pixmap_item.mousePressEvent = lambda e: print(e)


view = QGraphicsView(scene)
view.setRenderHint(QPainter.RenderHint.Antialiasing)
view.show()
app.exec()