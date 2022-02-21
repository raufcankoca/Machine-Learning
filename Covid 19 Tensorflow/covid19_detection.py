import os
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5 import QtCore
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np

model=load_model("covid19_model.h5")


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        #self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, False)
        self.button1=QPushButton("Browse",self)
        self.button2=QPushButton("Classify",self)
        self.label = QLabel("                Select an image", self)
        self.label_pred=QLabel("Prediction",self)
        self.setWindowTitle("Covid-19 Detection")



        self.button1.move(30,100)
        self.button2.move(30,220)
        self.button1.resize(200,100)
        self.button2.resize(200, 100)
        self.label.move(265,100)
        self.label.resize(224,224)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label_pred.resize(250,50)
        self.label_pred.move(160,400)



        self.button1.clicked.connect(self.browse)
        self.button2.clicked.connect(self.classify)
        self.setGeometry(600,600,500,500)
        self.show()

    def browse(self):
        file_name=QFileDialog.getOpenFileName(self,"Open File",os.getenv("HOME"),'Image files (*.jpeg *.gif)')

        img_path=file_name[0]
        print(img_path)
        img=cv2.imread(img_path)
        img=cv2.resize(img,(224,224))
        cv2.imwrite("sized.jpg",img)
        img_sized_path=os.path.realpath("sized.jpg")

        pixmap=QPixmap(img_sized_path)
        print(img_path)

        self.label.setPixmap(QPixmap(pixmap))


    def classify(self):
        image=cv2.imread("sized.jpg")
        image=np.reshape(image,(1,224,224,3))
        image=image/255.0
        image_pred=model.predict(image)[0][0]

        if image_pred>0.5:
            prediction="Covid Positive ={}".format(image_pred)
            self.label_pred.setStyleSheet("background-color: red;")
            self.label_pred.setFont(QFont("Arial font", 10))
        else:
            prediction="Covid Negative ={}".format(image_pred)
            self.label_pred.setStyleSheet("background-color: lightgreen;")
            self.label_pred.setFont(QFont("Arial font",10))

        self.label_pred.setText(str(prediction))

app=QApplication(sys.argv)
window=Window()
sys.exit(app.exec_())
