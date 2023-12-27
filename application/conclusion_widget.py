import joblib
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import matplotlib
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns  # for create custom graphs
import time


class ConclusionWidget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Conclusion')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # self.data_dir = '/home/vadim/PycharmProjects/MachineLearning/datasets/notMNIST_large'
        self.dataset_dir = '/home/vadim/PycharmProjects/MachineLearning/datasets/notMNIST_small'
        self.models_dir = '/home/vadim/PycharmProjects/MachineLearning/saved_models'
        self.models_names = []
        self.models = []
        self.test_dataset_2D = None
        self.test_converted_dataset = None
        self.test_labels = None
        self.test_labels_converted = None

        self.read_models()
        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.l_spb_test_size = QLabel('Test dataset size')
        self.spb_test_size = QSpinBox()
        self.spb_test_size.setRange(10, 99)
        self.spb_test_size.setValue(60)
        self.spb_test_size.setSingleStep(5)
        self.spb_test_size.setSuffix('  %')
        self.spb_test_size.setStyleSheet("QSpinBox::up-button {height: 20px;}"
                                         "QSpinBox::down-button {height: 20px;}")
        self.spb_test_size.setFixedWidth(120)

        self.btn_collect_dataset = QPushButton('Collect test dataset')
        self.btn_collect_dataset.setFixedWidth(200)

        self.btn_check_accuracy = QPushButton('Check all Accuracies')
        self.btn_check_accuracy.setFixedWidth(220)

        self.table_accuracy = QTableWidget()
        self.table_accuracy.setRowCount(len(self.models_names))
        self.table_accuracy.setColumnCount(4)
        self.table_accuracy.setHorizontalHeaderLabels(['Model name', 'Epoch count', 'Accuracy', 'Loss'])
        for i in range(self.table_accuracy.columnCount()):
            self.table_accuracy.horizontalHeaderItem(i).setTextAlignment(Qt.AlignCenter)
        self.table_accuracy.setEditTriggers(QAbstractItemView.NoEditTriggers)  # disable items editing
        for i in range(self.table_accuracy.rowCount()):
            self.table_accuracy.setItem(i, 0, QTableWidgetItem(str(self.models_names[i])))
            self.table_accuracy.setItem(i, 1, QTableWidgetItem('100'))
        self.table_accuracy.setColumnWidth(0, 500)
        self.table_accuracy.horizontalHeader().setStretchLastSection(True)

    def add_widgets_to_layout(self):
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        spb_test_size_layout = QVBoxLayout()
        spb_test_size_layout.addWidget(self.l_spb_test_size)
        spb_test_size_layout.addWidget(self.spb_test_size)

        btn_collect_dataset_layout = QVBoxLayout()
        btn_collect_dataset_layout.addWidget(QLabel(''))
        btn_collect_dataset_layout.addWidget(self.btn_collect_dataset)

        btn_check_accuracy_layout = QVBoxLayout()
        btn_check_accuracy_layout.addWidget(QLabel(''))
        btn_check_accuracy_layout.addWidget(self.btn_check_accuracy)

        controls_layout1 = QHBoxLayout()
        controls_layout1.addSpacing(1)
        controls_layout1.addLayout(spb_test_size_layout)
        controls_layout1.addSpacing(3)
        controls_layout1.addLayout(btn_collect_dataset_layout)
        controls_layout1.addSpacing(150)
        controls_layout1.addLayout(btn_check_accuracy_layout)

        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(controls_layout1)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.table_accuracy)
        self.main_layout.addSpacing(10)

        # self.main_layout.addSpacerItem(spacerItem)

    def read_models(self):
        for model_name in os.listdir(self.models_dir):
            self.models_names.append(model_name)

    def collect_test_dataset(self):
        images, labels = [], []
        for class_label in os.listdir(self.dataset_dir):  # create labels 0:9 for folders names
            class_dir = os.path.join(self.dataset_dir, class_label)  # directories for every class
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)  # get full path for image
                try:
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    images.append(image_array)
                    labels.append(class_label)
                except:
                    print(f'Cannot identify image {image}')
        images = np.array(images)
        labels = np.array(labels)

        test_dataset_size = self.spb_test_size.value()

        temp_dataset, test_dataset, temp_labels, test_labels = train_test_split(images, labels,
                                                                                test_size=(test_dataset_size / 100),
                                                                                random_state=42)
        self.test_labels = test_labels
        test_dataset_2D = self.convert_to_2D_array(test_dataset)

        self.test_labels_converted = to_categorical(LabelEncoder().fit_transform(test_labels))
        test_dataset_normalize = test_dataset / 255.0       # normalization

        self.test_dataset_2D = test_dataset_2D
        self.test_converted_dataset = test_dataset_normalize

        print('Test dataset collected successfully! ')

    def convert_to_2D_array(self, dataset):
        """ This function convert input dataset with 3D dimension to 2D array """
        numb_of_elements = len(dataset)  # number of images
        numb_of_pixels = len(dataset[0]) * len(dataset[0][0])  # size of image (28x28)
        dataset_2D = dataset.reshape(numb_of_elements, numb_of_pixels)
        return dataset_2D

    def btn_check_accuracy_clicked(self):
        self.load_models()
        test_losses, test_accuracies = [], []
        for i in range(len(self.models)):
            if self.models_names[i] == 'LAB1_logistic_regression.joblib':
                test_labels_predict = self.models[i].predict(self.test_dataset_2D)
                # loss = log_loss(self.test_labels, test_labels_predict)
                loss = 'No info'
                accuracy = accuracy_score(self.test_labels, test_labels_predict)
            else:
                loss, accuracy = self.models[i].evaluate(self.test_converted_dataset, self.test_labels_converted)
            if type(loss) is not str:
                test_losses.append(round((loss * 100), 2))
            else:
                test_losses.append(loss)
            test_accuracies.append(round((accuracy * 100), 2))

        self.update_table(test_accuracies, test_losses)

        # print(f'losses: {test_losses}')
        # print(f'accuracies: {test_accuracies}')

    def load_models(self):
        self.models = []
        for model_name in self.models_names:
            if model_name == 'LAB1_logistic_regression.joblib':
                self.models.append(joblib.load(self.models_dir + f'/{model_name}'))
            else:
                self.models.append(load_model(self.models_dir + f'/{model_name}'))

    def update_table(self, accuracy: list, loss: list):
        for i in range(self.table_accuracy.rowCount()):
            self.table_accuracy.setItem(i, 2, QTableWidgetItem(str(accuracy[i]) + ' %'))
            self.table_accuracy.setItem(i, 3, QTableWidgetItem(str(loss[i]) + ' %'))







