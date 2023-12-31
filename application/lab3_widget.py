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
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns  # for create custom graphs
import time


class Lab3Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 3')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # self.data_dir = '/home/vadim/PycharmProjects/MachineLearning/datasets/notMNIST_large'
        self.data_dir = '/home/vadim/PycharmProjects/MachineLearning/datasets/notMNIST_small'
        self.model = None
        self.converted_test_dataset = None

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFontPointSize(14)
        self.log_widget.setMaximumHeight(250)
        self.log_widget.setMaximumWidth(650)
        text = 'Laboratory 3: Implementation of a convolutional neural network\n\n\n'
        text += 'Dataset: notMNIST\n\n'
        self.log_widget.setText(text)

        self.l_spb_train_size = QLabel('Train size')
        self.spb_train_size = QSpinBox()
        self.spb_train_size.setRange(10, 99)
        self.spb_train_size.setValue(60)
        self.spb_train_size.setSingleStep(5)
        self.spb_train_size.setSuffix('  %')
        self.spb_train_size.setStyleSheet("QSpinBox::up-button {height: 20px;}"
                                          "QSpinBox::down-button {height: 20px;}")

        self.l_spb_epoch_count = QLabel('Epoch count')
        self.spb_epoch_count = QSpinBox()
        self.spb_epoch_count.setRange(1, 1000)
        self.spb_epoch_count.setValue(10)
        self.spb_epoch_count.setSingleStep(1)

        self.l_cb_model_parameters = QLabel('Model parameters')
        self.cb_model_parameters = QComboBox()
        self.cb_model_parameters.addItem('Default CNN')
        self.cb_model_parameters.setItemData(0, 'default_cnn')
        self.cb_model_parameters.addItem('Enable Max Pooling')
        self.cb_model_parameters.setItemData(1, 'max_pooling')
        self.cb_model_parameters.addItem('Enable Average Pooling')
        self.cb_model_parameters.setItemData(2, 'average_pooling')
        self.cb_model_parameters.addItem('LeNet-5 architecture')
        self.cb_model_parameters.setItemData(3, 'lenet5')

        self.btn_check_prediction = QPushButton('Result')
        self.btn_check_prediction.setFixedWidth(150)

        self.tab_widget_graphs = QTabWidget()

        self.btn_start = QPushButton('Start')
        self.btn_start.setFixedHeight(33)

    def add_widgets_to_layout(self):
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        spb_train_size_layout = QVBoxLayout()
        spb_train_size_layout.addWidget(self.l_spb_train_size)
        spb_train_size_layout.addWidget(self.spb_train_size)

        spb_epoch_count_layout = QVBoxLayout()
        spb_epoch_count_layout.addWidget(self.l_spb_epoch_count)
        spb_epoch_count_layout.addWidget(self.spb_epoch_count)

        cb_model_parameteres_layout = QVBoxLayout()
        cb_model_parameteres_layout.addWidget(self.l_cb_model_parameters)
        cb_model_parameteres_layout.addWidget(self.cb_model_parameters)

        controls_layout_1 = QHBoxLayout()
        controls_layout_1.addLayout(spb_train_size_layout)
        controls_layout_1.addLayout(spb_epoch_count_layout)

        controls_layout_2 = QHBoxLayout()
        controls_layout_2.addLayout(cb_model_parameteres_layout)

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(controls_layout_1)
        controls_layout.addSpacing(20)
        controls_layout.addLayout(controls_layout_2)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.btn_check_prediction)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.log_widget)
        top_layout.addLayout(controls_layout)

        self.main_layout.addLayout(top_layout)
        # self.main_layout.addWidget(self.log_widget)
        self.main_layout.addWidget(self.tab_widget_graphs)
        self.main_layout.addWidget(self.btn_start)
        # self.main_layout.addItem(spacerItem)

    def processor(self):
        """ Main part (generating neural network) """
        start_time = time.time()

        self.log_widget.clear()
        self.tab_widget_graphs.clear()
        self.model = None

        # # # Show examples of images # # #
        images, labels = self.collect_data()
        fig1, ax1 = plt.subplots(2, 5)
        fig1.suptitle('Examples of 10 random input images')
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[random.choice(range(0, len(images)))], cmap='grey')  # show 10 random images
            plt.axis(False)
            plt.title(str(i + 1))
        self.add_graphs_to_widget(fig1)
        # plt.show()

        # # # Check balance # # #
        classes_names, classes_counts, balance_flag = self.check_classes_balance(epsilon=5, labels=labels)
        if balance_flag:
            print('Classes are balanced')
            self.log_widget.append('Classes are balanced')
        else:
            print('Classes are unbalanced')
            self.log_widget.append('Classes are unbalanced')

        fig2, ax2 = plt.subplots(1, 1)  # show distribution on plot
        fig2.suptitle('Histogram of file distribution by class')
        plt.bar(classes_names, classes_counts)
        plt.xlabel('Classes by name'), plt.ylabel('Number of elements')
        self.add_graphs_to_widget(fig2)
        # plt.show()

        # # # Splitting on train, test, valid datasets # # #
        print(f'Total number of images is {len(images)}')
        self.log_widget.append(f'Total number of images is {len(images)}')
        train_dataset_size = self.spb_train_size.value()
        train_proportion, valid_proportion, test_proportion = self.calculate_datasets_proportions(train_dataset_size)
        print(f'The size of training dataset is {int(train_proportion * len(images))}')
        print(f'The size of validating dataset is {int(valid_proportion * len(images))}')
        print(f'The size of testing dataset is {int(test_proportion * len(images))}')
        self.log_widget.append(f'The size of training dataset is {int(train_proportion * len(images))}')
        self.log_widget.append(f'The size of validating dataset is {int(valid_proportion * len(images))}')
        self.log_widget.append(f'The size of testing dataset is {int(test_proportion * len(images))}')

        # # # Converting labels # # #
        encoded_labels = LabelEncoder().fit_transform(labels)  # converting to numbers
        converted_labels = to_categorical(encoded_labels)  # converting to one-hot encoding (like a binary)

        train_dataset, temp_dataset, train_labels, temp_labels = train_test_split(images, converted_labels,
                                                                                  test_size=(1 - train_proportion),
                                                                                  random_state=42)
        valid_dataset, test_dataset, valid_labels, test_labels = train_test_split(temp_dataset, temp_labels,
                                                                                  test_size=(test_proportion / (
                                                                                          valid_proportion + test_proportion)),
                                                                                  random_state=42)

        # # # Check and remove similar elements # # #
        train_dataset, train_labels = self.check_and_remove_similar(train_dataset, train_labels, valid_dataset,
                                                                    test_dataset)

        # # # Convert to numpy array # # #
        train_dataset = np.array(train_dataset)
        test_dataset = np.array(test_dataset)
        valid_dataset = np.array(valid_dataset)

        # # # Normalization # # #
        train_dataset = train_dataset / 255.0
        test_dataset = test_dataset / 255.0
        valid_dataset = valid_dataset / 255.0

        self.converted_test_dataset = test_dataset

        # # # Define Neural Network # # #
        self.init_neural_network_model()

        # # # Model Training # # #
        history = self.model.fit(train_dataset, train_labels, epochs=self.spb_epoch_count.value(), batch_size=32,
                                 validation_data=(valid_dataset, valid_labels))

        self.create_loss_and_accuracy_graphs(history)

        test_loss, test_accuracy = self.model.evaluate(test_dataset, test_labels)

        print(f'\nTest accuracy is {round((test_accuracy * 100), 2)} %')
        self.log_widget.append(f'\nTest accuracy is {round((test_accuracy * 100), 2)} %')

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'\nExecution time: {round(execution_time, 2)} seconds')
        self.log_widget.append(f'Execution time: {round(execution_time, 2)} seconds')

        # self.model.save('/home/vadim/PycharmProjects/MachineLearning/saved_models/LAB3_CNN_lenet5.h5')

    def init_neural_network_model(self):
        """ Parameters of Conv2D(): filters - Количество фильтров (ядер), которые слой будет использовать для
                                              извлечения признаков, каждый фильтр создает карту признаков;
                                    kernel-size - Кортеж из двух целых чисел (высота и ширина окна свертки);
                                    strides - Кортеж  из двух целых чисел (вертикальное и горизонтальное перемещение
                                              окна свертки при применении свертки;
                                    padding - Строка, указывающая режим дополнения (valid или same).
                                              valid: отсутствие дополнения,
                                              same: такое дополнение, чтобы вывод имел тот же размер, что и вход;
                                    activation - Функция активации, которая будет применена после свертки;
                                    input_shape - Кортеж, представляющий размер входного изображения. """

        if self.cb_model_parameters.currentData() == 'default_cnn':
            self.model = models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(units=10, activation='softmax')
            ])
            self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        elif self.cb_model_parameters.currentData() == 'max_pooling':
            self.model = models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(units=10, activation='softmax')
            ])
            self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        elif self.cb_model_parameters.currentData() == 'average_pooling':
            self.model = models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.AveragePooling2D(pool_size=(2, 2)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                layers.AveragePooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(units=10, activation='softmax')
            ])
            self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        elif self.cb_model_parameters.currentData() == 'lenet5':
            self.model = models.Sequential([
                layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.AveragePooling2D(),
                layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
                layers.AveragePooling2D(),
                layers.Flatten(),
                layers.Dense(units=120, activation='relu'),
                layers.Dense(units=84, activation='relu'),
                layers.Dense(units=10, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def collect_data(self):
        """ This function creates an array of images and an array of labels """
        images = []
        labels = []
        for class_label in os.listdir(self.data_dir):  # create labels 0:9 for folders names
            class_dir = os.path.join(self.data_dir, class_label)  # directories for every class
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
        return images, labels

    def check_classes_balance(self, epsilon, labels):
        """ This function counts the number of elements in every class and return balance flag
        (True - classes are balanced, False - classes are not balanced) """
        flag = True
        unique_classes, classes_counts = np.unique(labels,
                                                   return_counts=True)  # return number of classes and length of every class
        for elem1 in classes_counts:
            for elem2 in classes_counts:
                if abs(elem1 - elem2) > epsilon:
                    flag = False
                    break
        return unique_classes, classes_counts, flag

    def calculate_datasets_proportions(self, train_size):
        """ This function calculate proportions of every dataset """
        train = train_size / 100
        valid_and_test = 1 - train
        test = valid_and_test / 2
        valid = valid_and_test - test
        return train, valid, test

    def check_and_remove_similar(self, dataset1, labels1, dataset2, dataset3):
        """ This function find and removes elements from dataset_1 that similar with 2 other datasets """
        dataset1_list = dataset1.tolist()
        dataset2_list = dataset2.tolist()
        dataset3_list = dataset3.tolist()
        similar_indices = []

        for i, sublist1 in enumerate(dataset1_list):
            if sublist1 in dataset2_list or sublist1 in dataset3_list:
                similar_indices.append(i)

        dataset1_filtered = np.delete(dataset1, similar_indices, axis=0)
        labels1_filtered = np.delete(labels1, similar_indices, axis=0)
        numb_of_deleted = len(similar_indices)

        if numb_of_deleted != 0:
            print(f'{numb_of_deleted} similar elements were deleted')
            self.log_widget.append(f'{numb_of_deleted} similar elements were deleted')
        else:
            print('There are no similar elements')
            self.log_widget.append('There are no similar elements')
        return dataset1_filtered, labels1_filtered

    def add_graphs_to_widget(self, fig):
        """ This function adds graph to widget """
        canvas = FigureCanvas(fig)
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(canvas)
        self.tab_widget_graphs.addTab(tab, f'Graph {len(self.tab_widget_graphs) + 1}')

    def btn_check_prediction_clicked(self):
        """ This function chooses 10 random images from test dataset, predicts labels and
        shows input image and predicted label on graph """
        test_dataset_temp_index = np.random.choice(len(self.converted_test_dataset), size=10)   # firstly choose indexes
                                                                                                # because numpy needs
                                                                                                # 1-D array
        test_dataset_temp = self.converted_test_dataset[test_dataset_temp_index]
        predictions = self.model.predict(test_dataset_temp)
        predictions_numbers = np.argmax(predictions, axis=1)
        predictions_letters = self.convert_predictions_to_letters(predictions_numbers)

        fig, ax = plt.subplots(2, 5)
        fig.suptitle('Examples of predictions of 10 random input images')
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(test_dataset_temp[i], cmap='grey')  # show 10 random images
            plt.axis(False)
            plt.title(str(predictions_letters[i]))
        self.add_graphs_to_widget(fig)

    def convert_predictions_to_letters(self, predictions):
        """ This function converts predictions in numbers to predictions in letters by map """
        letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
        predictions_letters = [letter_map[number] for number in predictions]
        return predictions_letters

    def create_loss_and_accuracy_graphs(self, history):
        train_loss = history.history['loss']
        train_accuracy = history.history['accuracy']
        valid_loss = history.history['val_loss']
        valid_accuracy = history.history['val_accuracy']
        epochs = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Training and Validation Loss')
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, valid_loss, label='Validation Loss')
        plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.legend()
        self.add_graphs_to_widget(fig)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Training and Validation Accuracy')
        plt.plot(epochs, train_accuracy, label='Training Accuracy')
        plt.plot(epochs, valid_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs'), plt.ylabel('Accuracy'), plt.legend()
        self.add_graphs_to_widget(fig)