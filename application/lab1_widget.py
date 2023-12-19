from PyQt5.QtWidgets import *
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Lab1Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 1')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # self.data_dir = '~/PycharmProjects/MachineLearning/datasets/notMNIST_large'
        self.data_dir = '/home/vadim/PycharmProjects/MachineLearning/datasets/notMNIST_small'

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFontPointSize(14)
        self.log_widget.setMaximumHeight(250)
        text = 'Laboratory 1: Basics of Machine Learning\n\n\n'
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

        self.tab_widget_graphs = QTabWidget()

        self.btn_start = QPushButton('Start')
        self.btn_start.setFixedHeight(33)

    def add_widgets_to_layout(self):
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        spb_train_size_layout = QVBoxLayout()
        spb_train_size_layout.addWidget(self.l_spb_train_size)
        spb_train_size_layout.addWidget(self.spb_train_size)
        spb_train_size_layout.addSpacerItem(spacerItem)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.log_widget)
        top_layout.addLayout(spb_train_size_layout)

        self.main_layout.addLayout(top_layout)
        # self.main_layout.addWidget(self.log_widget)
        self.main_layout.addWidget(self.tab_widget_graphs)
        self.main_layout.addWidget(self.btn_start)
        # self.main_layout.addItem(spacerItem)

    def processor(self):
        self.log_widget.clear()
        self.tab_widget_graphs.clear()

        # # #   Task 1   # # #
        images, labels = self.collect_data()
        fig1, ax1 = plt.subplots(2, 5)
        fig1.suptitle('Examples of 10 random input images')
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[random.choice(range(0, len(images)))], cmap='grey')  # show 10 random images
            plt.axis(False)
            plt.title(str(i + 1))
        # plt.show()

        # # #   Task 2   # # #
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
        # plt.show()

        # # #   Task 3   # # #
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

        train_dataset, temp_dataset, train_labels, temp_labels = train_test_split(images, labels,
                                                                                  test_size=(1 - train_proportion),
                                                                                  random_state=42)
        valid_dataset, test_dataset, valid_labels, test_labels = train_test_split(temp_dataset, temp_labels,
                                                                                  test_size=(test_proportion / (
                                                                                              valid_proportion + test_proportion)),
                                                                                  random_state=42)

        # # #   Task 4   # # #
        train_dataset, train_labels = self.check_and_remove_similar(train_dataset, train_labels, valid_dataset, test_dataset)

        # # #   Task 5   # # #
        train_dataset = self.convert_to_2D_array(train_dataset)
        valid_dataset = self.convert_to_2D_array(valid_dataset)
        test_dataset = self.convert_to_2D_array(test_dataset)

        classificator = LogisticRegression(max_iter=100)  # max_iter=100 - default value
        classificator.fit(train_dataset, train_labels)  # training

        valid_labels_predict = classificator.predict(valid_dataset)
        accuracy = accuracy_score(valid_labels, valid_labels_predict)
        print(f'The accuracy of predicting is {round(accuracy * 100, 2)} %')
        self.log_widget.append(f'The accuracy of predicting is {round(accuracy * 100, 2)} %')

        rand_images_predict, rand_labels_predict = self.random_choice_from_2_arrays(valid_dataset,
                                                                               valid_labels_predict)  # choose random images to show result

        fig3, ax3 = plt.subplots(2, 5)
        fig3.suptitle('Examples of 10 random predicted images')
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(rand_images_predict[i].reshape(28, 28), cmap='grey')  # reshape to image size (28x28)
            plt.axis(False)
            plt.title(str(rand_labels_predict[i]))
        # plt.show()

        # save_result_to_file(train_size=int(train_proportion * len(images)), accuracy=round(accuracy * 100, 2),
        #                     file_dir='files/accuracy_history_small.txt')

        # Plotting the dependence of the classificator accuracy on the size of the training dataset
        train_sizes, accuracies = self.read_data_from_file(file_dir='/home/vadim/PycharmProjects/'
                                                                    'MachineLearning/files/accuracy_history_small.txt')
        x_axis = np.array(train_sizes)
        y_axis = np.array(accuracies)

        fig4, ax4 = plt.subplots(1, 1)  # show distribution on plot
        fig4.suptitle('The dependence of classificator accuracy on the size of the training dataset')
        plt.plot(x_axis, y_axis)
        plt.xlabel('The size of the training dataset, count'), plt.ylabel('Resulting accuracy, %')
        # plt.show()

        self.add_graphs_to_widget(fig1, fig2, fig3, fig4)

    def collect_data(self):
        """ This function creates an array of images and an array of labels """
        images = []
        labels = []
        for class_label in os.listdir(self.data_dir):  # create labels 0:9 for folders names
            class_dir = os.path.join(self.data_dir, class_label)  # directories for every class
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)  # get full path for image
                image = Image.open(image_path)
                image_array = np.array(image)
                images.append(image_array)
                labels.append(class_label)
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
        labels1_filtered = np.delete(labels1, similar_indices,  axis=0)
        numb_of_deleted = len(similar_indices)

        if numb_of_deleted != 0:
            print(f'{numb_of_deleted} similar elements were deleted')
            self.log_widget.append(f'{numb_of_deleted} similar elements were deleted')
        else:
            print('There are no similar elements')
            self.log_widget.append('There are no similar elements')
        return dataset1_filtered, labels1_filtered

    def convert_to_2D_array(self, dataset):
        """ This function convert input dataset with 3D dimension to 2D array """
        numb_of_elements = len(dataset)  # number of images
        numb_of_pixels = len(dataset[0]) * len(dataset[0][0])  # size of image (28x28)
        dataset_2D = dataset.reshape(numb_of_elements, numb_of_pixels)
        return dataset_2D

    def random_choice_from_2_arrays(self, data, labels):
        """ This function randomly chooses 10 images and 10 labels with same indices """
        rand_indices = random.choices(range(0, len(data)), k=10)
        rand_images_predict = data[rand_indices]
        rand_labels_predict = labels[rand_indices]
        return rand_images_predict, rand_labels_predict

    def save_result_to_file(self, train_size, accuracy, file_dir):
        """ This function make a dictionary of train dataset size and accuracy of prediction and save it to file """
        temp_dict = {"train_size": train_size, "accuracy": accuracy}
        file = open(file_dir, mode='a')
        # for key, value in temp_dict.items():
        file.write(f'{str(temp_dict)}\n')
        file.close()

    def read_data_from_file(self, file_dir):
        """ This function read a dictionary from file and return
        a list of size of training datasets and a list of accuracies """
        train_size_file = []
        accuracy_file = []
        with open(file_dir, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            line = eval(lines[i])
            train_size_file.append(line['train_size'])
            accuracy_file.append(line['accuracy'])

        return train_size_file, accuracy_file

    def add_graphs_to_widget(self, fig1, fig2, fig3, fig4):
        canvas1 = FigureCanvas(fig1)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.addWidget(canvas1)
        self.tab_widget_graphs.addTab(tab1, 'Graph 1')

        canvas2 = FigureCanvas(fig2)
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.addWidget(canvas2)
        self.tab_widget_graphs.addTab(tab2, 'Graph 2')

        canvas3 = FigureCanvas(fig3)
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.addWidget(canvas3)
        self.tab_widget_graphs.addTab(tab3, 'Graph 3')

        canvas4 = FigureCanvas(fig4)
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.addWidget(canvas4)
        self.tab_widget_graphs.addTab(tab4, 'Graph 4')
