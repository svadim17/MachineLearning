import numpy as np
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from sklearn.linear_model import LinearRegression
plt.switch_backend('TkAgg')         # to change backend on Tkinter library


# data_dir = 'data_lab1/notMNIST_large'
data_dir = 'data_lab1/notMNIST_small'


def collect_data(data_dir):
    """ This function creates an array of images and an array of labels """
    images = []
    labels = []
    for class_label in os.listdir(data_dir):     # create labels 0:9 for folders names
        print(class_label)
        class_dir = os.path.join(data_dir, class_label)      # directories for every class
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)    # get full path for image
            image = Image.open(image_path)
            image_array = np.array(image)
            images.append(image_array)
            labels.append(class_label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def check_classes_balance(epsilon, labels):
    """ This function counts the number of elements in every class and return balance flag
    (True - classes are balanced, False - classes are not balanced) """
    flag = True
    unique_classes, classes_counts = np.unique(labels, return_counts=True)      # return number of classes and length of every class
    for elem1 in classes_counts:
        for elem2 in classes_counts:
            if abs(elem1 - elem2) > epsilon:
                flag = False
                break
    return unique_classes, classes_counts, flag


# Task 1 #
images, labels = collect_data(data_dir)
fig, ax = plt.subplots(2, 5)
fig.suptitle('Examples of 10 random input images')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[random.choice(range(0, len(images)))], cmap='grey')       # show 10 random images
    plt.axis(False)
    plt.title(str(i + 1))
plt.show()


# Task 2 #
classes_names, classes_counts, balance_flag = check_classes_balance(epsilon=5, labels=labels)
if balance_flag:
    print('Classes are balanced')
else:
    print('Classes are unbalanced')

fig, ax = plt.subplots(1, 1)  # show distribution on plot
fig.suptitle('Histogram of file distribution by class')
plt.bar(classes_names, classes_counts)
plt.xlabel('Classes by name'), plt.ylabel('Number of elements')
plt.show()


# Task 3 #
