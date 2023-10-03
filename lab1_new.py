# # # Basics of Machine Learning # # #

import matplotlib
import numpy as np
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
plt.switch_backend('TkAgg')         # to change backend on Tkinter library
matplotlib.use('TkAgg', force=True)


data_dir = 'data_lab1/notMNIST_large'
# data_dir = 'data_lab1/notMNIST_small'


def collect_data(data_dir):
    """ This function creates an array of images and an array of labels """
    images = []
    labels = []
    for class_label in os.listdir(data_dir):     # create labels 0:9 for folders names
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


def calculate_datasets_proportions(train_size):
    """ This function calculate proportions of every dataset """
    train = train_size / 100
    valid_and_test = 1 - train
    test = valid_and_test / 4
    valid = valid_and_test - test
    return train, valid, test


def check_and_remove_similar(dataset1, labels1, dataset2, dataset3):
    """ This function find and removes elements from dataset_1 that similar with 2 other datasets """
    dataset1_list = dataset1.tolist()
    dataset2_list = dataset2.tolist()
    dataset3_list = dataset3.tolist()
    similar_indices = []

    for i, sublist1 in enumerate(dataset1_list):
        if sublist1 in dataset2_list or sublist1 in dataset3_list:
            similar_indices.append(i)

    dataset1_filtered = np.delete(dataset1, similar_indices, axis=0)
    labels1_filtered = np.delete(labels1, similar_indices)
    numb_of_deleted = len(similar_indices)

    if numb_of_deleted != 0:
        print(f'{numb_of_deleted} similar elements were deleted')
    else:
        print('There are no similar elements')
    return dataset1_filtered, labels1_filtered


def convert_to_2D_array(dataset):
    """ This function convert input dataset with 3D dimension to 2D array """
    numb_of_elements = len(dataset)                             # number of images
    numb_of_pixels = len(dataset[0]) * len(dataset[0][0])       # size of image (28x28)
    dataset_2D = dataset.reshape(numb_of_elements, numb_of_pixels)
    return dataset_2D


def random_choice_from_2_arrays(data, labels):
    """ This function randomly chooses 10 images and 10 labels with same indices """
    rand_indices = random.choices(range(0, len(data)), k=10)
    rand_images_predict = data[rand_indices]
    rand_labels_predict = labels[rand_indices]
    return rand_images_predict, rand_labels_predict


def save_result_to_file(train_size, accuracy):
    """ This function make a dictionary of train dataset size and accuracy of prediction and save it to file """
    temp_dict = {"train_size": train_size, "accuracy": accuracy}
    file = open('files/accuracy_history.txt', mode='a')
    # for key, value in temp_dict.items():
    file.write(f'{str(temp_dict)}\n')
    file.close()


# # #   Task 1   # # #
images, labels = collect_data(data_dir)
fig, ax = plt.subplots(2, 5)
fig.suptitle('Examples of 10 random input images')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[random.choice(range(0, len(images)))], cmap='grey')       # show 10 random images
    plt.axis(False)
    plt.title(str(i + 1))
plt.show()


# # #   Task 2   # # #
classes_names, classes_counts, balance_flag = check_classes_balance(epsilon=5, labels=labels)
if balance_flag:
    print('Classes are balanced')
else:
    print('Classes are unbalanced')

fig, ax = plt.subplots(1, 1)    # show distribution on plot
fig.suptitle('Histogram of file distribution by class')
plt.bar(classes_names, classes_counts)
plt.xlabel('Classes by name'), plt.ylabel('Number of elements')
plt.show()


# # #   Task 3   # # #
# train = 94 %      valid = 5 %     test = 1 %      (calculated from input task)
print(f'Total number of images is {len(images)}')
train_dataset_size = float(input('Choose the size of train dataset in % : '))
train_proportion, valid_proportion, test_proportion = calculate_datasets_proportions(train_dataset_size)
print(f'The size of training dataset is {train_proportion * len(images)}')
print(f'The size of validating dataset is {valid_proportion * len(images)}')
print(f'The size of testing dataset is {test_proportion * len(images)}')

train_dataset, temp_dataset, train_labels, temp_labels = train_test_split(images, labels,
                                    test_size=(1 - train_proportion), random_state=42)
valid_dataset, test_dataset, valid_labels, test_labels = train_test_split(temp_dataset, temp_labels,
                                    test_size=(test_proportion / (valid_proportion + test_proportion)), random_state=42)


# # #   Task 4   # # #
train_dataset, train_labels = check_and_remove_similar(train_dataset, train_labels, valid_dataset, test_dataset)


# # #   Task 5   # # #
train_dataset = convert_to_2D_array(train_dataset)
valid_dataset = convert_to_2D_array(valid_dataset)
test_dataset = convert_to_2D_array(test_dataset)

classificator = LogisticRegression(max_iter=100)    # max_iter=100 - default value
classificator.fit(train_dataset, train_labels)      # training

valid_labels_predict = classificator.predict(valid_dataset)
accuracy = accuracy_score(valid_labels, valid_labels_predict)
print(f'The accuracy of predicting is {round(accuracy * 100, 2)} %')

rand_images_predict, rand_labels_predict = random_choice_from_2_arrays(valid_dataset, valid_labels_predict)  # choose random images to show result

fig, ax = plt.subplots(2, 5)
fig.suptitle('Examples of 10 random predicted images')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(rand_images_predict[i].reshape(28, 28), cmap='grey')       # reshape to image size (28x28)
    plt.axis(False)
    plt.title(str(rand_labels_predict[i]))
plt.show()

save_result_to_file(train_size=int(train_proportion * len(images)), accuracy=round(accuracy * 100, 2))

# Plotting the dependence of the classificator accuracy on the size of the training dataset

