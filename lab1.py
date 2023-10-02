# # #   Основы машинного обучения   # # #

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import os
import random
from sklearn.linear_model import LinearRegression
plt.switch_backend('TkAgg')         # to change backend on Tkinter library


def random_select_img(root_dir):
    """ This function random choices one image from every folder """
    rand_images = []
    for folder in os.listdir(root_dir):
        inner_folder = os.path.join(root_dir, folder)
        inner_folder_files = os.listdir(inner_folder)
        rand_images.append(inner_folder + '/' + random.choice(inner_folder_files))
    return rand_images


def check_classes_balance(root_dir, epsilon):
    """ This function counts number of files in each class and compares this values with each other.
    If the difference is greater than epsilon, it means that classes are not balanced"""
    numb_of_files_in_classes = []
    for folder in os.listdir(root_dir):
        inner_folder = os.path.join(root_dir, folder)
        inner_folder_files = os.listdir(inner_folder)
        numb_of_files_in_classes.append(len(inner_folder_files))

    balance_flag = True
    for i in numb_of_files_in_classes:
        first_elem = i
        for element in numb_of_files_in_classes:
            if abs(first_elem - element) > epsilon:
                balance_flag = False
                break
    return numb_of_files_in_classes, balance_flag


# def collect_all_images(root_dir):
#     """ This function collect all images in one list """
#     all_files = []
#     for folder in os.listdir(root_dir):
#         inner_folder = os.path.join(root_dir, folder)
#         inner_folder_files = os.listdir(inner_folder)
#         [all_files.append(elem) for elem in inner_folder_files]
#     return all_files


def collect_samples(root_dir, train_count, validate_count, test_count):
    """ This function first collect train, validate and test samples from one folder.
    Then with another function find similar between samples and removes them.
    Function collects parts of data from all inner folders and packs it in 3 samples """
    train_sample, validate_sample, test_sample = [], [], []
    for folder in os.listdir(root_dir):
        inner_folder = os.path.join(root_dir, folder)
        inner_folder_files = os.listdir(inner_folder)
        train_part = random.choices(inner_folder_files, k=train_count)
        validate_part = random.choices(inner_folder_files, k=validate_count)
        test_part = random.choices(inner_folder_files, k=test_count)

        # check similar elements and remove them
        status, train_part_removed, validate_part_removed = remove_similar_elements(train_part, validate_part, test_part)
        while status:
            train_part = train_part_removed + random.choices(inner_folder_files, k=(train_count - len(train_part_removed)))     # add missing elements
            validate_part = validate_part_removed + random.choices(inner_folder_files, k=(validate_count - len(validate_part_removed)))    # add missing elements
            # again find and remove similar elements if they exist
            status, train_part_removed, validate_part_removed = remove_similar_elements(train_part, validate_part, test_part)

        train_sample.append(train_part)
        validate_sample.append(validate_part)
        test_sample.append(test_part)

    train_sample = list_of_lists_to_list(input_list=train_sample)
    validate_sample = list_of_lists_to_list(input_list=validate_sample)
    test_sample = list_of_lists_to_list(input_list=test_sample)

    return train_sample, validate_sample, test_sample


def remove_similar_elements(train_part, validate_part, test_part):
    """ This function makes sets from lists, find similar elements and removes them"""
    train_set = set(train_part)
    validate_set = set(validate_part)
    test_set = set(test_part)
    intersect_12 = train_set.intersection(validate_set)
    intersect_13 = train_set.intersection(test_set)
    intersect_23 = validate_set.intersection(test_set)

    if intersect_12 or intersect_13 or intersect_23:                        # do it if not empty
        train_part = list(train_set - intersect_12 - intersect_13)
        validate_part = list(validate_set - intersect_23)
        status = True       # status = True if similar elements between samples were removed
    else:
        status = False      # status = False if there are no similar elements between samples

    return status, train_part, validate_part


def list_of_lists_to_list(input_list):
    """ This function convert a list of lists to one list """
    out_list = []
    for inner_list in input_list:
        for elem in inner_list:
            out_list.append(elem)
    return out_list


root_dir = 'data_lab1/notMNIST_large'


# Task 1 #
fig, ax = plt.subplots(2, 5)
fig.suptitle('Input images')
selected_img = random_select_img(root_dir)       # choose random images from all files
for i in range(len(selected_img)):
    selected_img[i] = plt.imread(selected_img[i])
    plt.subplot(2, 5, i + 1)
    plt.imshow(selected_img[i], cmap='grey')
    plt.title(f'Image {i + 1}')
plt.show()


# Task 2 #
epsilon = 5     # epsilon for check classes balance
numb_of_files_in_classes, balance_flag = check_classes_balance(root_dir, epsilon)

if balance_flag:
    print('Classes are balanced')
else:
    print('Classes are unbalanced')

fig, ax = plt.subplots(1, 1)  # show distribution on plot
fig.suptitle('Histogram of file distribution by class')
indexes_bar = os.listdir(root_dir)
plt.bar(indexes_bar, numb_of_files_in_classes)
plt.xlabel('Classes'), plt.ylabel('Number of files')
plt.show()


# Task 3 and Task 4 #
train_sample, validate_sample, test_sample = collect_samples(root_dir,
                                                             train_count=20000, validate_count=1000, test_count=1900)
print(train_sample[0:3])


# Task 5 #
log_reg_classificator = LinearRegression()

















