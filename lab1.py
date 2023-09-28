# # #   Основы машинного обучения   # # #

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import os
import random
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
    """ This function """
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


def collect_samples(root_dir, count):
    """ This function collect sample by random images, where count is given """
    data = []
    for folder in os.listdir(root_dir):
        inner_folder = os.path.join(root_dir, folder)
        inner_folder_files = os.listdir(inner_folder)
        data.append(random.choices(inner_folder_files, k=count))
    return data


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

fig, ax = plt.subplots(1, 1)                # show distribution on plot
fig.suptitle('Histogram of file distribution by class')
indexes_bar = os.listdir(root_dir)
plt.bar(indexes_bar, numb_of_files_in_classes)
plt.xlabel('Classes'), plt.ylabel('Number of files')
plt.show()

if balance_flag:
    print('Classes are balanced')
else:
    print('Classes are unbalanced')


# Task 3 #
# all_files = collect_all_images(root_dir)
train_sample = collect_samples(root_dir, count=20000)
validate_sample = collect_samples(root_dir, count=1000)
test_sample = collect_samples(root_dir, count=1900)
















