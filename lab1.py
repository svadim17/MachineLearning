# # #   Основы машинного обучения   # # #

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import os
import random
plt.switch_backend('TkAgg')         # to change backend on Tkinter library


def random_select_img():
    root_dir = 'data_lab1/notMNIST_large'
    rand_images = []
    for folder in os.listdir(root_dir):
        inner_folder = os.path.join(root_dir, folder)
        inner_folder_files = os.listdir(inner_folder)
        rand_images.append(inner_folder + '/' + random.choice(inner_folder_files))
    return rand_images

# Task 1 #
fig, ax = plt.subplots(2, 5)
fig.suptitle('Input images')
selected_img = random_select_img()       # choose random images from all files
for i in range(len(selected_img)):
    selected_img[i] = plt.imread(selected_img[i])
    plt.subplot(2, 5, i + 1)
    plt.imshow(selected_img[i])
    plt.title(f'Image {i + 1}')
plt.show()
