# # #   Основы машинного обучения   # # #

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import os
import random
plt.switch_backend('TkAgg')         # to change backend on Tkinter library


def random_select_img(numb_of_img):
    root_dir = 'data_lab1/notMNIST_large'
    all_files = []
    for folder in os.listdir(root_dir):
        inner_folder = os.path.join(root_dir, folder)
        inner_folder_files = os.listdir(inner_folder)
        for i in range(len(inner_folder_files)):
            all_files.append(inner_folder_files[i])
    return random.choices(all_files, k=numb_of_img)


numb_of_img = 4
selected_img = [None] * numb_of_img

fig, ax = plt.subplots(1, 4)
fig.suptitle('Input images')

selected_img = random_select_img(numb_of_img)       # choose random images from all files

for i in range(numb_of_img):
    ax[i].title.set_text(f'Image {i + 1}')
    ax[i].imshow(selected_img[i])
plt.show()

# img1 = plt.imread('data_lab1/notMNIST_large/J/a2F6b28udHRm.png')