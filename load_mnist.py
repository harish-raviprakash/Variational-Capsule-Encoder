'''
Variational Capsule Encoders - BCaps
Original Paper by Harish RaviPrakash, Syed Muhammad Anwar and Ulas Bagci
Code written by: Harish RaviPrakash
If you use significant portions of this code or the ideas from our paper, please cite it :)

This file is used for loading training, and testing data into the models.
It is specifically designed to handle 2D single-channel data.
Modifications will be needed to train/test on normal 3-channel images.
'''

from __future__ import print_function

import threading
from os.path import join, basename, isfile
from os import mkdir, chdir
import glob
import csv
import SimpleITK as sitk
import numpy as np
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from keras.utils import to_categorical
debug = 1


def load_data():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist, fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def load_data_1D():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist, fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)
