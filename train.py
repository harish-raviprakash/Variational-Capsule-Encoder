"""
Variational Capsule Encoders - BCaps
Original Paper by Harish RaviPrakash, Syed Muhammad Anwar and Ulas Bagci
Code written by: Harish RaviPrakash
If you use significant portions of this code or the ideas from our paper, please cite it :)

This file is used for training models. Please see the README for details about training.
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os import makedirs
from os.path import join
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf


def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def get_callbacks(arguments):
    monitor_name = 'val_loss'

    csv_logger = CSVLogger(join(arguments.log_dir, arguments.net + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, batch_size=arguments.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(join(arguments.check_dir, arguments.net + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=False, save_weights_only=False,
                                       verbose=1, mode='min', period=1)
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=10, verbose=1, mode='min')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=35, verbose=0, mode='min')
    lr_decay = LearningRateScheduler(schedule=lambda epoch: arguments.initial_lr * (arguments.lr_decay ** epoch))

    return [csv_logger, model_checkpoint]


def get_loss(loss_weight):
    losses = ['mse']
    loss_weights = [loss_weight]
    return losses, loss_weights


def compile_model(args, uncomp_model, loss_weight):
    # Set optimizer loss and metrics
    try:
        opt = Adam(lr=args.initial_lr, amsgrad=True)
    except:
        opt = Adam(lr=args.initial_lr)

    # Get loss and weighting
    loss, loss_weighting = get_loss(loss_weight)
    # If using CPU or single GPU
    if args.gpus <= 1:
        uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=['accuracy'])
        return uncomp_model
    # If using multiple GPUs
    else:
        from keras.utils import multi_gpu_model
        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=['accuracy'])
            model = multi_gpu_model(uncomp_model, gpus=args.gpus)
            model.__setattr__('callback_model', uncomp_model)
        model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=['accuracy'])
        return model


def plot_training(training_history, arguments):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    ax1.plot(training_history.history['acc'])
    ax1.plot(training_history.history['val_acc'])
    ax1.set_title('Reconstruction')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(0, len(training_history.history['acc'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax2.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(arguments.output_dir, arguments.net + '_plots_' + arguments.time + '.png'))
    plt.close()


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    makedirs(model_name)

    sp = StratifiedShuffleSplit(n_splits=1, test_size=0.09, random_state=42)
    for tr_ind, test_ind in sp.split(x_test, y_test):
        print(test_ind)
    img_recon = decoder.predict(encoder.predict(x_test[test_ind], batch_size=batch_size)[2])
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    figure2 = np.zeros((digit_size * n, digit_size * n))
    i = 0
    j = 0
    for k, ind in enumerate(test_ind):
        x_decoded = img_recon[k]
        digit = x_decoded.reshape((digit_size, digit_size))
        digit_gt = x_test[ind].reshape((digit_size, digit_size))
        if k % n == 0 and k > 0:
            i += 1
            j = 0
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit
        figure2[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit_gt
        j += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    filename = join(model_name, "digits_recon.png")
    plt.savefig(filename)
    plt.imshow(figure2, cmap='Greys_r')
    filename = join(model_name, "digits_gt.png")
    plt.savefig(filename)
    # plt.show()


def train(args, u_model, encoder, decoder, data):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # Compile the loaded model
    model = compile_model(args=args, uncomp_model=u_model, loss_weight=x_train.shape[1])

    # Set the callbacks
    callbacks = get_callbacks(args)

    if args.aug_data:
        def train_generator(x, y, batch_size, shift_fraction=0.):
            train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                               height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while 1:
                x_batch, y_batch = generator.next()
                yield (x_batch, x_batch)

        history = model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                                      steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                                      epochs=100,
                                      validation_data=[x_test, x_test], callbacks=callbacks)

    else:
        history = model.fit(x_train, x_train, batch_size=args.batch_size, epochs=args.epochs,
                            validation_data=[x_test, x_test], callbacks=callbacks)
    plot_training(history, args)
    plot_results((encoder, decoder), (x_test, y_test), batch_size=args.batch_size, model_name=args.net)
