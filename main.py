'''
Variational Capsule Encoders - BCaps
Original Paper by Harish RaviPrakash, Syed Muhammad Anwar and Ulas Bagci
Code written by: Harish RaviPrakash
If you use significant portions of this code or the ideas from our paper, please cite it :)

This is the main file for the project. From here you can train and test the model.
Please see the README for detailed instructions for this project.
'''

from __future__ import print_function
import os
from os.path import join, exists
from os import makedirs, environ
import argparse
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
import numpy as np
from vaeCaps import varCapsuleAE
from keras.utils import plot_model
from train import train
from load_mnist import load_data_1D
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd
from test import computeReconQuality


def main(args):
    args.time = time
    if not exists(args.save_dir):
        makedirs(args.save_dir)

    # Create all the output directories
    args.check_dir = join(args.save_dir, 'saved_models', args.net)
    try:
        makedirs(args.check_dir)
    except:
        pass

    args.log_dir = join(args.save_dir, 'logs', args.net)
    try:
        makedirs(args.log_dir)
    except:
        pass

    args.tf_log_dir = join(args.save_dir, 'tf_logs', args.time)
    try:
        makedirs(args.tf_log_dir)
    except:
        pass

    args.output_dir = join(args.save_dir, 'plots', args.net)
    try:
        makedirs(args.output_dir)
    except:
        pass
    (x_train, y_train), (x_test, y_test) = load_data_1D()
    writer = pd.ExcelWriter('classification_scores_mnist.xlsx', engine='xlsxwriter')
    clf = svm.SVC(C=100, kernel='rbf', gamma=0.01)  # MNIST
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_excel(writer, sheet_name='original')
    # define model
    encoder, decoder, model = varCapsuleAE(input_shape=x_train.shape[1:], latent_dim=2)
    print(model.summary())
    plot_model(model, to_file='vaeCaps.png', show_shapes=True)
    plot_model(encoder, to_file='vaeCaps_encoder.png', show_shapes=True)
    plot_model(decoder, to_file='vaeCaps_decoder.png', show_shapes=True)
    # train model
    train(args=args, u_model=model, encoder=encoder, decoder=decoder, data=((x_train, y_train), (x_test, y_test)))
    img_recon = model.predict(x_test, batch_size=args.batch_size)
    y_pred2 = clf.predict(img_recon)
    report2 = classification_report(y_test, y_pred2, output_dict=True)
    df = pd.DataFrame(report2).transpose()
    df.to_excel(writer, sheet_name=args.net)
    print('Classification report: ', report2)
    writer.save()
    writer.close()
    computeReconQuality(args.net, x_test, img_recon)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Capsules')
    parser.add_argument('--save_dir', type=str, default='/path/to/store/results',
                        help='The root directory where your datasets are stored.')
    parser.add_argument('--net', type=str.lower, default='mnistcapsules2', help='Define network name.')
    parser.add_argument('--aug_data', type=int, default=0, choices=[0, 1],
                        help='Whether or not to use data augmentation during training.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training/testing.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--recon_wei', type=float, default=0.392,
                        help="If using capsnet: The coefficient (weighting) for the loss of decoder")
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')

    arguments = parser.parse_args()

    # Mask the GPUs for TensorFlow
    if arguments.which_gpus == -2:
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                  'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)
