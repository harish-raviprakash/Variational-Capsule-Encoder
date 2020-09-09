"""
Variational Capsule Encoders - BCaps
Original Paper by Harish RaviPrakash, Syed Muhammad Anwar and Ulas Bagci
Code written by: Harish RaviPrakash
If you use significant portions of this code or the ideas from our paper, please cite it :)

This file is used for testing the reconstruction quality of the images.
"""
import numpy as np
from skimage.measure import compare_ssim as ssim
import statistics
import pandas as pd


def computeReconQuality(model_name, x_test, img_recon):
    """
    Computes Mean SSIM and Normalized MSE for the reconstructed images.
    """
    mean_ssim = []
    stddev_ssim = []
    stddev_mse = []
    mean_mse = []
    digit_size = 28
    SSIM = []
    MSE = []
    for k in range(x_test.shape[0]):
        x_decoded = img_recon[k, :]
        digit = x_decoded.reshape(digit_size, digit_size)
        digit_gt = x_test[k].reshape(digit_size, digit_size)
        SSIM.append(ssim(digit_gt, digit))
        MSE.append(np.mean(np.square(digit - digit_gt)))
    mean_ssim.append(statistics.mean(SSIM))
    stddev_ssim.append(statistics.stdev(SSIM))
    mean_mse.append(np.mean(np.asarray(MSE)))
    stddev_mse.append(np.std(np.asarray(MSE)))
    df = pd.DataFrame.from_dict({'Model': model_name, 'Mean SSIM': mean_ssim, 'Std dev SSIM': stddev_ssim,
                             'Mean NMSE': mean_mse, 'Std dev NMSE': stddev_mse})
    df.to_excel('metrics_mnist.xlsx', header=True, index=False)