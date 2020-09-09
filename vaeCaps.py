"""
Bayesian Capsules with modified sampling and Batch Normalization
Code written by: Harish RaviPrakash
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at harishr@knights.ucf.edu.

This file contains the network architecture for estimating aleatoric uncertainty in the data.
For epistemic uncertainty, dropout needs to be added to the network architecture.
"""

from Capsules import DigiCaps, Length, KLDivergenceLayer
import keras.backend as K
from keras import layers
from keras.models import Model
import tensorflow as tf
# from BatchNormalization import BatchCapsuleNormalization


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    mean = tf.reduce_mean(z_mean)
    std = K.std(z_log_var)
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim), mean=mean, stddev=std)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def varCapsuleAE(input_shape, latent_dim=2):
    """
        Basing the structure on the 1D VAE as in keras example code.
        :param input_shape: input dimension (image size in 1D)
        :param latent_dim: number of output classes
        :return: model
        """
    input_img = layers.Input(shape=input_shape)
    # Reshape the image to capsule
    _, C = input_img.get_shape()
    x = layers.Reshape((1, C.value))(input_img)

    # Layer 1: Digit capsule without Leaky routing
    y = DigiCaps(num_capsule=8, num_atoms=64, routings=3, name='digitcaps_1')(x)
    y_bn = layers.BatchNormalization(axis=-2, name='caps_bn')(y)

    # Layer 2: Digit capsule without Leaky routing
    cap_mean = DigiCaps(num_capsule=latent_dim, num_atoms=64, routings=3, name='digitcaps_mean')(y_bn)
    cap_var = DigiCaps(num_capsule=latent_dim, num_atoms=64, routings=3, name='digitcaps_var')(y_bn)
    cap_mean_norm = Length(num_classes=latent_dim, name='latent_mean')(cap_mean)
    cap_var_norm = Length(num_classes=latent_dim, name='latent_var')(cap_var)

    cap_mean_norm, cap_var_norm = KLDivergenceLayer(name='kl_loss')([cap_mean_norm, cap_var_norm])

    # Latent sampling - use reparameterization trick to push the sampling out as input
    z_bayes = layers.Lambda(sampling, name='reparameterization')([cap_mean_norm, cap_var_norm])

    encoder = Model(inputs=[input_img], outputs=[cap_mean_norm, cap_var_norm, z_bayes])

    # Decoder network
    latent_inputs = layers.Input(shape=(latent_dim,))
    # Fully connected layer
    fc2 = layers.Dense(units=512, activation='relu', name='fc2')(latent_inputs)
    bn = layers.BatchNormalization(name='fc_bn')(fc2)
    fc3 = layers.Dense(units=input_shape[0], activation='sigmoid', name='recon_out')(bn)

    decoder = Model(inputs=[latent_inputs], outputs=[fc3])
    outputs = decoder(encoder(input_img)[2])
    model = Model(inputs=[input_img], outputs=[outputs])
    return encoder, decoder, model

