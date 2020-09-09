import keras.backend as K
from keras import layers
from keras.models import Model
import tensorflow as tf


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class KLDivergenceLayer(layers.Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(KLDivergenceLayer, self).get_config()
        return config


def vae(input_shape, latent_dim=2):
    inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Dense(512, activation='relu')(inputs)
    bn = layers.BatchNormalization(axis=-1, name='bn1')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(bn)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(bn)

    # Adding KL loss
    z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(512, activation='relu')(latent_inputs)
    bn2 = layers.BatchNormalization(axis=-1, name='bn2')(x)
    outputs = layers.Dense(input_shape[0], activation='sigmoid')(bn2)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    return encoder, decoder, vae