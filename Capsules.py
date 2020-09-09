'''
Variational Capsule Encoders - BCaps
Original Paper by Harish RaviPrakash, Syed Muhammad Anwar and Ulas Bagci
Code written by: Harish RaviPrakash
If you use significant portions of this code or the ideas from our paper, please cite it :)
Digitcapsule layer from https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py
If you use significant portions of this code or the ideas from our paper, please cite it :)

This file contains the definitions of the various capsule layers and dynamic routing and squashing functions.
'''

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from keras.utils.conv_utils import conv_output_length, deconv_length
import numpy as np


class resizeToCaps(layers.Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, num_atoms, **kwargs):
        # self.is_placeholder = True
        super(resizeToCaps, self).__init__(**kwargs)
        self.num_atoms = num_atoms

    def call(self, inputs):
        x = K.expand_dims(inputs, axis=-1)
        y = K.tile(x, [1, 1, self.num_atoms])
        y = K.print_tensor(y, message='tiled_tensor: ')
        return y

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[1], self.num_atoms])

    def get_config(self):
        config = {
            'num_atoms': self.num_atoms
        }
        base_config = super(resizeToCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def __init__(self, num_classes, seg=True, **kwargs):
        super(Length, self).__init__(**kwargs)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes

    def call(self, inputs, **kwargs):
        if inputs.get_shape().ndims != 3:
            assert inputs.get_shape()[-2].value == 1, 'Error: Must have num_capsules = 1 going into Length'
            inputs = K.squeeze(inputs, axis=-2)
            activation = tf.norm(inputs, axis=-1, keepdims=True)
        else:
            activation = tf.norm(inputs, axis=-1)
        # activation = tf.Print(activation, [activation], message="Length: ")
        return activation

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    def __init__(self, resize_masks=False, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        masked = K.batch_flatten(K.expand_dims(mask, -1) * inputs)
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


class FlattenCapsuleLayer(layers.Layer):
    """
    Used for flattening the input from [B, H, W, C, A] to [B, H*W*C, A]
    """
    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)

        input_tensor_reshaped = K.reshape(inputs, [
            input_shape[0], self.input_num_capsule * self.input_height * self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_num_capsule * self.input_height * self.input_width,
                                         self.input_num_atoms))
        return input_tensor_reshaped

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2] * input_shape[3], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(FlattenCapsuleLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class DigiCaps(layers.Layer):
    # TODO: Change this description
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_num_atoms] and output shape = \
    [None, num_capsule, num_atoms]. For Dense Layer, input_num_atoms = num_atoms = 1.

    :param num_capsule: number of capsules in this layer
    :param num_atoms: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, num_atoms, routings=3, leaky_routing=False,
                 kernel_initializer='he_normal', **kwargs):
        super(DigiCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.routings = routings
        self.leaky_routing = leaky_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_num_atoms = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.num_atoms, self.input_num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_num_atoms]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_num_atoms]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_num_atoms]
        # W.shape=[num_capsule, input_num_capsule, num_atoms, input_num_atoms]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_num_atoms] x [num_atoms, input_num_atoms]^T -> [num_atoms].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, num_atoms]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, num_atoms]
            # The first two dimensions as `batch` dimension,
            # then matmul: [input_num_capsule] x [input_num_capsule, num_atoms] -> [num_atoms].
            # outputs.shape=[None, num_capsule, num_atoms]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmul: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.num_atoms])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'routings': self.routings,
            'leaky_routing': self.leaky_routing,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(DigiCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors
