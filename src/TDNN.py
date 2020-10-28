import tensorflow as tf
from ops import conv2d
from base import Model

class TDNN(Model):

    def __init__(self, input_, embed_dim = 15,
                 feature_maps = [200, 200, 200],
                 kernels = [1, 2, 3]):
        self.embed_dim = embed_dim
        self.feature_maps = feature_maps
        self.kernels = kernels

        # length = self.__length(input)
        input_ = tf.expand_dims(input_, -1)
        layers = []

        for idx, kernel_dim in enumerate(kernels):
            conv = conv2d(input_, feature_maps[idx], kernel_dim, self.embed_dim, name="kernel%d" % idx)
            pool = tf.reduce_max(tf.tanh(conv), axis=1, keep_dims=True)

            layers.append(tf.reshape(pool, [-1, feature_maps[idx]]))

        if len(kernels) > 1:
            self.output = tf.concat(layers, 1)
        else:
            self.output = layers[0]

    def __length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
