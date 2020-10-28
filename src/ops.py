import tensorflow as tf


def conv2d(input_, output_dim, k_h, k_w,
           stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', output_dim, initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
    return conv


def conv2d_same(input_, output_dim, k_h, k_w,
           stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', output_dim, initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME') + b
    return conv



def highway(x, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    with tf.variable_scope('highway'):
        for idx in range(layer_size):
            W_T = tf.get_variable("weight_transform%d" % idx, [size, size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_T = bias

            W = tf.get_variable("weight%d" % idx, [size, size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = 0.1

            T = tf.sigmoid(tf.matmul(x, W_T) + b_T)
            H = f(tf.matmul(x, W) + b)
            C = 1. - T

            y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
            x = y
    return y