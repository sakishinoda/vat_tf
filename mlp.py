import tensorflow as tf
import numpy

import layers as L

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('layer_sizes', '1200-600-300-150-10', "layer sizes")
tf.app.flags.DEFINE_float('noise_stddev', 0.5, "")

def gaussian_noise_layer(x, stddev, seed=None):
    return x + tf.random_normal(shape=x.get_shape(), stddev=stddev, seed=seed)

def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    x = tf.reshape(x, [x.get_shape().as_list()[0], -1])
    layer_sizes = numpy.asarray(FLAGS.layer_sizes.split('-'), numpy.int32)
    num_layers = len(layer_sizes) - 1
    rng = numpy.random.RandomState(seed)
    h = x
    for l, dim in enumerate(layer_sizes):
        inp_dim = h.get_shape()[1]
        with tf.variable_scope(str(l)):
            W = tf.get_variable(
                'W',
                shape=[inp_dim, dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                 seed=rng.randint(123456),
                                                                 dtype=tf.float32))
            b = tf.get_variable(
                'b',
                shape=[dim],
                initializer=tf.constant_initializer(0.0))
            h = tf.nn.xw_plus_b(h, W, b)
            h = L.bn(h, dim, is_training=is_training,
                     update_batch_stats=update_batch_stats)

            if l < num_layers - 1:
                h = tf.nn.relu(h)
                h = gaussian_noise_layer(
                    h,
                    stddev=FLAGS.noise_stddev,
                    seed=rng.randint(123456)) if FLAGS.noise_stddev > 0 and stochastic else h
    return h

