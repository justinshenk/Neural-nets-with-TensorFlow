import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import minibatches, one_hot, load_mnist_data
from progressbar import * # sudo pip3 install progressbar33
from network_builder import *

RAND_SEED = 0

np.random.seed(RAND_SEED)

data_train, labels_train, data_test, labels_test, data_validation, labels_validation = load_mnist_data()
batches = minibatches(data_train, labels_train, batch_size=32, loop=True)
def next_batch():
    return next(batches)

def batch_normalize(batch):
    mean, variance = tf.nn.moments(batch, [0])
    normalized = tf.nn.batch_normalization(batch, mean, variance,
            None, None, 1e-10) # what epsilon do I use?
    return normalized


if __name__ == "__main__":

    batch_size = 32
    n_epochs   = 100

    tf.set_random_seed(RAND_SEED)
    tf.reset_default_graph()

    ############################################################################
    #                                Generator                                 #
    ############################################################################
    # Input layer
    inputs = tf.placeholder(tf.float32, shape=[None, 100])
    batch_size = tf.shape(inputs)[0]
    
    with tf.variable_scope("gen_ff1"):
        ff1 = feedForwardLayer(inputs, 100, 512 * 4 * 4, None)
        ff1_reshaped = tf.reshape(ff1, [100, 4, 4, 512])
        ff1_normalized = batch_normalize(ff1_reshaped)
        ff1_out = tf.nn.relu(ff1_normalized)
    with tf.variable_scope("gen_conv1"):
        filter1 = tf.get_variable("filter", [5, 5, 256, 512],
                initializer=tf.truncated_normal_initializer(stddev=.1, seed=RAND_SEED))
        bias1 = tf.get_variable("bias", [256],
                initializer=tf.constant_initializer(.1))
        conv1 = tf.nn.conv2d_transpose(ff1_out, filter1, tf.pack([batch_size, 7, 7, 256]), strides=[1,2,2,1], padding='SAME')
        conv1_normalized = batch_normalize(conv1)
        conv1_out = tf.nn.relu(conv1_normalized + bias1)

    with tf.variable_scope("gen_conv2"):
        filter2 = tf.get_variable("filter", [5, 5, 128, 256],
                initializer=tf.truncated_normal_initializer(stddev=.1, seed=RAND_SEED))
        bias2 = tf.get_variable("bias", [128], initializer=tf.constant_initializer(.1))
        conv2 = tf.nn.conv2d_transpose(conv1_out, filter2, tf.pack([batch_size, 14, 14, 128]), strides=[1,2,2,1], padding='SAME')
        conv2_normalized = batch_normalize(conv2)
        conv2_out = tf.nn.relu(conv2_normalized + bias2)

    with tf.variable_scope("gen_conv3"):
        filter3 = tf.get_variable("filter", [5, 5, 1, 128],
                initializer=tf.truncated_normal_initializer(stddev=.1, seed=RAND_SEED))
        bias3 = tf.get_variable("bias", [1],
                initializer=tf.constant_initializer(.1))
        conv3 = tf.nn.conv2d_transpose(conv2_out, filter3, tf.pack([batch_size, 28, 28, 1]), strides=[1,2,2,1], padding='SAME')
        conv3_normalized = batch_normalize(conv3)
        conv3_out = tf.nn.sigmoid(conv3_normalized + bias3)

    images, labels = next_batch()
    images = np.expand_dims(images, 3)
    real_images = tf.Variable(images, trainable=False, dtype=tf.float32)
    image_set = tf.concat(0, [conv3_out, real_images])
    targets_distinguisher = tf.constant([0] * 32 + [1] * 32, shape=[64,1],
            dtype=tf.float32)
    targets_generator = tf.constant([1] * 32, shape=[32,1])

    ########################################################################
    #                            Distinguisher                             #
    ########################################################################
    
    dist_input = image_set
    with tf.variable_scope("dist_conv1"):
        dist_conv1 = convLayer(dist_input, 5, 5, 1, 64, act_fun=tf.nn.elu,
                strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope("dist_conv2"):
        dist_conv2 = convLayer(dist_conv1, 5, 5, 64, 128, act_fun=tf.nn.elu,
                strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope("dist_conv3"):
        dist_conv3 = convLayer(dist_conv2, 5, 5, 128, 256, act_fun=tf.nn.elu,
                strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope("dist_conv4"):
        dist_conv4 = convLayer(dist_conv3, 5, 5, 256, 512, act_fun=tf.nn.elu,
                strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope("dist_ff1"):
        dist_ff1 = feedForwardLayer(tf.reshape(dist_conv4, [-1, 64 * 28 * 28]),
                64 * 28 * 28, 1, act_fun=None)

    # import ipdb; ipdb.set_trace()
    cross_entropy_distinguisher = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(dist_ff1, targets_distinguisher))
    optimizer_dist = tf.train.AdamOptimizer(0.0002)
    optimizer_dist.minimize(cross_entropy_distinguisher, var_list=[v for v in
        tf.trainable_variables() if 'dist' in v.name])

    cross_entropy_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(conv3_out, targets_generator))
    optimizer_gen = tf.train.AdamOptimizer(0.0002)
    optimizer_gen.minimize(cross_entropy_generator, var_list=[v for v in tf.trainable_variables() if 'gen' in v.name])

    pass

