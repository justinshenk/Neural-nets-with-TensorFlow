import tensorflow as tf
import numpy as np
import exercise8

RAND_SEED = exercise8.RAND_SEED

def feedForwardLayer(inputs, n_in, n_out, act_fun=tf.nn.tanh):
    weights = tf.get_variable("weights", [n_in, n_out],
            initializer=tf.truncated_normal_initializer(stddev=.1,
                seed=RAND_SEED)) 
    bias = tf.get_variable("bias", [n_out], initializer=tf.constant_initializer(0.1)) 
    if act_fun:
        return act_fun(tf.matmul(inputs, weights) + bias)
    else:
        return tf.matmul(inputs, weights) + bias

def convLayer(inputs, kx, ky, channels, feature_maps, act_fun=tf.nn.tanh,
        strides=[1,1,1,1], padding='SAME'):
    kernels = tf.get_variable("kernels", [kx, ky, channels, feature_maps],
            initializer=tf.truncated_normal_initializer(stddev=.1,
                seed=RAND_SEED))
    bias = tf.get_variable("bias", [feature_maps],
            initializer=tf.constant_initializer(.1))
    convolution = tf.nn.conv2d(inputs, kernels, strides=strides, padding=padding)
    if act_fun:
        return act_fun(convolution + bias)
    else:
        return convolution + bias

def transposedConvLayer(inputs, kx, ky, channels, feature_maps, out_shape, act_fun=tf.nn.tanh,
        strides=[1,1,1,1], padding='SAME'):
    filter = tf.get_variable("filter", [kx, ky, feature_maps, channels],
            initializer=tf.truncated_normal_initializer(stddev=.1,
                seed=RAND_SEED))
    bias = tf.get_variable("bias", [feature_maps],
            initializer=tf.constant_initializer(.1))
    import ipdb; ipdb.set_trace()
    convolution = tf.nn.conv2d_transpose(inputs, filter, out_shape, strides=strides, padding=padding)
    if act_fun:
        return act_fun(convolution + bias)
    else:
        return convolution + bias

