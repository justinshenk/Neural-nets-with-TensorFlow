# from utils import load_mnist_data, minibatches, one_hot
import tensorflow as tf
from progressbar import *  # sudo pip3 install progressbar33
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if __name__ == "__main__":
    batch_size = 1000
    n_epochs = 10000

    ############################################################################
    #                              Define network                              #
    ############################################################################
    # Input layer
    kernels1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=.1))
    bias1 = tf.Variable(tf.constant(.1, shape=[32]))
    x_flat = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.reshape(x_flat, [-1, 28, 28, 1])
    d = tf.placeholder(tf.int32, [None, 10])

    # Apply convolution kernels to input x
    convolution1 = tf.nn.conv2d(
        x, kernels1, strides=[1, 1, 1, 1], padding="SAME")

    # Calculate neuron outputs by applying the activation1 function
    conv1 = activation1 = tf.nn.tanh(convolution1 + bias1)

    # Apply max pooling to outputs
    pool1 = tf.nn.max_pool(
        activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # second convolutional layer
    kernel2_size = 5
    n2_feature_maps = 64
    bias2 = tf.Variable(tf.constant(.1, shape=[n2_feature_maps]))
    kernels2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=.1))
    convolution2 = tf.nn.conv2d(
        pool1, kernels2, strides=[1, 1, 1, 1], padding="SAME")
    conv2 = activation2 = tf.nn.tanh(convolution2 + bias2)
    pool2 = tf.nn.max_pool(
        activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )  # question: how do I select the kernel sizes for pooling or convolution?

    # first FF layer
    bias3 = tf.Variable(tf.constant(.1, shape=[1024]))
    ffn3_weights = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024]))
    activation3 = tf.nn.tanh(
        tf.matmul(tf.reshape(pool2, [-1, 7 * 7 * 64]), ffn3_weights) + bias3)

    # second FF layer (readout)
    bias4 = tf.Variable(tf.constant(.1, shape=[10]))
    ffn4_weights = tf.Variable(tf.truncated_normal([1024, 10]))
    y = activation4 = tf.matmul(activation3, ffn4_weights) + bias4

    ############################################################################
    #                         Define training process                          #
    ############################################################################

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, d))
    # I also tried the gradient descent optimizer, but that did not improve
    # performance withe learning rates 0.2, 0.5 or 0.8. Maybe one would hate to
    # wait ridiculously long.
    training_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    ############################################################################
    #                           Accuracy computation                           #
    ############################################################################

    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    training_step_accuracy = []
    validation_acc = []

    ############################################################################
    #                              Train the net                               #
    ############################################################################

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        widgets = [
            'Training: ',
            Percentage(), ' ',
            AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' ',
            ETA()
        ]
        pbar = ProgressBar(widgets=widgets, maxval=n_epochs).start()

        plt.ion()
        plt.gca().set_ylim([0, 1])
        plt.gca().set_xlim([0, n_epochs / 50])

        for i in range(n_epochs):
            pbar.update(i)
            batch = mnist.train.next_batch(batch_size)
            sess.run(training_step, feed_dict={x_flat: batch[0], d: batch[1]})
            if i > 1 and i % 50 == 0:
                curr_acc = sess.run(
                    accuracy, feed_dict={
                        x_flat: batch[0],
                        d: batch[1]
                    })
                curr_val_acc = sess.run(
                    accuracy,
                    feed_dict={
                        x_flat: mnist.validation.images,
                        d: mnist.validation.labels
                    })
                validation_acc.append(curr_val_acc)
                training_step_accuracy.append(curr_acc)
                plt.plot(training_step_accuracy, color="b")
                plt.plot(validation_acc, color="r")
                plt.draw()
                plt.pause(0.0001)

                # print("step %d, accuracy on current batch: %g" % (i, curr_acc))
        # I split this up to not run out of memory
        test_acc1 = sess.run(
            accuracy,
            feed_dict={
                x_flat: mnist.test.images[:5000],
                d: mnist.test.labels[:5000]
            })
        test_acc2 = sess.run(
            accuracy,
            feed_dict={
                x_flat: mnist.test.images[5000:],
                d: mnist.test.labels[5000:]
            })
        test_acc = (test_acc1 + test_acc2) / 2
        print("Final accuracy on test set: %g" % test_acc)
        plt.plot(training_step_accuracy, color="b")
        plt.plot(validation_acc, color="r")
        plt.ioff()
        plt.show()
