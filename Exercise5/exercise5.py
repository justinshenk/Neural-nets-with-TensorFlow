import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import load_svhn_data, minibatches, one_hot
from progressbar import * # sudo pip3 install progressbar33
from network_builder import *

RAND_SEED = 0

np.random.seed(RAND_SEED)

if __name__ == "__main__":

    batch_size = 100
    n_epochs   = 1000
    learning_rate = 1e-2
    act_fun = tf.nn.elu

    tf.set_random_seed(RAND_SEED)
    tf.reset_default_graph()

    ############################################################################
    #                              Define network                              #
    ############################################################################
    # Input layer
    x        = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    # targets
    d        = tf.placeholder(tf.int32, [None, 10])

    # first conv layer
    k1_sz    = 3
    n_feat1 = 16
    with tf.variable_scope("conv1"):
        act1 = convLayer(x, k1_sz, k1_sz, 1, n_feat1, act_fun=act_fun)
    pool1 = tf.nn.max_pool(act1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # second convolutional layer
    k2_sz    = 3
    n_feat2 = 32
    with tf.variable_scope("conv2"):
        act2 = convLayer(pool1, k2_sz, k2_sz, n_feat1, n_feat2, act_fun=act_fun)
    # pool2 = tf.nn.max_pool(act2, ksize=[1,2,2,1], strides=[1,2,2,1],
    #         padding="SAME")

    # third convolutional layer
    k3_sz    = 3
    n_feat3 = 64
    with tf.variable_scope("conv3"):
        act3 = convLayer(act2, k3_sz, k3_sz, n_feat2, n_feat3, act_fun=act_fun)
    pool3 = tf.nn.max_pool(act3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") 

    # fourth convolutional layer
    n_feat4 = 16
    with tf.variable_scope("conv4"):
        act4 = convLayer(pool3, 1, 1, n_feat3, n_feat4, act_fun=act_fun)

    # fifth convolutional layer
    k5_sz    = 3
    n_feat5 = 64
    with tf.variable_scope("conv5"):
        act5 = convLayer(act4, k5_sz, k5_sz, n_feat4, n_feat5, act_fun=act_fun)
    pool4 = tf.nn.max_pool(act4, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") 

    # first FF layer
    n1_hidden = 1024
    n_in      = 4*4*n_feat5//4
    with tf.variable_scope("ff1"):
        act4 = feedForwardLayer(tf.reshape(pool4, [-1, n_in]), n_in, n1_hidden, act_fun=act_fun)

    # second FF layer (readout)
    n2_hidden    = 10
    with tf.variable_scope("ff2"):
        y = feedForwardLayer(act4, n1_hidden, n2_hidden, act_fun=act_fun)

    ############################################################################
    #                         Define training process                          #
    ############################################################################

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, d))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(cross_entropy)
    training_step = optimizer.apply_gradients(gradients)

    ############################################################################
    #                           Accuracy computation                           #
    ############################################################################

    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(d,1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    training_step_accuracy  = []
    validation_acc          = []

    ############################################################################
    #                              Train the net                               #
    ############################################################################


    widgets = ['Training: ', Percentage(), ' ', AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' ', ETA()]
    pbar    = ProgressBar(widgets=widgets, maxval=n_epochs).start()
    d_train, l_train, d_test, l_test, d_val, l_val = load_svhn_data(normalize=True)
    batches = minibatches(d_train, l_train, batch_size=batch_size)
    training_step_accuracy  = []
    val_accuracy  = []
    save_file = "./exercise5.ckpt"
    plt.ion()
    plt.gca().set_ylim([0,1])
    plt.gca().set_xlim([0,n_epochs/30])


    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists(save_file):
            saver.restore(sess, save_file)
        else:
            sess.run(tf.initialize_all_variables())
        for i in range(n_epochs):
            pbar.update(i)
            data, labels = batches.__next__()
            sess.run(training_step, feed_dict={x: data, d: labels})
            if i > 1 and i % 30 == 0:
                curr_train_acc = sess.run(accuracy, feed_dict={ x: data, d: labels })
                curr_val_acc = sess.run(accuracy, feed_dict={ x: d_val, d:
                    one_hot(l_val, 10) })
                training_step_accuracy.append(curr_train_acc)
                training_step_accuracy.append(curr_val_acc)
                plt.plot(training_step_accuracy, color = "b")
                plt.plot(val_accuracy, color = "b")
                plt.draw()
                plt.pause(0.0001)
        test_acc = sess.run(accuracy, feed_dict={x: d_test, d: one_hot(l_test, 10)})
        print("Final accuracy on test set: %f" % test_acc)
        saver.save(sess, save_file)
        plt.plot(training_step_accuracy, color = "b")
        plt.plot(val_accuracy, color = "b")
        plt.ioff()
        plt.show()
