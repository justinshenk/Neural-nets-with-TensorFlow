from utils import load_mnist_data, minibatches, one_hot
import tensorflow as tf
from progressbar import * # sudo pip3 install progressbar33

if __name__ == "__main__":
    d_train, l_train, d_test, l_test, d_val, l_val = load_mnist_data()
    batch_size = 1
    n_epochs = 10

    ############################################################################
    #                              Define network                              #
    ############################################################################
    kernel1_size = 5
    input_size = (28, 28)
    n1_feature_maps = 32
    # Input layer
    kernels1 = tf.Variable(tf.truncated_normal([kernel1_size,kernel1_size,1,n1_feature_maps], stddev=.1))
    bias1    = tf.Variable(tf.constant(.1, shape=[n1_feature_maps]))
    x_flat   = tf.placeholder(tf.float32, shape=[None, 784])
    x        = tf.reshape(x_flat, [-1, input_size[0], input_size[1], 1])
    d        = tf.placeholder(tf.int32, [None, 10])

    # Apply convolution kernels to input x
    convolution1 = tf.nn.conv2d(x, kernels1, strides=[1,1,1,1], padding="SAME")
    # Calculate neuron outputs by applying the activation1 function
    conv1 = activation1 = tf.nn.tanh(convolution1 + bias1)
    # Apply max pooling to outputs
    pool1 = tf.nn.max_pool(activation1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # second convolutional layer
    kernel2_size    = 5
    n2_feature_maps = 64
    bias2           = tf.Variable(tf.constant(.1, shape=[n2_feature_maps]))
    kernels2        = tf.Variable(tf.truncated_normal([kernel2_size, kernel2_size,
        n1_feature_maps, n2_feature_maps], stddev=.1))
    convolution2    = tf.nn.conv2d(pool1, kernels2, strides=[1,1,1,1], padding="SAME")
    conv2           = activation2 = tf.nn.tanh(convolution2 + bias2)
    pool2           = tf.nn.max_pool(activation2, ksize=[1,2,2,1],
            strides=[1,2,2,1], padding="SAME") # question: how do I select the kernel sizes for pooling or convolution?

    # first FF layer
    n3_ffn = 1024
    bias3  = tf.Variable(tf.constant(.1, shape=[n3_ffn]))
    pool2_shape = pool2.get_shape()
    ffn3_weights = tf.Variable(tf.truncated_normal([pool2_shape[1].value * pool2_shape[2].value
        * pool2_shape[3].value, n3_ffn]))
    activation3 = tf.nn.tanh(tf.matmul(tf.reshape(pool2, [batch_size,-1]), ffn3_weights) + bias3)
     
    # seconf FF layer (readout)
    n4_ffn = 10
    bias4  = tf.Variable(tf.constant(.1, shape=[n4_ffn]))
    ffn3_shape = activation3.get_shape()
    ffn4_weights = tf.Variable(tf.truncated_normal([ffn3_shape[1].value, n4_ffn]))
    y = activation4 = tf.matmul(activation3, ffn4_weights) + bias4

    ############################################################################
    #                         Define training process                          #
    ############################################################################
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, d)
    optimizer     = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    training_step = optimizer.minimize(cross_entropy)

    ############################################################################
    #                           Accuracy computation                           #
    ############################################################################
    
    # check if neuron firing strongest coincides with max value position in real
    # labels
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ############################################################################
    #                              Train the net                               #
    ############################################################################
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        widgets = ['Training: ', Percentage(), ' ', AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' ', ETA()]
        pbar    = ProgressBar(widgets=widgets, maxval=n_epochs).start()
        for i in range(n_epochs):
            for mb, labels in minibatches(d_train, l_train, batch_size=batch_size):
                sess.run(training_step, feed_dict={x_flat: mb, d: labels})
            current_accuracy = sess.run(accuracy, feed_dict={x_flat: d_train, d: one_hot(l_train, 10)})
            print(current_accuracy)
            pbar.update(i)

