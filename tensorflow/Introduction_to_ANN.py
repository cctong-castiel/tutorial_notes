import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from functools import partial

## using Perception for modelling
def perceptron():
    iris = load_iris()
    X = iris.data[:, (2,3)] # petal length, petal width
    y = (iris.target == 0).astype(np.int)

    per_clf = Perceptron(random_state=42)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred)

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def DNN():
    # import dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/')

    # Construction Phase
    n_inputs = 28 * 28 #MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    training = tf.placeholder_with_default(False, shape=(), name='training')
    '''
    with tf.name_scope('dnn'):
        hidden1 = neuron_layer(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name='outputs')
    '''
    with tf.name_scope('dnn'):
        # initialization
        init = tf.contrib.layers.variance_scaling_initializer() # He initialization
        #init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG') # Xavier initialization

        my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
        hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', kernel_initializer=init)
        bn1 = my_batch_norm_layer(hidden1)
        bn1_act = tf.nn.elu(bn1)
        hidden2 = tf.layers.dense(bn1_act, n_hidden2, name='hidden2')
        bn2 = my_batch_norm_layer(hidden2)
        bn2_act = tf.nn.elu(bn2)
        logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name='outputs')
        logits = my_batch_norm_layer(logits_before_bn)
    
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    
    learning_rate = 0.01
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Execution Phase
    n_epochs = 40
    batch_size = 50
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # update batch training data mean and standard deviation

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run([training_op, extra_update_ops], feed_dict={X:X_batch, y:y_batch})
            acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
            acc_val = accuracy.eval(feed_dict={X:mnist.validation.images,
                                               y:mnist.validation.labels})
            print(epoch, "Training accuracy:", acc_train, "Val accuracy:", acc_val)
        save_path = saver.save(sess, './my_model_final.ckpt')

        saver.restore(sess, './my_model_final.ckpt')
        X_new_scaled = mnist.validation.images
        Z = logits.eval(feed_dict={X:X_new_scaled})
        y_pred = np.argmax(Z, axis=1)
        print(y_pred)
    
    #with tf.Session() as sess:
        

if __name__ == '__main__':
    #perceptron()
    DNN()