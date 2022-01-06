import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from functools import partial
import re

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
    # var
    var = {
        'load_flag': 'fresh'
    }

    # import dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/')

    # Construction Phase
    n_inputs = 28 * 28 #MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X_1')
    y = tf.placeholder(tf.int64, shape=(None), name='y_1')

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
    
    if var['load_flag'] == 'fresh':
        with tf.name_scope('train'):
            # Gradient Clipping
            '''
            threshold = 1.0
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
            training_op = optimizer.apply_gradients(capped_gvs)
            '''
            # Freezing the lower layers
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden2|logits_before_bn")
            training_op = optimizer.minimize(loss, var_list=train_vars)

        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        saver = tf.train.Saver()
    else:
        # reusing tensorflow trained model
        # needs to comment out X, y and tf.name_scope('dnn')
        saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        y = tf.get_default_graph().get_tensor_by_name("y:0")
        accuracy = tf.get_default_graph().get_tensor_by_name("eval/Mean:0")
        training_op = tf.get_default_graph().get_operation_by_name("train/GradientDescent")
        
        for op in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
            if re.search("dnn", str(op)):
                print(op)
        '''
        for op in (X, y, accuracy, training_op):
            tf.add_to_collection("my_important_ops", op)'''

        X, y, accuracy, training_op = tf.get_collection("my_important_ops")
    
    '''
    # Get a handle on the assignment nodes for the hidden1 variables
    original_w = tf.Variable(tf.random_normal([n_inputs, n_hidden1]))
    original_b = tf.Variable(tf.random_normal([1, n_hidden1]))
    graph = tf.get_default_graph()
    assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
    assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
    init_kernel = assign_kernel.inputs[1]
    init_bias = assign_bias.inputs[1]
    init = tf.global_variables_initializer()'''

    # Execution Phase
    n_epochs = 40
    batch_size = 50
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # update batch training data mean and standard deviation
    
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
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
    '''
    with tf.Session() as sess:
        init()
        saver.restore(sess, "./my_model_final.ckpt")
        h1_cache = sess.run(hidden1, feed_dict={X:mnist.train.images})
        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(mnist.train.num_examples)
            hidden1_batches = np.array_split(h1_cache[shuffled_idx], mnist.train.num_examples // batch_size)
            y_batches = np.array_split(mnist.train.labels[shuffled_idx], mnist.train.num_examples // batch_size)
            for hidden1_batch, y_batch in zip(hidden1_batches, y_batches):
                sess.run(training_op, feed_dict={hidden1:hidden1_batch, y:y_batch})
            acc_train = accuracy.eval(feed_dict={X:hidden1_batch,
                                                 y:y_batch})
            acc_val = accuracy.eval(feed_dict={X:mnist.validation.images,
                                               y:mnist.validation.labels})
            print(epoch, "Training accuracy:", acc_train, "Val accuracy:", acc_val)
        save_path = saver.save(sess, './my_model_final.ckpt')

        saver.restore(sess, './my_model_final.ckpt')
        X_new_scaled = mnist.validation.images
        Z = logits.eval(feed_dict={X:X_new_scaled})
        y_pred = np.argmax(Z, axis=1)
        print(y_pred)'''
    
    #with tf.Session() as sess:
        

if __name__ == '__main__':
    #perceptron()
    DNN()