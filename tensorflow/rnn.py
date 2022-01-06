import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

def non_tensorflow_pipe():
    # variables
    n_inputs = 3
    n_neurons = 5

    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])

    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
    b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

    # initialization
    init = tf.global_variables_initializer()

    # prepare mini-batch
    X0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
    X1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]])

    # run init
    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0:X0_batch, X1:X1_batch})
    print("Y0_val is: \n", Y0_val)
    print("Y1_val is: \n", Y1_val)

def static_unrolling():
    # variables
    n_inputs = 3
    n_neurons = 5
    n_steps = 2
    run_flag = 1

    if run_flag == 0:
        X0 = tf.placeholder(tf.float32, [None, n_inputs])
        X1 = tf.placeholder(tf.float32, [None, n_inputs])

        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
        output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

        Y0, Y1 = output_seqs

        print("Y0 isL \n", Y0)
        print("Y1 is: \n", Y1)
    elif run_flag == 1:
        X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2]))
        print("X_seqs is: \n", X_seqs)
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
        output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
        print("output_seqs is: \n", output_seqs)
        print("state is: \n", states)
        outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])
        print("outputs is: \n", outputs)

        # initialization
        init = tf.global_variables_initializer()

        X_batch = np.array([
            [[0,1,2],[9,8,7]],
            [[3,4,5],[0,0,0]],
            [[6,7,8],[6,5,4]],
            [[9,0,1],[3,2,1]]
        ])

        with tf.Session() as sess:
            init.run()
            outputs_val = outputs.eval(feed_dict={X: X_batch})
            print("outputs_val is: \n", outputs_val)

def dynamic_rolling():
    # variables
    n_inputs = 3
    n_neurons = 5
    n_steps = 2
    run_flag = 1

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    
    seq_length = tf.placeholder(tf.int32, [None])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    # initialization
    init = tf.global_variables_initializer()

    X_batch = np.array([
        [[0,1,2],[9,8,7]],
        [[3,4,5],[0,0,0]],
        [[6,7,8],[6,5,4]],
        [[9,0,1],[3,2,1]]
    ])
    seq_length_batch = np.array([2,1,2,2])

    with tf.Session() as sess:
        init.run()
        outputs_val, states_val = sess.run(
            [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch}
        )
        print("outputs_val is: \n", outputs_val)
        print("states_val is: \n", states_val)

def rnn_train():
    # variables
    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10
    n_epochs = 100
    batch_size = 150

    learning_rate = 0.001

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

def rnn_timeseries():
    # variables
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs
    )
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # cost function
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    n_iterations = 1500
    batch_size = 50
    X_batch = np.array([i for i in range(100)]).reshape((-1, n_steps, n_inputs))
    y_batch = np.array([i for i in range(1,101,1)]).reshape((-1, n_steps, n_inputs))
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        X_new = X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)

def rnn_stack_timeseries():
    # variables
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    # cost function
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    n_iterations = 1500
    batch_size = 50
    X_batch = np.array([i for i in range(100)]).reshape((-1, n_steps, n_inputs))
    y_batch = np.array([i for i in range(1,101,1)]).reshape((-1, n_steps, n_inputs))
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        X_new = X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)

def deep_rnn():
    # variables
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    n_layers = 3

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                                   activation=tf.nn.relu) 
                                                   for layer in range(n_layers)]
    '''
    devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
    layers = [tf.contrib.rnn.DeviceWrapper(
        dev, tf.contrib.rnn.BasicRNNCell(
            num_units=n_neurons
        )
    ) for dev in devices]'''
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    # cost function
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    n_iterations = 1500
    batch_size = 50
    X_batch = np.array([i for i in range(100)]).reshape((-1, n_steps, n_inputs))
    y_batch = np.array([i for i in range(1,101,1)]).reshape((-1, n_steps, n_inputs))
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        X_new = X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)

def deep_dropout_rnn():
    # variables
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    n_layers = 3

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    keep_prob = tf.placeholder_with_default(1.0, shape=())
    cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                                   activation=tf.nn.relu) 
                                                   for layer in range(n_layers)]
    cells_drop = [
        tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells
    ]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    # cost function
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    n_iterations = 1500
    batch_size = 50
    train_keep_prob = 0.5
    X_batch = np.array([i for i in range(100)]).reshape((-1, n_steps, n_inputs))
    y_batch = np.array([i for i in range(1,101,1)]).reshape((-1, n_steps, n_inputs))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            _, mse = sess.run([training_op, loss], feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
        saver.save(sess, "./models/rnn/rnn_dropout_ts_model")

    with tf.Session() as sess:
        saver.restore(sess, "./models/rnn/rnn_dropout_ts_model")
        X_new = X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)

def lstm_timeseries():
    # variables
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    n_layers = 3

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, 
                                           use_peepholes=True) 
                                                   for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    # cost function
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    n_iterations = 1500
    batch_size = 50
    X_batch = np.array([i for i in range(100)]).reshape((-1, n_steps, n_inputs))
    y_batch = np.array([i for i in range(1,101,1)]).reshape((-1, n_steps, n_inputs))
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        X_new = X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)

def gru_timeseries():
    # variables
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    n_layers = 3

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons) 
                                                   for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    # cost function
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    n_iterations = 1500
    batch_size = 50
    X_batch = np.array([i for i in range(100)]).reshape((-1, n_steps, n_inputs))
    y_batch = np.array([i for i in range(1,101,1)]).reshape((-1, n_steps, n_inputs))
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        X_new = X_batch
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        print(y_pred)

if __name__ == '__main__':
    #non_tensorflow_pipe()
    #static_unrolling()
    #dynamic_rolling()
    #rnn_train()
    #rnn_timeseries()
    #deep_rnn()
    #deep_dropout_rnn()
    #lstm_timeseries()
    gru_timeseries()