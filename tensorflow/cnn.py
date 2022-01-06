import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

def cnn():

    """CNN pipeline"""

    # load sample images
    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    dataset = np.array([china, flower], dtype=np.float32)
    batch_size, height, width, channels = dataset.shape
    print(f"dataset shape is {dataset.shape}")

    # Create 2 filters
    filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1 # vertical line
    filters[3, :, :, 1] = 1 # horizontal line
    print(f"filters is: \n {filters}")

    # Create a graph with input X plus a convolutional layer applying the 2 filters
    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")

    # Create a graph with convolution layer plus a pooling layer 
    max_pool = tf.nn.max_pool(convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.Session() as sess:
        output = sess.run(max_pool, feed_dict={X: dataset})

    #plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
    plt.imshow(output[0])
    plt.show()


if __name__ == "__main__":
    cnn()