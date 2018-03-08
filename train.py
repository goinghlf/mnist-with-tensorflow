# -*- coding: utf-8 -*-
import tensorflow as tf

# Read the training data set from the file
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define the x to be entered and y_ to be expected, just two placeholders, similar to two variables that need to be entered at training time
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# Define two functions for variable initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Define convolution OP
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Define pooling OP
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Define the weight and bias of the first layer convolution
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Make a length of 784 one-dimensional x to a 28x28 matrix, this is the size of the original image.
x_image = tf.reshape(x, [-1,28,28,1])

# After Relu activation function, then pooling.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# After this pooling, the image size becomes half of the original, that is 14 × 14
h_pool1 = max_pool_2x2(h_conv1)

# The output of Relu function, and then through a layer of convolutional neural network, which is similar to the first layer of convolutional neural network.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# After this pooling, the image size is further reduced by half, that is, 7 × 7
h_pool2 = max_pool_2x2(h_conv2)

# Now that the image size is reduced to 7x7, we add a fully connected layer of 1024 neurons to process the entire image. We reshape the tensor output from the pooling layer into a one-dimensional vector, multiply the weight matrix, and add bias, then use ReLU on it.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout,to prevent overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add a softmax layer at last
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

# Create Session，start training
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(20000):
	  batch = mnist.train.next_batch(50)
	  if i%100 == 0:
	    train_accuracy = accuracy.eval(feed_dict={
		x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	save_path = saver.save(sess, "./weight/model.ckpt")
	print "test accuracy %g"%accuracy.eval(feed_dict={
	    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
