# -*- coding: utf-8 -*-
import tensorflow as tf

# 从文件中读入训练数据集
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义将要输入的x和期望输出的y，这里只是两个占位符，这类似与两个变量，需要在训练的时候输入真实的值。
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 我们定义两个函数用于变量初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 定义卷积OP
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 定义池化OP
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 定义第一层卷积的权值和偏置
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 对长度为784的一维的x做个变形，这里是变成28x28的矩阵，这个是原始图像的尺寸。
x_image = tf.reshape(x, [-1,28,28,1])

# 经过Relu激活函数，然后进行池化。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 经过这个池化后，图像尺寸变为原来的一半，即14×14
h_pool1 = max_pool_2x2(h_conv1)

# 将Relu函数的输出，再经过一层卷积神经网络，用法与第一层卷积神经网络类似。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 经过这个赤化后，图像尺寸再减小一半，即7×7
h_pool2 = max_pool_2x2(h_conv2)

# 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一维向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout,为了防止过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后加入一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

# 创建Session，开始训练
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
