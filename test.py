# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
import numpy as np

def ImageToMatrix(filename):
    im = Image.open(filename) 
    #im.show()
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.array(data,dtype="float")/255.0
    return data

print "Recognition..., please wait!"
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./weight/model.ckpt.meta')
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name("Placeholder:0")
    y_conv = graph.get_tensor_by_name("Softmax:0")
    keep_prob = graph.get_tensor_by_name("Placeholder_2:0")
    saver.restore(sess, "./weight/model.ckpt")
    output =  sess.run(y_conv, {x_input: ImageToMatrix("4.png"), keep_prob: 1.0})
    print "The num is", tf.argmax(output, 1).eval()
