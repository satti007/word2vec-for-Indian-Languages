'''
* @file tensorflowFuntions.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Fri Mar 22 20:29:54 IST 2019
* @Contains tensorflow functions
'''

import numpy as np
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)

def conv2d(input_layer,n_filters,ksize,stride,padding):
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=n_filters,
			kernel_size=ksize,stride=stride,padding=padding,activation_fn=tf.nn.relu)

def max_pool2d(input_layer,ksize,stride):
	return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

def cnnLayer(l,embd_dim,inputs,n_filters,maxlen):
	layer_l = conv2d(inputs,n_filters,[l,embd_dim],1,'VALID')
	max_l   = max_pool2d(layer_l,[maxlen+1-l,1],[1,1])
	max_l   = tf.squeeze(max_l)
	
	return max_l