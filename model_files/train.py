'''
* @file train.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Tue Apr 2 12:27:24 IST 2019
* @Contains train.py
'''

import os
import sys
sys.path.insert(0, '../utilities_files/')

import time
import argparse
import numpy as np
import utilsForTrain
import commonFuctions
import tensorflow as tf
import tensorflowFuntions

np.random.seed(1234)
tf.set_random_seed(1234)

train_data,train_params = utilsForTrain.parseArguments()
idx2unit     = train_data[0]
idx2word     = train_data[1]
word2Unitidx = train_data[2]
train_words  = train_data[3]
num_layers   = train_params[15]
train_words    = train_words[:100000]
unit,model,maxlen   = train_params[0],train_params[1],train_params[2]
lr,ws,neg,epochs    = train_params[3],train_params[4],train_params[5],train_params[6]
embd_dim,rep_dim    = train_params[7],train_params[8]
batch_size,save_dir = train_params[9],train_params[10]
pretrain,state      = train_params[11],train_params[12]
l2reg,beta          = train_params[13],train_params[14]

n_vocab     = len(idx2word)
n_embds     = len(idx2unit) + 1
n_filters   = rep_dim//num_layers
n_batches   = len(train_words)//batch_size
train_words = train_words[:n_batches*batch_size]
mask_init        = np.ones([n_embds,embd_dim])
mask_init[-1][:] = np.zeros([1,embd_dim])
epoch_log_file   = open('epoch_log_{}.txt'.format(rep_dim),'wb',0)

CNN_flag = 0
if model == 'CNN':
	CNN_flag = 1
	if unit == 'syl':
		widths = [1,2,3,4]
	else:
		widths = [4,5,6,7]
else:
	embd_dim = rep_dim

train_graph = tf.Graph()
initializer = tf.contrib.layers.xavier_initializer()
with train_graph.as_default():
	inputs  = tf.placeholder(tf.int32  ,[None,None],name='inputs')
	labels  = tf.placeholder(tf.int32  ,[None,1]   ,name='labels')
	mask    = tf.placeholder(tf.float32,[None,None],name='mask')
	embeds_matrix = tf.Variable(tf.concat([initializer((n_embds-1,embd_dim)),tf.zeros([1,embd_dim])],axis=0))
	embeds_lookup = tf.nn.embedding_lookup(embeds_matrix,inputs,max_norm=1)
	softmax_w = tf.Variable(initializer((n_vocab,rep_dim)))
	softmax_b = tf.Variable(tf.zeros(n_vocab))
	
	if CNN_flag:
		input_p   = tf.expand_dims(embeds_lookup, -1)	
		max_1 = tensorflowFuntions.cnnLayer(widths[0],embd_dim,input_p,n_filters,maxlen)
		max_2 = tensorflowFuntions.cnnLayer(widths[1],embd_dim,input_p,n_filters,maxlen)
		max_3 = tensorflowFuntions.cnnLayer(widths[2],embd_dim,input_p,n_filters,maxlen)
		max_4 = tensorflowFuntions.cnnLayer(widths[3],embd_dim,input_p,n_filters,maxlen)
		word_rep = tf.concat([max_1,max_2,max_3,max_4],axis=1)
	else:
		word_rep = tf.reduce_mean(embeds_lookup,axis=1)
	
	loss = tf.nn.sampled_softmax_loss(weights=softmax_w, biases=softmax_b,
									  labels=labels    , inputs=word_rep,
									  num_sampled=neg  , num_classes=n_vocab,num_true=1)
	
	t_loss  = tf.reduce_mean(loss)
	
	if l2reg:
		t_vars  = tf.trainable_variables()
		l2_loss = tf.multiply(tf.reduce_mean([tf.nn.l2_loss(v) for v in t_vars if 'bias' not in v.name]),beta)
		cost    = tf.add_n([t_loss,l2_loss])
	else:
		cost    = t_loss
	
	optimizer      = tf.train.AdamOptimizer(lr)
	grad_var_pairs = optimizer.compute_gradients(cost)
	train_op_all   = optimizer.apply_gradients(grad_var_pairs[1:])
	train_op_mask  = optimizer.apply_gradients([(grad_var_pairs[0][0]*mask,grad_var_pairs[0][1])])

with tf.Session(graph=train_graph) as sess:
	sess.run(tf.global_variables_initializer())
	if pretrain:
		f = np.load(save_dir+'weights_{}.npz'.format(state))
		init_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
		assign_ops   = [w.assign(v) for w, v in zip(tf.trainable_variables(), init_weights)]
		sess.run(tf.global_variables_initializer())
		sess.run(assign_ops)
		print ('weights loaded from {} epoch'.format(state))
	
	for e in range(0,epochs):
		avg_loss   = 0
		epoch_loss = 0
		iteration  = 0
		show_step  = 10
		start_time = time.time()
		epoch_time = time.time()
		batch_data = utilsForTrain.get_batches(train_words,word2Unitidx,batch_size,ws)
		for x,y in batch_data:
			batch_loss,_,_ = sess.run([cost,train_op_all,train_op_mask],feed_dict={inputs:x,labels:y,mask:mask_init})
			avg_loss = avg_loss + batch_loss
			if (iteration+1)%show_step == 0:
				epoch_loss = epoch_loss + avg_loss
				print('Epoch {}/{}, iter: {}, avg_batch_loss: {:.4f}, time_taken : {:.4f}'.
				format(e+1+state,epochs+state,iteration+1,avg_loss/show_step ,time.time()-start_time))
				avg_loss   = 0
				start_time = time.time()
			iteration = iteration + 1
		Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
		np.savez(save_dir+"weights_"+str(e+1+state)+".npz", *Wts)
		print ('Weights saved at {} epoch, avg_epoch_loss: {:.4f}'.format(e+1+state,epoch_loss/(iteration)))
		epoch_log_file.write(('Weights saved at {} epoch, avg_epoch_loss: {:.4f}\n'.format(e+1+state,epoch_loss/(iteration))).encode('utf-8'))
		print ('Epoch time taken : {:.4f}'.format(time.time()-epoch_time))
		epoch_time = time.time()