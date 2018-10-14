import time
import utils
import argparse
import numpy as np
import tensorflow as tf

np.random.seed(1234)
tf.set_random_seed(1234)

def get_target(words,idx,window_size):
	R = np.random.randint(1, window_size+1)
	start = idx - R if (idx - R) > 0 else 0
	stop  = idx + R
	target_words = set(words[start:idx] + words[idx+1:stop+1])
	
	return list(target_words)

def get_batches(words,word2Syllidx,batch_size,window_size):
	for idx in range(0,len(words),batch_size):
		x, y  = [], []
		batch = words[idx:idx+batch_size]
		for i in range(len(batch)):
			batch_x = batch[i]
			batch_y = get_target(batch,i,window_size)
			y.extend(batch_y)
			x.extend([word2Syllidx[batch_x]]*len(batch_y))
		yield np.array(x), np.array(y)[:,None]

def conv2d(input_layer,filters,ksize,stride,padding):
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
			kernel_size=ksize,stride=stride,padding=padding,activation_fn=tf.nn.tanh)

def max_pool2d(input_layer,ksize,stride):
	return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

max_syl = 10
def cnnLayer(l,embd_dim,inputs):
	layer_l = conv2d(inputs,1,[l,embd_dim],1,'VALID')
	max_l   = max_pool2d(layer_l,[max_syl+1-l,1],[1,1])
	max_l   = tf.squeeze(max_l)
	max_l   = tf.expand_dims(max_l,1)
	
	return max_l

train_data,train_params = utils.get_arguments()
word2idx,idx2word = train_data[0],train_data[1]
syl2idx,idx2syl   = train_data[2],train_data[3]
word2Syllidx      = train_data[4]
train_words       = train_data[5].tolist()
train_words = train_words[:1000]
lang   = train_params[0]
lr,dim = train_params[1],train_params[2]
ws,neg = train_params[3],train_params[4]
epochs,save_dir = train_params[5],train_params[6]
pretrain,state  = train_params[7],train_params[8]

rep_dim  = 5
embd_dim = dim
n_embds  = len(syl2idx) + 1
n_vocab  = len(word2idx)
batch_size  = 1000
n_batches   = len(train_words)//batch_size
train_words = train_words[:n_batches*batch_size]

train_graph = tf.Graph()
with train_graph.as_default():
	inputs  = tf.placeholder(tf.int32,[None,None],name='inputs')
	labels  = tf.placeholder(tf.int32,[None,1],name='labels')
	embeds_matrix = tf.Variable(tf.random_uniform((n_embds,embd_dim),-1,1))
	embeds_lookup = tf.nn.embedding_lookup(embeds_matrix,inputs)
	softmax_w = tf.Variable(tf.truncated_normal((n_vocab,rep_dim)))
	softmax_b = tf.Variable(tf.zeros(n_vocab))
	input_p   = tf.expand_dims(embeds_lookup, -1)
	
	max_2 = cnnLayer(2,embd_dim,input_p)
	max_3 = cnnLayer(3,embd_dim,input_p)
	max_4 = cnnLayer(4,embd_dim,input_p)
	max_5 = cnnLayer(5,embd_dim,input_p)
	max_6 = cnnLayer(6,embd_dim,input_p)
	
	word_rep = tf.stack([max_2,max_3,max_4,max_5,max_6],axis=1)
	word_rep = tf.squeeze(word_rep)
	
	loss = tf.nn.sampled_softmax_loss(weights=softmax_w,biases=softmax_b,
									  labels=labels,inputs=word_rep,
									  num_sampled=neg,num_classes=n_vocab,num_true=1)
	
	cost = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session(graph=train_graph) as sess:
	sess.run(tf.global_variables_initializer())
	if pretrain:
		f = np.load(save_dir + '/weights_{}.npz'.format(state))
		initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
		assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
		sess.run(tf.global_variables_initializer())
		sess.run(assign_ops)
		print ('weights loaded from {} epoch'.format(state))
	
	for e in range(0,epochs):
		loss = 0
		iteration  = 0
		start_time = time.time()
		batch_data = get_batches(train_words,word2Syllidx,batch_size,ws)
		for x,y in batch_data:
			batch_loss, _ = sess.run([cost,optimizer],feed_dict={inputs:x,labels:y})
			# loss += batch_loss
			# if (iteration+1)%100 == 0:
			print('Epoch {}/{}, iteration {} Avg.train_loss: {:.4f}, time_taken : {}'.
			format(e+1+state,epochs+state,iteration+1,batch_loss,time.time()-start_time))
			# loss = 0
			start_time = time.time()
			iteration = iteration + 1
		Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
		np.savez(save_dir+"/weights_"+str(e+1+state)+".npz", *Wts)
		print ('Weights saved at {} epoch'.format(e+1+state))