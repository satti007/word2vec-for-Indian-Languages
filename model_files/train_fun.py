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

def get_batches(words,batch_size,window_size):
	for idx in range(0,len(words),batch_size):
		x, y  = [], []
		batch = words[idx:idx+batch_size]
		for i in range(len(batch)):
			batch_x = batch[i]
			batch_y = get_target(batch,i,window_size)
			y.extend(batch_y)
			x.extend([batch_x]*len(batch_y))
		yield x,y

def conv2d(input_layer,filters,ksize,stride,padding):
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
			kernel_size=ksize,stride=stride,padding=padding,activation_fn=tf.nn.tanh)

def body(i,x1,n):
	x1 = tf.concat([x1,zero],axis=0)
	i  = tf.add(i,1)
	
	return i,x1,n

def condition(i,x1,n):
	return tf.less(i,n)

def f2(input): 
	return input

def f1(input,x,y):
	x1 = input_p
	n  = tf.subtract(y,x)
	i  = tf.constant(0)
	i,x1,n = tf.while_loop(condition,body, (i,x1,n))
	x1 = tf.expand_dims(x1, 0)
	
	return x1

def cnnLayer(l,embd_dim,inputs,input_x,input_p):
	x = tf.shape(inputs)[1]
	y = tf.constant(l)
	input_l = tf.cond(tf.less(x, y),lambda:f1(input_p,x,y),lambda:f2(input_x))
	layer_l = conv2d(input_l,1,[l,embd_dim],1,'VALID')
	max_l   = tf.reduce_max(layer_l)
	
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
	inputs  = tf.placeholder(tf.int32,[1,None],name='inputs')
	labels  = tf.placeholder(tf.int32,[1,None],name='labels')
	embeds_matrix = tf.Variable(tf.random_uniform((n_embds,embd_dim),-1,1))
	embeds_lookup = tf.nn.embedding_lookup(embeds_matrix,inputs)
	softmax_w = tf.Variable(tf.truncated_normal((n_vocab,rep_dim)))
	softmax_b = tf.Variable(tf.zeros(n_vocab))
	input_x  = tf.reshape(embeds_lookup, [1,-1,embd_dim,1])
	input_p  = tf.reshape(embeds_lookup, [-1,embd_dim,1])
	zero     = tf.zeros([1,embd_dim,1],dtype=tf.float32,name=None)
	
	max_2 = cnnLayer(2,embd_dim,inputs,input_x,input_p)
	max_3 = cnnLayer(3,embd_dim,inputs,input_x,input_p)
	max_4 = cnnLayer(4,embd_dim,inputs,input_x,input_p)
	max_5 = cnnLayer(5,embd_dim,inputs,input_x,input_p)
	max_6 = cnnLayer(6,embd_dim,inputs,input_x,input_p)
	
	word_rep = tf.stack([max_2,max_3,max_4,max_5,max_6],axis=0)
	word_rep = tf.reshape(word_rep,[1,rep_dim])
	
	loss = tf.nn.sampled_softmax_loss(weights=softmax_w,biases=softmax_b,
									  labels =labels,inputs=word_rep,
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
		start_time = time.time()
		for i,idx in enumerate(train_words):
			word_loss    = 0
			syl_idxs     = np.array([word2Syllidx[idx]])
			target_words = get_target(train_words,i,ws)
			for tar in target_words:
				tar_loss,_ = sess.run([cost,optimizer],feed_dict={inputs:syl_idxs,labels:np.array([[tar]])})
				word_loss  += tar_loss
			loss += word_loss/len(target_words)
			if (i+1)%1000 == 0:
				print('Epoch {}/{}, iteration {} Avg.train_loss: {:.4f}, time_taken : {}'.
				format(e+1+state,epochs+state,i+1,loss/1000,time.time()-start_time))
				loss = 0
				start_time = time.time()
		Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
		np.savez(save_dir+"/weights_"+str(e+1+state)+".npz", *Wts)
		print ('Weights saved at {} epoch'.format(e+1+state))
