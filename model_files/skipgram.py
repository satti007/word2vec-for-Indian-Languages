import time
import utils
import numpy as np
import tensorflow as tf

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
		yield np.array(x), np.array(y)[:,None]

train_data,train_params = utils.get_arguments()
word2idx,idx2word = train_data[0],train_data[1]
train_words	      = train_data[4].tolist()
lr,dim = train_params[1],train_params[2]
ws,neg = train_params[3],train_params[4]
epochs,save_dir = train_params[5],train_params[6]
pretrain,state  = train_params[7],train_params[8]

batch_size  = 1000
embed_dim   = dim
n_vocab	    = len(word2idx)
n_batches   = len(train_words)//batch_size
train_words = train_words[:n_batches*batch_size]

train_graph = tf.Graph()
with train_graph.as_default():
	inputs = tf.placeholder(tf.int32, [None], name='inputs')
	labels = tf.placeholder(tf.int32, [None,1], name='labels')
	embeds_matrix = tf.Variable(tf.random_uniform((n_vocab,embed_dim), -1, 1),name='embeds_matrix')
	embeds_lookup = tf.nn.embedding_lookup(embeds_matrix,inputs)
	softmax_w = tf.Variable(tf.truncated_normal((n_vocab,embed_dim)),name='softmax_w')
	softmax_b = tf.Variable(tf.zeros(n_vocab),name='softmax_b')
	loss = tf.nn.sampled_softmax_loss(weights=softmax_w,biases=softmax_b,
								  	  inputs=embeds_lookup,labels=labels,
								  	  num_sampled=neg,num_classes=n_vocab)
	cost = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session(graph=train_graph) as sess:
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
		batch_data = get_batches(train_words,batch_size,ws)
		for x,y in batch_data:
			batch_loss, _ = sess.run([cost,optimizer],feed_dict={inputs:x,labels:y})
			loss += batch_loss
			if (iteration+1)%100 == 0:
				print('Epoch {}/{}, iteration {} Avg.train_loss: {:.4f}, time_taken : {}'.
				format(e+1+state,epochs+state,iteration+1,loss/100,time.time()-start_time))
				loss = 0
				start_time = time.time()
			iteration = iteration + 1
		Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
		np.savez(save_dir+"/weights_"+str(e+1+state)+".npz", *Wts)
		print ('Weights saved at {} epoch'.format(e+1+state))