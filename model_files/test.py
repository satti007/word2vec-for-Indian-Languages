'''
* @file test.py
* @author Satish Golla
* @date 22-03-2019
* Contains testing code
'''

import os
import sys
sys.path.insert(0, '../utilities_files/')

import numpy as np
import utilsForTest
import commonFuctions
import tensorflow as tf
import tensorflowFuntions

np.random.seed(1234)
tf.set_random_seed(1234)

minlen,maxlen,models_dir,model_type,weights_dir,language,data_dir,sim_file_path = utilsForTest.parseArguments()
pair2score,word2Unitidx,vocabSize = utilsForTest.get_wordUnits(minlen,maxlen,language,model_type,data_dir,sim_file_path)

unitEmbd_dim     = 300
wordRep_dim      = int(weights_dir.split('_')[-1].strip('/'))
cnn_numFilters   = wordRep_dim//4

CNN_flag = 0
if model_type[:3] == 'CNN':
	CNN_flag = 1
	if model_type[-4:].strip('/') == 'syl':
		widths = [1,2,3,4]
	else:
		widths = [4,5,6,7]
else:
	unitEmbd_dim = wordRep_dim

train_graph = tf.Graph()
with train_graph.as_default():
	inputs  = tf.placeholder(tf.int32 ,[None,None],name='inputs')
	embeds_matrix = tf.Variable(tf.random_uniform((vocabSize,unitEmbd_dim), -1, 1),name='embeds_matrix')
	embeds_lookup = tf.nn.embedding_lookup(embeds_matrix,inputs,max_norm=1)
	
	if CNN_flag:
		input_p   = tf.expand_dims(embeds_lookup, -1)	
		max_1 = tensorflowFuntions.cnnLayer(widths[0],unitEmbd_dim,input_p,cnn_numFilters,maxlen)
		max_2 = tensorflowFuntions.cnnLayer(widths[1],unitEmbd_dim,input_p,cnn_numFilters,maxlen)
		max_3 = tensorflowFuntions.cnnLayer(widths[2],unitEmbd_dim,input_p,cnn_numFilters,maxlen)
		max_4 = tensorflowFuntions.cnnLayer(widths[3],unitEmbd_dim,input_p,cnn_numFilters,maxlen)
		word_rep = tf.concat([max_1,max_2,max_3,max_4],axis=1)
	else:
		word_rep = tf.reduce_mean(embeds_lookup,axis=1)

word2idx  = {}
wordUnits = np.zeros([len(word2Unitidx),maxlen],dtype = int)
for idx,word in enumerate(word2Unitidx):
	word2idx[word] = idx
	wordUnits[idx] = word2Unitidx[word]

wts_dir = models_dir+model_type+weights_dir
weights = os.listdir(wts_dir)
weights.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

print ('')
rho = []
rho_zero = []
for wts in weights:
	print ('Loading {} ...'.format(wts))
	model_weights = commonFuctions.loadWeights(wts_dir,wts)
	with tf.Session(graph=train_graph) as sess:
		sess.run(tf.global_variables_initializer())
		assign_ops = [w.assign(v) for w,v in zip(tf.trainable_variables(),model_weights)]
		sess.run(tf.global_variables_initializer())
		sess.run(assign_ops)
		wordEmbeds = sess.run(word_rep,feed_dict={inputs:wordUnits})
	
	word2Emb = {}
	for idx,word in enumerate(word2Unitidx):
		word2Emb[word] = wordEmbeds[word2idx[word]]
	
	embd_score  = []
	human_score = []
	for pair in pair2score:
		human_score.append(pair2score[pair])
		embd_score.append(commonFuctions.get_wordSim(word2Emb[pair[0]],word2Emb[pair[1]]))
	
	coef = commonFuctions.getRho(human_score,embd_score)
	print('rho: ',coef)
	rho.append(coef)
	
	human_score,embd_score = utilsForTest.remove_zeroScores(human_score,embd_score)
	coef = commonFuctions.getRho(human_score,embd_score)
	print('rho: ',coef)
	rho_zero.append(coef)
	print ('\n\n')

max_idx = np.argsort(rho)[-1]
print ('Max rho is at {} epoch and value is : {}'.format(max_idx+1,rho[max_idx]))
print ('rho_zero at max rho epoch is: {}'.format(rho_zero[max_idx]))
utilsForTest.writeTocsv(rho,rho_zero,models_dir+model_type,wordRep_dim)