import sys
import pickle
import numpy as np
import scipy.stats
import tensorflow as tf

np.random.seed(1234)
tf.set_random_seed(1234)

INDIC_NLP_LIB_HOME  = 'indic_nlp_library'					# path to the library source (needed for syllabification)
INDIC_NLP_RESOURCES = 'indic_nlp_resources'					# path to the resources needed by the above library

# loading the library for syllabification
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.syllable import  syllabifier

# function to get syllables of a word in an language
def getSyllables(word,lang):
	return syllabifier.orthographic_syllabify(word,lang)

def loadFile(name):
	print (name)
	with open(name, 'rb') as f:
		return pickle.load(f)

def checkOOV(w,w_syl,syl2idx,max_syl,pad_syl,OOV,OOV_count,OOV_sylls,OOV_words):
	wordSyl = []
	for syl in w_syl:
		if syl in syl2idx:
			wordSyl.append(syl2idx[syl])
		else:
			OOV = True
			OOV_count += 1
			OOV_sylls.append(w)
			wordSyl.append(pad_syl)
	if OOV:
		OOV_words.append(w)	
	
	wordSyl.extend([pad_syl]*(max_syl-len(w_syl))) 
	return wordSyl,OOV,OOV_count,OOV_sylls,OOV_words

def get_wordSyallables(sim_file,syl2idx,max_syl,pad_syl,lang):
	data      = open(sim_file).readlines()
	wordIdx   = 0
	OOV_count = 0
	OOV_sylls = []
	OOV_words = []
	OOV_index = []
	pair2score  = {}
	word2Sylidx = {}
	
	for idx,l in enumerate(data):
		OOV1,OOV2 = False,False
		l       = l.strip()
		w1,w2,s = l.split(',')[0].strip(),l.split(',')[1].strip(),float(l.split(',')[2].strip())
		w1_syl  = getSyllables(w1,lang)
		w2_syl  = getSyllables(w2,lang)
		if len(w1_syl) <= 4 and len(w2_syl) <= 4:
			w1_syl,OOV1,OOV_count,OOV_sylls,OOV_words = checkOOV(w1,w1_syl,syl2idx,max_syl,pad_syl,OOV1,OOV_count,OOV_sylls,OOV_words)
			w2_syl,OOV2,OOV_count,OOV_sylls,OOV_words = checkOOV(w2,w2_syl,syl2idx,max_syl,pad_syl,OOV2,OOV_count,OOV_sylls,OOV_words)
			word2Sylidx[w1] = w1_syl
			word2Sylidx[w2] = w2_syl
			pair2score[(w1,w2)] = s
		if OOV1 or OOV2:
			OOV_index.append(idx)
	
	return OOV_count,OOV_sylls,OOV_words,OOV_index,pair2score,word2Sylidx

def conv2d(input_layer,filters,ksize,stride,padding):
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
			kernel_size=ksize,stride=stride,padding=padding,activation_fn=tf.nn.relu)

def max_pool2d(input_layer,ksize,stride):
	return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

def cnnLayer(l,embd_dim,inputs,filters,max_syl):
	layer_l = conv2d(inputs,filters,[l,embd_dim],1,'VALID')
	max_l   = max_pool2d(layer_l,[max_syl+1-l,1],[1,1])
	max_l   = tf.squeeze(max_l)
	
	return max_l

def get_wordSim(vec1,vec2):
	return round(10*(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))),3)

# data_dir = '../../../data/corpora/wiki/stats/'
data_dir = '../../../data/corpora/andhrajyothy/stats/'
sim_file = '../../../data/WordSim/Telugu-WS.txt'

lang     = 'te'
max_syl  = 4
syl2idx  = loadFile(data_dir +'syl2idx.pkl')
pad_syl  = len(syl2idx)
OOV_count,OOV_sylls,OOV_words,OOV_index,pair2score,word2Sylidx = get_wordSyallables(sim_file,syl2idx,max_syl,pad_syl,lang)
if OOV_count:
	print ('The following {} number of syllables are OOV: {}'.format(OOV_count,OOV_sylls))
	print ('The above missing syllables are of the following words: {}'.format(OOV_words))


rep      = int(sys.argv[1])
end      = int(sys.argv[2])
filters  = rep//4
embd_dim = 300	# syl embd_dim
n_vocab  = len(syl2idx) + 1

train_graph = tf.Graph()
with train_graph.as_default():
	inputs  = tf.placeholder(tf.int32 ,[None,None],name='inputs')
	embeds_matrix = tf.Variable(tf.random_uniform((n_vocab,embd_dim), -1, 1),name='embeds_matrix')
	embeds_lookup = tf.nn.embedding_lookup(embeds_matrix,inputs,max_norm=1)
	input_p   = tf.expand_dims(embeds_lookup, -1)
	
	max_1 = cnnLayer(1,embd_dim,input_p,filters,max_syl)
	max_2 = cnnLayer(2,embd_dim,input_p,filters,max_syl)
	max_3 = cnnLayer(3,embd_dim,input_p,filters,max_syl)
	max_4 = cnnLayer(4,embd_dim,input_p,filters,max_syl)
	# max_5 = cnnLayer(5,embd_dim,input_p,filters,max_syl)
	# max_6 = cnnLayer(6,embd_dim,input_p,filters,max_syl)
	
	# word_rep = tf.concat([max_2,max_3,max_4,max_5,max_6],axis=1)
	word_rep = tf.concat([max_1,max_2,max_3,max_4],axis=1)

wordSyl = np.zeros([len(word2Sylidx),max_syl],dtype = int)
for i,w in enumerate(word2Sylidx):
	wordSyl[i] = word2Sylidx[w]

file = open('exper_1to4_{}/epoch_log.txt'.format(rep),'wb',0)
for state in range(1,end+1):
	with tf.Session(graph=train_graph) as sess:
		sess.run(tf.global_variables_initializer())
		f = np.load('exper_1to4_{}/weights/weights_{}.npz'.format(rep,state))
		initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
		initial_weights = initial_weights[:1] + initial_weights[3:]
		assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(),initial_weights)]
		sess.run(tf.global_variables_initializer())
		sess.run(assign_ops)
		wordEmbeds = sess.run(word_rep,feed_dict={inputs:wordSyl})
	
	word2Emb = {}
	for i,w in enumerate(word2Sylidx):
		word2Emb[w] = wordEmbeds[i]
	
	emb_score = []
	human_score = []
	for pair in pair2score:
		human_score.append(pair2score[pair])
		emb_score.append(get_wordSim(word2Emb[pair[0]],word2Emb[pair[1]]))
	
	print (emb_score)
	print (len(emb_score))
	print ('rho : ',round(100*(scipy.stats.spearmanr(human_score,emb_score)[0]),3))
	file.write(('{}, rho : {}\n'.format(state,round(100*(scipy.stats.spearmanr(human_score,emb_score)[0]),3))).encode('utf-8'))
	
	zero_ids = []
	for idx,s in enumerate(emb_score):
		# print (idx, s)
		if s == 0:
			# print ('T')
			zero_ids.append(idx)
	
	print (zero_ids)
	
	emb_score1   = []
	human_score1 = []
	for i in range(len(human_score)):
		if i not in zero_ids:
			emb_score1.append(emb_score[i])
			human_score1.append(human_score[i])
	
	print ('rho : ',round(100*(scipy.stats.spearmanr(human_score1,emb_score1)[0]),3))
	file.write(('{}, rho : {}\n'.format(state,round(100*(scipy.stats.spearmanr(human_score1,emb_score1)[0]),3))).encode('utf-8'))