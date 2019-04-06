'''
* @file functionsForfastext.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Mon Mar 19 15:29:54 IST 2019
* @Contains functions needed to train, test fastext model
'''

## The below code has the following purpose:
## For removing words with len > 11 from sentences.txt
## sentences.txt contains cleaned data (after preprocessing)
## sentences.txt contains one sentence per line 
## This is done as fastext doesn't have an option for this task

def removeWords():
	import os
	
	data_dir   = '/DDP/data/corpora/andhrajyothy/stats/'
	data_words = open(data_dir + 'sentences.txt','r').read().split()
	print ('The number of data_words: {} M'.format(round(len(data_words)/pow(10,6),3)))
	
	wordMaxlen = 11
	train_file = open(data_dir + 'train_fastext.txt','wb',0)
	data_lines = open(data_dir + 'sentences.txt','r').readlines()
	for line in data_lines:
		sentence  = ''
		words	 = line.split()
		for idx,w in enumerate(words):
			w = w.strip()
			if len(w) <= wordMaxlen:
				sentence = sentence + w + ' '
			if idx == len(words)-1:
				sentence = sentence.strip()
				sentence = sentence + '\n'
		train_file.write(sentence.encode('utf-8'))
	
	train_file.close()
	train_file  = open(data_dir + 'train_fastext.txt','r')
	train_words = train_file.read().split()
	print ('The number of train_words: {} M'.format(round(len(train_words)/pow(10,6),3)))

#################################################################################################################################

## This code is for running experiments with fastext model
def runExpers():
	import os
	
	dim = [200,300,500]
	for d in dim:
		cmd = './fasttext skipgram -input /DDP/data/corpora/andhrajyothy/stats/train_fastext.txt '
		cmd = cmd +	'-output ../fastext_{} '.format(d)
		cmd = cmd + '-minCount 5 -minn 3 -maxn 6 -t 1 '
		cmd = cmd + '-dim {} -ws 5 -neg 10 -loss ns '.format(d)
		cmd = cmd +	'-lr 0.005  -epoch 20  -thread 8'
		print (cmd)
		print ('###############################')
		print ()
		os.system(cmd)
		print ()
		print ('###############################')

#################################################################################################################################

def doTest():
	import io
	import math
	import scipy.stats
	import numpy as np
	
	# Function to load word embeddings saved in a txt file
	# The format of embeddings file should be as follows,
	# "words" dimension				* FIRST LINE (has the dimension of the embeddings)
	# word1 x1 x2 ......... xd		* SECOND LINE(word1 follwoed by space separated vector)
	# word2 y1 y2 ......... yd
	def read_txt_embeddings(emb_path,full_vocab=False,max_vocab=200000,emb_dim=300):
		word2id = {}				  # dictionary to store the index of word 'W' as word2id['W'] = id
		vectors = []				  # list of vectors such that embedding of word 'W' is vectors[word2id['W']]
		
		with io.open(emb_path, 'r', newline='\n', errors='ignore') as f:
			for i, line in enumerate(f):
				if i == 0:									  # ignore the first line
					split = line.split()
					assert len(split) == 2
					assert emb_dim == int(split[1])
					print (split[0] + '\t' + split[1])
				else:
					word, vect = line.rstrip().split(' ', 1)	# split the line into word and vector
					vect = np.fromstring(vect, sep=' ')			# form a vector from space separated string
					if np.linalg.norm(vect) == 0:				# avoid to have null embeddings
						vect[0] = 0.01
					if word in word2id:
						if full_vocab:							# just throws a warning
							print("Word '%s' found twice in %s embedding file" % (word, emb_path))
					else:
						if not vect.shape == (emb_dim,):
							print("Invalid dimension (%i) for %s word '%s' in line %i" % (vect.shape[0], emb_path, word, i))
							continue
						assert vect.shape == (emb_dim,), i
						word2id[word] = len(word2id)
						vectors.append(vect[None])
					# print (word + '\t' + str(len(word2id)))
				if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
					break
		
		assert len(word2id) == len(vectors)
		print("Loaded %i pre-trained word embeddings." % len(vectors))
		
		id2word	= {v: k for k, v in word2id.items()}
		embeddings = np.concatenate(vectors, 0)
		
		return embeddings,word2id,id2word

	# return vector for a given word
	def get_wordEmbedding(word,word2id,embeddings,OOV_count,OOV_words,OOV):
		try:
			vec = embeddings[word2id[word]]					# get the embedding if it is in vocabulary
			OOV = False
		except:
			vec = np.zeros([300])							# return zero vector if OOV
			if word not in OOV_words:
				OOV_words.append(word)
				OOV_count = OOV_count + 1
			OOV = True
			print ('OOV word: {}'.format(word))
		
		return vec,OOV_count,OOV_words,OOV
	
	# caluclate cosine similarity
	def get_wordSim(vec1,vec2):
		return round(10*(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))),3)
	
	dim = [200,300,500]
	d2rho = {}
	d2rho_OOV = {}
	for d in dim:
		print ('#########################################\n')
		print ('Testing for {} dim......'.format(d))
		emb_path = '../model_files/models/fastext/SD/fastext_{}.vec'.format(d)	# path to embeddings (text)file
		embeddings,word2id,id2word = read_txt_embeddings(emb_path,True,400000,d)
		
		
		sim_file = '../../../data/WordSim/Telugu-WS.txt'		# path to WordSim dataset
		lines	 = open(sim_file).readlines()
		emb_score   = []
		human_score = []
		OOV_count = 0
		OOV_words = []
		OOV_index = []
		for idx,l in enumerate(lines):
			OOV1,OOV2 = False,False
			l	    = l.strip()
			w1,w2,s = l.split(',')[0].strip(),l.split(',')[1].strip(),float(l.split(',')[2].strip()) # split the line to get pair of words and their human_score
			v1,OOV_count,OOV_words,OOV1 = get_wordEmbedding(w1,word2id,embeddings,OOV_count,OOV_words,OOV1) # get vector for word 'W1'
			v2,OOV_count,OOV_words,OOV2 = get_wordEmbedding(w2,word2id,embeddings,OOV_count,OOV_words,OOV2) # get vector for word 'W2'
			if OOV1 or OOV2:
				e_sim = 0										   # if any word is OVV then assign sim = 0 
				OOV_index.append(idx)
			else:
				e_sim		= get_wordSim(v1,v2)				   # cosine similarity btw v1 & v2
			emb_score.append(e_sim)
			human_score.append(s)
			print (w1+' '+w2+' '+str(s)+' '+str(e_sim))
		
		OOV_words = list(set(OOV_words))
		OOV_count = len(OOV_words)
		print ('\nTotal OOV words: ',OOV_count)
		print ('They are: ')
		for w in OOV_words:
			print (w, end=',')
		
		print ('')
		rho = round(100*(scipy.stats.spearmanr(human_score,emb_score)[0]),3)
		d2rho[d] = rho
		print ('rho is with OVV words: ',rho)
		
		emb_score1   = [s for idx,s in enumerate(emb_score) if idx not in OOV_index]
		human_score1 = [s for idx,s in enumerate(human_score) if idx not in OOV_index]
		rho_OOV = round(100*(scipy.stats.spearmanr(human_score1,emb_score1)[0]),3)
		d2rho_OOV [d] = rho_OOV
		print ('rho is with out OVV words: ',rho_OOV)
		print ('#########################################\n')
		break
	
	print ('rho:')
	print (d2rho)
	print ('rho_OOV:')
	print (d2rho_OOV)

##################################################################################################################################
# removeWords()
# runExpers()
doTest()

'''
import numpy as np
import pandas as pd

df  = pd.DataFrame()
dim = [200,300,500]

for d in dim:
	loss_dim = []
	log_data = open('fastext_{}.log'.format(d)).readlines()
	for i in range(0,101):
		lines = ''
		for ld in log_data:
			if i < 10:
				if 'Progress:   {}.0%'.format(i) in ld:
					lines = lines + ld
			elif i < 100:
				if 'Progress:  {}.0%'.format(i) in ld:
					lines = lines + ld
			else:
				if 'Progress: {}.0%'.format(i) in ld:
					lines = lines + ld				
		values = list(map(float,re.findall(r'loss:  (.*?) ETA:',lines)))
		loss_dim.append(round(np.mean(values),3))
	print (loss_dim)
	col = str(d)
	df[col] = loss_dim

df.to_csv('loss.csv', sep='	', index=True, index_label='progress')
'''