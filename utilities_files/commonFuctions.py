'''
* @file commonFuctions.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Fri Mar 22 20:29:54 IST 2019
* @Contains common functions
'''

import pickle
import scipy.stats
import numpy as np
from collections import Counter

# function to load the data from *.pkl
def load_pickleFile(file_path):
	print ('Loading {} file...'.format(file_path))
	with open(file_path, 'rb') as f:
		return pickle.load(f)

# function to load the data from file_name.npy
def load_npyFile(file_name,isDict):
	print ('Loaded {}'.format(file_name))
	if isDict:
		return np.load(file_name).item()
	
	return np.load(file_name)

def loadWeights(wts_dir,wts):
	f = np.load(wts_dir+wts)
	weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	weights = weights[:1] + weights[3:]
	
	return weights

# function to save the data as file_name.npy
def saveTofile(save_dir,file_name,data):
	np.save(save_dir+'{}.npy'.format(file_name),data)
	print ('Saved {}.npy'.format(file_name))

def get_wordSim(vec1,vec2):
	return round(10*(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))),3)

def getRho(human_score,embd_score):
	print ('length of embd_scores: ',len(embd_score))
	print ('Embd_scores: ')
	print (embd_score)
	
	return round(100*(scipy.stats.spearmanr(human_score,embd_score)[0]),3)

# combine two dicts
def combineDict(word2freq,c_dict,c):
	print('Combining vocabulary from:  {} of vocab_size: {}K'.format(c,len(c_dict)//pow(10,3)))
	com_dict   = Counter(word2freq) + Counter(c_dict)
	com_dict   = dict(sorted(com_dict.items(),key=lambda x:x[1],reverse=True))
	print('After combining vocabulary, vocab_size: {}K'.format(len(com_dict)//pow(10,3)))
	
	return com_dict

# write stats to a file
def save_stats(stats,files_dir):
	file = open(files_dir+'stats.txt','w')
	file.write('Number of sentences:           {}M\n'.format(round(stats[0]/pow(10.0,6),2)))
	file.write('Number of unique sentences:    {}M\n'.format(round(stats[1]/pow(10.0,6),2)))
	file.write('Number of words    in vocabulary: {}M\n'.format(round(stats[2]/pow(10.0,6),2)))
	file.write('Number of sp.chars in vocabulary: {}\n'.format(len(stats[3])))
	file.write('Number of tokens    in corpora:   {}M\n'.format(round(stats[4]/pow(10.0,6),2)))
	file.write('Number of sp.tokens in corpora:   {}M\n'.format(round(stats[5]/pow(10.0,6),2)))
	file.write('Number of words with freq>=5 in vocabulary:  {}K\n'.format(round(stats[6]/pow(10.0,3),2)))
	file.write('Number of words with freq>=10 in vocabulary: {}K\n'.format(round(stats[7]/pow(10.0,3),2)))
	file.write('Number of words with freq>=20 in vocabulary: {}K\n'.format(round(stats[8]/pow(10.0,3),2)))
	file.write('\n\nsp.chars = sp.tokens:\n{}'.format(stats[3]))
	file.close()

# caluclate the stats
def get_stats(word_to_freq,tot_sen,uni_sen,files_dir,punctuation):
	word_to_freq = dict(sorted(word_to_freq.items(),key=lambda x:x[1],reverse=True))
	tot_vocab  = len(word_to_freq)
	pun_vocab  = [w for w in punctuation if w in word_to_freq]
	tot_tokens = sum(word_to_freq.values())
	pun_tokens = sum(word_to_freq[w] for w in punctuation if w in word_to_freq)
	vocab_5    = sum(1 for f in word_to_freq.values() if f >= 5)
	vocab_10   = sum(1 for f in word_to_freq.values() if f >= 10)
	vocab_20   = sum(1 for f in word_to_freq.values() if f >= 20)
	
	save_stats([tot_sen,uni_sen,tot_vocab,pun_vocab,tot_tokens,pun_tokens,vocab_5,vocab_10,vocab_20],files_dir)


'''
import pickle
import numpy as np

def save_obj(obj,name):
	with open(name,'wb') as f:
		pickle.dump(obj,f)

hyb2idx = np.load('hyb2idx.npy').item()
name = 'hyb2idx.pkl'
obj  = hyb2idx
save_obj(obj,name)
'''