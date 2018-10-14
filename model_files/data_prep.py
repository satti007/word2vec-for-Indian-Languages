# -*- coding: utf-8 -*-
import time
import argparse
import numpy as np
import pandas as pd

np.random.seed(1234)

INDIC_NLP_LIB_HOME  = 'indic_nlp_library'		# path to the library source
INDIC_NLP_RESOURCES = 'indic_nlp_resources'		# path to the resources needed by the library

# loading the library for syllabification
import sys
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp import loader
loader.load()

from indicnlp.syllable import  syllabifier

# function to get syallables of a word in an language
def getSyallables(word,lang):
	return syllabifier.orthographic_syllabify(word,lang)

# function to save the data as file_name.npy
def saveTofile(file_name,data):
	np.save('stats/{}.npy'.format(file_name),data)
	print ('Saved {}.npy'.format(file_name))

# function to pad the syallables to make all words have same no.of.syallables
def padSyallables(word2Sylidx,syl2idx,max_syls):
	pad_syl = len(syl2idx)
	for word in word2Sylidx:
		word2Sylidx[word] = [syl2idx[syl] for syl in word2Sylidx[word]]
		word2Sylidx[word].extend([pad_syl]*(max_syls-len(word2Sylidx[word])))
	
	return word2Sylidx

# function to create syl2idx, idx2syl dicts
def createSyllableTables(word2idx,lang):
	syl_idx  = 0
	max_syls = 0
	word2Sylidx = {}
	syl2idx,idx2syl = {},{}
	for word in word2idx:
		syllables = getSyallables(word,lang)
		word2Sylidx[word2idx[word]] = syllables
		if max_syls < len(syllables):
			max_syls = len(syllables)
		for syl in syllables:
			if syl not in syl2idx:
				syl2idx[syl] = syl_idx
				idx2syl[syl_idx] = syl
				syl_idx = syl_idx + 1
	
	return syl2idx,idx2syl,word2Sylidx,max_syls

# function for sampling
def subSampling(tokens,vocab,word2freq,idx2word,threshold):
	total_count   = sum(word2freq.values())
	word2probDrop = {word : 1 - np.sqrt(threshold/(float(word2freq[word])/total_count)) for word in vocab}
	sampled_words = [w for w in tokens if np.random.random() < (1 - word2probDrop[idx2word[w]])]
	
	return sampled_words

# function to create word2idx, idx2word dicts
def createLookuptables(train_vocab,word2freq):
	sorted_vocab = sorted(train_vocab, key=lambda word: word2freq[word], reverse=True)	# sort the words by their frequency
	idx2word = {idx  : word for idx, word in enumerate(sorted_vocab)}					# idx2word dict {0:w1, 1:w2,....}	
	word2idx = {word : idx for idx, word in idx2word.items()}							# word2idx dict {w1:0, w2:1,....}
	
	return word2idx, idx2word

# preprocessing function
def preprocess(freq_file,data_file,lang,minCount=5,wordMaxlen=1,doSampling=True,threshold=1e-5):
	print ('Starting preprocessing...')
	word2freq_df = pd.read_csv(freq_file,header=None)						# load the word2freq dataframe
	word2freq    = dict(zip(list(word2freq_df[0]),list(word2freq_df[1])))	# convert the dataframe to dict
	all_vocab    = list(word2freq_df[0])									# vocabulary as list
	print ('The total number of words in vocabulary are: {}K'.format(len(all_vocab)//pow(10,3)))
	train_vocab = [w for w in all_vocab if word2freq[w]>=minCount and len(w)<=wordMaxlen]	# valid vocab after freq,length cut-offs
	print ('The number of words in vocabulary with freq >={} and length <={} are : {}K'.format(minCount,wordMaxlen,len(train_vocab)//pow(10,3)))
	word2idx,idx2word = createLookuptables(train_vocab,word2freq)			# create word2idx, idx2word dicts
	saveTofile('word2idx',word2idx)											
	saveTofile('idx2word',idx2word)											
	text  = open(data_file,'r').read()										# read the corpora file
	all_words = text.split()												# get all the words
	all_words = [w.strip() for w in all_words] 								# strip '\n', ' ' from the words
	print ('The total number of tokens in corpora are:{}M'.format(len(all_words)//pow(10,6)))
	train_words = [word2idx[w] for w in all_words if w in word2idx]			# take words only which are in valid vocab
	print ('The total number of tokens in corpora with freq >={} and length <={} are : {}M'.format(minCount,wordMaxlen,len(train_words)//pow(10,6)))
	saveTofile('train_words',np.asarray(train_words))
	if doSampling:															
		train_words = subSampling(train_words,train_vocab,word2freq,idx2word,threshold)
		print ('The number of tokens in corpora after subSampling are: {}M'.format(len(train_words)//pow(10,6)))
		saveTofile('sampledTrain_words',np.asarray(train_words))
	syl2idx,idx2syl,word2Sylidx,max_syls = createSyllableTables(word2idx,lang)
	print ('The total number of syllables are: ',len(syl2idx))
	word2Sylidx = padSyallables(word2Sylidx,syl2idx,max_syls)
	saveTofile('syl2idx',syl2idx)
	saveTofile('idx2syl',idx2syl)
	saveTofile('word2Sylidx',word2Sylidx)
	
	return [train_words,word2idx,idx2word,syl2idx,idx2syl]

# function to get the arguments
def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--freq_file', type=str)					# word2feq dict file(.csv)
	ap.add_argument('--data_file', type=str)					# corpora(text) file(.txt)
	ap.add_argument('--lang',      type=str)					# corpora language needed for the syllabification
	ap.add_argument('--minCount'  , type=int,  default = 5)		# (min)word freq cut-off
	ap.add_argument('--wordMaxlen', type=int,  default = 1)		# (max)word length cut-off
	ap.add_argument('--doSampling', type=bool, default = True)	# do subSampling if true 
	ap.add_argument('--threshold' , type=float,default = 1e-5)	# subSampling threshold
	
	print ('Parsing the Arguments...')
	args = vars(ap.parse_args())
	
	lang = args['lang']
	freq_file  = args['freq_file']
	data_file  = args['data_file']
	minCount   = args['minCount']
	wordMaxlen = args['wordMaxlen']
	doSampling = args['doSampling']
	threshold  = args['threshold']
	
	print ('Arguments Parsing Done!')
	print ('Arguments details: ')
	print ('lang: ',lang)
	print ('freq_file: ',freq_file)
	print ('data_file: ',data_file)
	print ('minCount  ,wordMaxlen: ',minCount,wordMaxlen)
	print ('doSampling,threshold : ',doSampling,threshold)
	
	return [freq_file,data_file,lang,minCount,wordMaxlen,doSampling,threshold]

pre_params = get_arguments()
train_data = preprocess(pre_params[0],pre_params[1],pre_params[2],pre_params[3],pre_params[4],pre_params[5],pre_params[6])