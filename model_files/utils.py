# -*- coding: utf-8 -*-
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

def getSyallables(word,lang):
	return syllabifier.orthographic_syllabify(word,lang)

def loadFile(file_name,isDict):
	print ('Loaded {}'.format(file_name))
	if isDict:
		return np.load(file_name).item()
	
	return np.load(file_name)

def str2bool(v):
	if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--lang'    , type=str)
	ap.add_argument('--save_dir', type=str)
	ap.add_argument('--idx2syl' , type=str)
	ap.add_argument('--idx2word', type=str)
	ap.add_argument('--train_words'   , type=str)
	ap.add_argument('--sampled_words' , type=str)
	ap.add_argument('--word2Sylidx_3' , type=str)
	ap.add_argument('--word2Sylidx_7' , type=str)
	ap.add_argument('--word2Sylidx_11', type=str)
	ap.add_argument('--word2Sylidx_3to7', type=str)
	ap.add_argument('--train_words_3to7'   , type=str)	
	ap.add_argument('--alpha' , type=float,default = 0.001)
	ap.add_argument('--lr'    , type=float,default = 0.05)
	ap.add_argument('--dim'   , type=int, default = 100)
	ap.add_argument('--ws'    , type=int, default = 5)
	ap.add_argument('--neg'   , type=int, default = 5)
	ap.add_argument('--epochs', type=int, default = 5)
	ap.add_argument('--state' , type=int, default = 0)
	ap.add_argument('--pretrain'  , type=str2bool, default = False)
	ap.add_argument('--doSampling', type=str2bool, default = True)
	
	print ('Parsing the Arguments')
	args = vars(ap.parse_args())
	
	doSampling = args['doSampling']
	if doSampling:
		train_words = loadFile(args['sampled_words'],False)
	else:
		train_words = loadFile(args['train_words'],False)
	
	pretrain  = args['pretrain']
	if pretrain:
		state = args['state']
	else:
		state = 0
	
	lr    = args['lr']
	ws    = args['ws']
	dim   = args['dim']
	neg   = args['neg']
	lang  = args['lang']
	alpha = args['alpha']
	epochs   = args['epochs']
	save_dir = args['save_dir']
	idx2syl  = loadFile(args['idx2syl'],True)
	idx2word = loadFile(args['idx2word'],True)
	word2Sylidx_3  = loadFile(args['word2Sylidx_3'],True)
	word2Sylidx_7  = loadFile(args['word2Sylidx_7'],True)
	word2Sylidx_11 = loadFile(args['word2Sylidx_11'],True)
	word2Sylidx_3to7 = loadFile(args['word2Sylidx_3to7'],True)
	train_words_3to7 = loadFile(args['train_words_3to7'],False)

	print ('Arguments Parsing Done!')
	print ('Arguments details: ')
	print ('lang: ',lang)
	print ('doSampling    : ',doSampling)
	print ('save_dir      : ',save_dir)
	print ('ws, dim, neg, : ',ws,dim,neg)
	print ('lr, epochs    : ',lr,epochs)
	print ('pretrain,state: ',pretrain,state)
	print ('alpha: ',alpha)
	
	return [idx2syl,idx2word,train_words,word2Sylidx_3,word2Sylidx_7,word2Sylidx_11,word2Sylidx_3to7,train_words_3to7],[lang,lr,dim,ws,neg,epochs,save_dir,pretrain,state,alpha]

def tofastextFormat(state,save_dir,file_name,stats_file):
weights_f   = np.load(save_dir+'/weights_{}.npz'.format(state))
all_weights = [weights_f[p] for p in sorted(weights_f.files,key=lambda s: int(s[4:]))]
idx2word    = np.load(stats_file+'/idx2word.npy').item()
emdeddings  = all_weights[0]

file        = open(file_name,'w')
file.write('{} {}'.format(emdeddings.shape[0],emdeddings.shape[1]))
for idx in idx2word:
	file.write('\n{} '.format(idx2word[idx]))
	vec = ' '.join(str(d) for d in emdeddings[idx])
	vec = vec.strip()
	file.write(vec)

file.close()

# tofastextFormat(2,'weights','skipgram.vec','stats')

# idx2syl  = loadFile(args['idx2syl'],True)
# idx2word = loadFile(args['idx2word'],True)
# word2Sylidx_3  = loadFile(args['word2Sylidx_3'],True)
# word2Sylidx_7  = loadFile(args['word2Sylidx_7'],True)
# word2Sylidx_11 = loadFile(args['word2Sylidx_11'],True)
# word2Sylidx_3to7 = loadFile(args['word2Sylidx_3to7'],True)
# train_words_3to7 = loadFile(args['train_words_3to7'],False)
