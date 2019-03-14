# -*- coding: utf-8 -*-
import argparse
import numpy as np

def loadFile(file_name,isDict):
	print ('Loaded {}'.format(file_name))
	if isDict:
		return np.load(file_name).item()
	
	return np.load(file_name).tolist()

def str2bool(v):
	if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--idx2sylzz' , type=str)
	ap.add_argument('--idx2word', type=str)
	ap.add_argument('--save_dir', type=str)
	ap.add_argument('--train_words', type=str)
	ap.add_argument('--word2Sylidx', type=str)
	ap.add_argument('--lr'    , type=float,default = 0.005)
	ap.add_argument('--dim'   , type=int, default = 300)
	ap.add_argument('--ws'    , type=int, default = 5)
	ap.add_argument('--neg'   , type=int, default = 10)
	ap.add_argument('--epochs', type=int, default = 20)
	ap.add_argument('--state' , type=int, default = 0)
	ap.add_argument('--max_syl'   , type=int, default = 7)
	ap.add_argument('--rep_dim'   , type=int, default = 300)
	ap.add_argument('--batch_size', type=int, default = 1000)
	ap.add_argument('--pretrain'  , type=str2bool, default = False)
	
	print ('Parsing the Arguments')
	args = vars(ap.parse_args())
	
	lr      = args['lr']
	ws      = args['ws']
	dim     = args['dim']
	neg     = args['neg']
	epochs  = args['epochs']
	max_syl = args['max_syl']
	rep_dim = args['rep_dim']
	batch_size = args['batch_size']
	
	pretrain = args['pretrain']
	save_dir = args['save_dir']
	state    = args['state']
	
	print ('Arguments Parsing Done!')
	print ('Arguments details : ')
	print ('lr, ws, dim, neg  : ',lr,ws,dim,neg)
	print ('max_syl, rep_dim  : ',max_syl,rep_dim)
	print ('epochs, batch_size: ',epochs,batch_size)
	
	print ('pretrain,state: ',pretrain,state)
	print ('save_dir      : ',save_dir)
	
	idx2syl     = loadFile(args['idx2syl'] ,True)
	idx2word    = loadFile(args['idx2word'],True)
	train_words = loadFile(args['train_words'],False)
	word2Sylidx = loadFile(args['word2Sylidx'],True)
	
	return [idx2syl,idx2word,train_words,word2Sylidx],[lr,ws,dim,neg,epochs,batch_size,max_syl,rep_dim,pretrain,state,save_dir]

'''
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
'''
# tofastextFormat(2,'weights','skipgram.vec','stats')
# idx2syl  = loadFile(args['idx2syl'],True)
# idx2word = loadFile(args['idx2word'],True)
# word2Sylidx_3  = loadFile(args['word2Sylidx_3'],True)
# word2Sylidx_7  = loadFile(args['word2Sylidx_7'],True)
# word2Sylidx_11 = loadFile(args['word2Sylidx_11'],True)
# word2Sylidx_3to7 = loadFile(args['word2Sylidx_3to7'],True)
# train_words_3to7 = loadFile(args['train_words_3to7'],False)
