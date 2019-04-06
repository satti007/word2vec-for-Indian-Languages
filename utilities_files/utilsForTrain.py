'''
* @file utilsForTrain.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Tue Apr 2 03:21:54 IST 2019
* @Contains helper functions to run train.py
'''

import argparse
import numpy as np
import commonFuctions

def str2bool(v):
	if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def parseArguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--unit'    , type=str, choices=['char','syl'])
	ap.add_argument('--model'   , type=str, choices=['CNN' ,'Mean'])
	ap.add_argument('--maxlen'  , type=int)
	ap.add_argument('--idx2unit', type=str)
	ap.add_argument('--idx2word', type=str)
	ap.add_argument('--word2Unitidx', type=str)
	ap.add_argument('--num_layers'  , type=int)
	ap.add_argument('--lr'    , type=float, default = 1e-4)
	ap.add_argument('--ws'    , type=int, default = 5)
	ap.add_argument('--neg'   , type=int, default = 10)
	ap.add_argument('--epochs', type=int, default = 20)
	ap.add_argument('--embd_dim'   , type=int, default = 300)
	ap.add_argument('--rep_dim'    , type=int, default = 300)
	ap.add_argument('--batch_size' , type=int, default = 1000)
	ap.add_argument('--train_words', type=str)
	ap.add_argument('--save_dir'   , type=str)
	ap.add_argument('--pretrain'   , type=str2bool, default = False)
	ap.add_argument('--state'      , type=int     , default = None)
	ap.add_argument('--l2reg'      , type=str2bool, default = False)
	ap.add_argument('--beta'       , type=float   , default = 1e-3)
	
	print ('Parsing the Arguments')
	args = vars(ap.parse_args())
	
	unit     = args['unit']
	model    = args['model']
	maxlen   = args['maxlen']
	num_layers = args['num_layers']
	
	lr      = args['lr']
	ws      = args['ws']
	neg     = args['neg']
	epochs  = args['epochs']
	embd_dim   = args['embd_dim']
	rep_dim    = args['rep_dim']
	batch_size = args['batch_size']
	
	save_dir = args['save_dir']
	pretrain = args['pretrain']
	state    = args['state']
	l2reg    = args['l2reg']
	beta     = args['beta']
	
	print ('Arguments Parsing Done!')
	print ('Arguments details   : ')
	print ('unit,model,maxlen       : ',unit,model,maxlen)
	print ('lr,ws,neg,num_layers    : ',lr,ws,neg,num_layers)
	print ('epochs,batch_size       : ',epochs,batch_size)
	print ('embd_dim,rep_dim        : ',embd_dim,rep_dim)
	print ('save_dir,pretrain,state : ',pretrain,state,save_dir)
	print ('l2reg,beta : ',l2reg,beta)

	idx2unit     = commonFuctions.load_npyFile(args['idx2unit']     ,True)
	idx2word     = commonFuctions.load_npyFile(args['idx2word']     ,True)
	word2Unitidx = commonFuctions.load_npyFile(args['word2Unitidx'] ,True)
	train_words  = commonFuctions.load_npyFile(args['train_words']  ,False)
	train_words  = train_words.tolist()
	
	return [idx2unit,idx2word,word2Unitidx,train_words],[unit,model,maxlen,lr,ws,neg,epochs,embd_dim,rep_dim,batch_size,save_dir,pretrain,state,l2reg,beta,num_layers]

def getTarget_word(words,idx,ws):
	word_ws = np.random.randint(1,ws+1)
	start   = idx - word_ws if (idx - word_ws) > 0 else 0
	stop    = idx + word_ws
	target_words = set(words[start:idx] + words[idx+1:stop+1])
	
	return list(target_words)

def get_batches(train_words,word2Unitidx,batch_size,ws):
	for idx in range(0,len(train_words),batch_size):
		x, y  = [], []
		batch = train_words[idx:idx+batch_size]
		for i in range(len(batch)):
			batch_x = batch[i]
			batch_y = getTarget_word(batch,i,ws)
			y.extend(batch_y)
			x.extend([word2Unitidx[batch_x]]*len(batch_y))
		yield np.array(x), np.array(y)[:,None]