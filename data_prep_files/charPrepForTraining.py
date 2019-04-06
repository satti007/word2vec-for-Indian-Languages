'''
* @file charPrepForTraining.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Sat Mar 23 08:45:23 IST 2019
* @Contains code for chars preparing data for traning
'''

import sys
sys.path.insert(0, '../utilities_files/')

import re
import numpy as np
import commonFuctions
re_tel = re.compile(u'[^\u0C00-\u0C7F]+') # regex for only telugu characters

# function to create char2idx,idx2char dicts
def createCharTables(word2idx):
	ch_idx  = 0
	word2Chidx = {}
	char2idx,idx2char = {},{}
	for word in word2idx:
		word1 = re_tel.sub(r'',word)
		chars = list(word1)
		for ch in chars:
			if ch not in char2idx:
				char2idx[ch] = ch_idx
				idx2char[ch_idx] = ch
				ch_idx = ch_idx + 1
		
		word2Chidx[word2idx[word]] = [char2idx[ch] for ch in chars]
	
	return char2idx,idx2char,word2Chidx

def padChars_toMax(word2Chidx,char2idx,Min,Max):
	pad_ch        = len(char2idx)
	char_MintoMax = []
	word2Chidx_MintoMax = {}
	for word in word2Chidx:
		if len(word2Chidx[word]) >= Min and len(word2Chidx[word]) <= Max:
			chars = np.copy(word2Chidx[word]).tolist()
			chars.extend([pad_ch]*(Max-len(word2Chidx[word])))
			char_MintoMax.extend(chars)
			char_MintoMax = list(set(char_MintoMax))
			word2Chidx_MintoMax[word] = chars
	
	char_MintoMax.append(pad_ch)
	return word2Chidx_MintoMax,char_MintoMax


corpora_name = sys.argv[1]
train_file   = sys.argv[2]
Min, Max = int(sys.argv[3]),int(sys.argv[4])

data_dir = '../../../data/corpora/{}/stats/'.format(corpora_name)
word2idx = commonFuctions.load_npyFile(data_dir+'word2idx.npy',1)
char2idx,idx2char,word2Chidx = createCharTables(word2idx)
print ('The total number of chars are: ',len(char2idx))
commonFuctions.saveTofile(data_dir,'char2idx',char2idx)
commonFuctions.saveTofile(data_dir,'idx2char',idx2char)
commonFuctions.saveTofile(data_dir,'word2Chidx',word2Chidx)

train_words = commonFuctions.load_npyFile(data_dir+'{}.npy'.format(train_file),0)
word2Chidx_MintoMax,char_MintoMax = padChars_toMax(word2Chidx,char2idx,Min,Max)
train_words_MintoMax = [w for w in train_words if w in word2Chidx_MintoMax]
print ('The number of chars in words  with num of chars {}to{} are : {}'.format(Min,Max,len(char_MintoMax)))
print ('The number of words in vocabulary with num of chars {}to{} are : {}K'.format(Min,Max,len(word2Chidx_MintoMax)//pow(10,3)))
print ('The number of tokens in corpora   with num of chars {}to{} are : {}M'.format(Min,Max,len(train_words_MintoMax)//pow(10,6)))
commonFuctions.saveTofile(data_dir,'word2Chidx_{}to{}'.format(Min,Max),word2Chidx_MintoMax)
commonFuctions.saveTofile(data_dir,'train_words_{}to{}'.format(Min,Max),np.asarray(train_words_MintoMax))