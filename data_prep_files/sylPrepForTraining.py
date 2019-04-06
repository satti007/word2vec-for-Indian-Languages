'''
* @file sylPrepForTraining.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Sat Mar 23 08:45:23 IST 2019
* @Contains code for syllables preparing data for traning
'''

import sys
sys.path.insert(0, '../utilities_files/')

import numpy as np
import pandas as pd
import commonFuctions
import syllabification

np.random.seed(1234)

# function to create syl2idx, idx2syl dicts
def createSyllableTables(word2idx,lang):
	syl_idx  = 0
	word2Sylidx = {}
	syl2idx,idx2syl = {},{}
	for word in word2idx:
		syllables = syllabification.getSyllables(word,lang)
		for syl in syllables:
			if syl not in syl2idx:
				syl2idx[syl] = syl_idx
				idx2syl[syl_idx] = syl
				syl_idx = syl_idx + 1
		
		word2Sylidx[word2idx[word]] = [syl2idx[syl] for syl in syllables]
	
	return syl2idx,idx2syl,word2Sylidx

def padSyllables_toMax(word2Sylidx,syl2idx,Min,Max):
	pad_syl      = len(syl2idx)
	syl_MintoMax = []
	word2Sylidx_MintoMax = {}
	for word in word2Sylidx:
		if len(word2Sylidx[word]) >= Min and len(word2Sylidx[word]) <= Max:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(Max-len(word2Sylidx[word])))
			syl_MintoMax.extend(syllables)
			syl_MintoMax = list(set(syl_MintoMax))
			word2Sylidx_MintoMax[word] = syllables
	
	syl_MintoMax.append(pad_syl)
	return word2Sylidx_MintoMax,syl_MintoMax

corpora_name = sys.argv[1]
train_file   = sys.argv[2]
Min, Max = int(sys.argv[3]),int(sys.argv[4])

lang = 'te'
data_dir = '../../../data/corpora/{}/stats/'.format(corpora_name)
word2idx = commonFuctions.load_npyFile(data_dir+'word2idx.npy',1)
syl2idx,idx2syl,word2Sylidx = createSyllableTables(word2idx,lang)
print ('The total number of syllables are: ',len(syl2idx))
commonFuctions.saveTofile(data_dir,'syl2idx',syl2idx)
commonFuctions.saveTofile(data_dir,'idx2syl',idx2syl)
commonFuctions.saveTofile(data_dir,'word2Sylidx',word2Sylidx)

train_words = commonFuctions.load_npyFile(data_dir+'{}.npy'.format(train_file),0)
word2Sylidx_MintoMax,syl_MintoMax = padSyllables_toMax(word2Sylidx,syl2idx,Min,Max)
train_words_MintoMax = [w for w in train_words if w in word2Sylidx_MintoMax]
print ('The number of syllables in words  with num of syll {}to{} are : {}'.format(Min,Max,len(syl_MintoMax)))
print ('The number of words in vocabulary with num of syll {}to{} are : {}K'.format(Min,Max,len(word2Sylidx_MintoMax)//pow(10,3)))
print ('The number of tokens in corpora   with num of syll {}to{} are : {}M'.format(Min,Max,len(train_words_MintoMax)//pow(10,6)))
commonFuctions.saveTofile(data_dir,'word2Sylidx_{}to{}'.format(Min,Max),word2Sylidx_MintoMax)
commonFuctions.saveTofile(data_dir,'train_words_{}to{}'.format(Min,Max),np.asarray(train_words_MintoMax))





'''
# function to pad the syllables to make a batch of words have same no.of.syllables
# words with 1,2,3     num of syllables, pad them so that all of them become 3 syllabi words
# words with 4,5,6,7   num of syllables, pad them so that all of them become 7 syllabi words
# words with 8,9,10,11 num of syllables, pad them so that all of them become 11 syllabi words
def padSyllables_3c7c11(word2Sylidx,syl2idx):
	pad_syl = len(syl2idx)
	word2Sylidx_3  = {}
	word2Sylidx_7  = {}
	word2Sylidx_11 = {}
	for word in word2Sylidx:
		if len(word2Sylidx[word]) <= 3:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(3-len(word2Sylidx[word])))
			word2Sylidx_3[word] = syllables
		elif len(word2Sylidx[word]) <= 7:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(7-len(word2Sylidx[word])))
			word2Sylidx_7[word] = syllables
		else:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(11-len(word2Sylidx[word])))
			word2Sylidx_11[word] = syllables
		
	return word2Sylidx_3,word2Sylidx_7,word2Sylidx_11

word2Sylidx_3,word2Sylidx_7,word2Sylidx_11 = padSyllables_3c7c11(word2Sylidx,syl2idx)
commonFuctions.saveTofile(save_dir,'word2Sylidx_3' ,word2Sylidx_3)
commonFuctions.saveTofile(save_dir,'word2Sylidx_7' ,word2Sylidx_7)
commonFuctions.saveTofile(save_dir,'word2Sylidx_11',word2Sylidx_11)
'''