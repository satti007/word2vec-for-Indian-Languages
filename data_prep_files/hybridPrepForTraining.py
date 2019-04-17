'''
* @file hybridPrepForTraining.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Tue Apr 16 03:24:04 IST 2019
* @Contains code for hybrid (syl + char) preparing data for traning
'''

import sys
sys.path.insert(0, '../utilities_files/')

import re
import numpy as np
import commonFuctions
import syllabification
re_tel = re.compile(u'[^\u0C00-\u0C7F]+') # regex for only telugu characters

def padHybrid_toMax(word2Hybidx,pad_hyb,Min,Max):
	word2Hybidx_MintoMax = {}
	for word in word2Hybidx:
		if len(word2Hybidx[word]) >= Min and len(word2Hybidx[word]) <= Max:
			units = np.copy(word2Hybidx[word]).tolist()
			units.extend([pad_hyb]*(Max-len(word2Hybidx[word])))
			word2Hybidx_MintoMax[word] = units
	
	return word2Hybidx_MintoMax

def createHybridTables(word2idx,hyb2idx,idx2hyb,hyb_sylcount,lang):
	hyb_idx     = len(hyb2idx)
	word2Hybidx = {}
	for word in word2idx:
		hyb    = []
		word1  = re_tel.sub(r'',word)
		syllables = syllabification.getSyllables(word1,lang)
		for syl in syllables:
			if syl in hyb2idx:
				hyb.append(hyb2idx[syl])
			else:
				syl_ch = list(syl)
				for ch in syl_ch:
					if ch not in hyb2idx:
						hyb2idx[ch] = hyb_idx
						idx2hyb[hyb_idx]  = ch
						hyb_idx += 1
					hyb.append(hyb2idx[ch])
		
		word2Hybidx[word2idx[word]] = hyb
	
	return word2Hybidx,hyb2idx,idx2hyb,hyb_idx - hyb_sylcount

def get_cutoffSyllables(syl2count,idx2syl):
	hyb_idx = 0
	hyb2idx,idx2hyb = {},{}
	for syl in syl2count:
		if syl2count[syl] >= cutoff and syl != len(idx2syl):
			hyb2idx[idx2syl[syl]] = hyb_idx
			idx2hyb[hyb_idx]  = idx2syl[syl]
			hyb_idx += 1
	
	return hyb2idx,idx2hyb,hyb_idx

def get_syllablesCount(word2Sylidx):
	syl2count    = {}
	for word in word2Sylidx:
		wordSyl = word2Sylidx[word]
		for syl in wordSyl:
			if syl in syl2count:
				syl2count[syl] += 1
			else:
				syl2count[syl] = 1
	
	return syl2count


lang,cutoff  = 'te',20
corpora_name = sys.argv[1]
train_file   = sys.argv[2]
Min, Max     = int(sys.argv[3]),int(sys.argv[4])
data_dir     = '../../../data/corpora/{}/stats/'.format(corpora_name)
idx2syl      = commonFuctions.load_npyFile(data_dir+'idx2syl.npy',1)
word2idx     = commonFuctions.load_npyFile(data_dir+'word2idx.npy',1)
word2Sylidx  = commonFuctions.load_npyFile(data_dir+'word2Sylidx_1to4.npy',1)
train_words  = commonFuctions.load_npyFile(data_dir+'{}.npy'.format(train_file),0)

syl2count    = get_syllablesCount(word2Sylidx)
hyb2idx,idx2hyb,hyb_sylcount = get_cutoffSyllables(syl2count,idx2syl)
word2Hybidx,hyb2idx,idx2hyb,hyb_charcount = createHybridTables(word2idx,hyb2idx,idx2hyb,hyb_sylcount,lang)

word2Hybidx_MintoMax = padHybrid_toMax(word2Hybidx,len(hyb2idx),Min,Max)
train_words_MintoMax = [w for w in train_words if w in word2Hybidx_MintoMax]

commonFuctions.saveTofile(data_dir,'hyb2idx' ,hyb2idx)
commonFuctions.saveTofile(data_dir,'idx2hyb' ,idx2hyb)
commonFuctions.saveTofile(data_dir,'word2Hybidx',word2Hybidx)
commonFuctions.saveTofile(data_dir,'word2Hybidx_{}to{}'.format(Min,Max),word2Hybidx_MintoMax)
commonFuctions.saveTofile(data_dir,'train_words_{}to{}'.format(Min,Max),np.asarray(train_words_MintoMax))

print ('The total number of syllables  in hybrid model are : {}'.format(hyb_sylcount))
print ('The total number of characters in hybrid model are : {}'.format(hyb_charcount))
print ('The number of words in vocabulary within hybrid limits are : {}K'.format(len(word2Hybidx_MintoMax)//pow(10,3)))
print ('The number of tokens in corpora   within hybrid limits are : {}M'.format(len(train_words_MintoMax)//pow(10,6)))