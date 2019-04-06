'''
* @file combineStatsFromallCoarpora.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Fri Aug 31 06:18:54 IST 2018
* @Contains code to combine stats(vocab, senetences, etc) from all corpora
'''

import os
import sys
sys.path.insert(0, '../utilities_files/')

import string
import numpy as np
import pandas as pd
import commonFuctions

wiki_dir  = '../../../data/corpora/wiki/stats/'			# base directory for wiki stats
WD_dir    = '../../../data/corpora/webdunia/stats/'		# base directory for webdunia stats
AJ_dir1   = '../../../data/corpora/andhrajyothy/stats/'	# base directory for andhrajyothy stats
files_dir = '../../../data/corpora/all_combined/stats/'	# save all the stats to this directory
corpora_dir = [WD_dir,AJ_dir,wiki_dir] 					# lis of all corpora

tot_sen   = 0											# Total num of sentences in all corpora
hash_sent = 0											# Total num of hashed sentences in all corpora
word2freq = {}											# Dict to store combined freq of a word in combined vocab

word_doc    = open(files_dir+'vocab.txt','wb',0)		# txt file to store (combined)vocabulary
sent_doc    = open(files_dir+'sentences.txt','wb',0)	# txt file to write sentences
punctuation = list(string.punctuation)					# list of special characters

print('Starting combining vocabulary from different corporas...')
for c in corpora_dir:
	c_vocab   = pd.read_csv(c +'word2freq.csv',header=None)
	c_dict    = dict(zip(list(c_vocab[0]),list(c_vocab[1])))
	word2freq = commonFuctions.combineDict(word2freq,c_dict,c)
	tot_sen   = tot_sen   + float(open(c+'stats.txt').readline().strip().split()[-1][0:-1])
	hash_sent = hash_sent + pd.read_csv(c +'line2hash.csv',header=None).shape[0]
	print ('Merging sentences.txt from '+ c)
	os.system('cat {}sentences.txt {}sentences.txt >> {}sentences.txt'.format(c,files_dir,files_dir))
	print('Total number of sentences after Merging: {}M'.format(tot_sen))
	print('\n')

pd.DataFrame.from_dict(data=word2freq,orient='index').to_csv(files_dir+'word2freq.csv',header=None)
word_doc.write(('\n'.join(word2freq.keys())).encode('utf-8'))
word_doc.write('\n'.encode('utf-8'))
word_doc.close()

tot_sen = tot_sen*pow(10,6)
uni_sen = hash_sent	
commonFuctions.get_stats(word2freq,tot_sen,uni_sen,files_dir,punctuation)
print('Done!!')


# TO CHECK INTERSECTION OF SENTENCES BTW CORPORA
'''
over_sen = 0
remove_sen = {}
for i in range(0,len(corpora_dir)):
	del_l = []
	for j in range(i+1,len(corpora_dir)):
		c_i   = list(pd.read_csv(corpora_dir[i]+'line2hash.csv',header=None)[0])
		c_j   = list(pd.read_csv(corpora_dir[j]+'line2hash.csv',header=None)[0])
		c_ij  = list(set(c_i).intersection(c_j))
		del_l = list(set(del_l).union(c_ij))
	
	remove_sen[corpora_dir[i]] = del_l

for rs in remove_sen:
	over_sen = over_sen + len(remove_sen[rs])

print('Total nuber of overlapping sentences:{}'.format(over_sen))
'''