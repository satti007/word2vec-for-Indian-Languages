import string
import numpy as np
import pandas as pd
from collections import Counter

WD_dir    = '../../data/corpora/webdunia/stats/'		# base directory for webdunia stats
AJ_dir    = '../../data/corpora/andhrajyothy/stats/'	# base directory for andhrajyothy stats
wiki_dir  = '../../data/corpora/wiki/stats/'			# base directory for wiki stats
files_dir = '../../data/corpora/all_combined/stats/'	# save all the stats to this directory

corpora_dir = [WD_dir,AJ_dir,wiki_dir] 					# lis of all corpora

word_doc    = open(files_dir+'vocab.txt','w')			# txt file to store (combined)vocabulary
punctuation = list(string.punctuation)					# list of special characters

# write stats to a file
def save_stats(stats):
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
def get_stats(word_to_freq,tot_sen,uni_sen):
	word_to_freq = dict(sorted(word_to_freq.items(),key=lambda x:x[1],reverse=True))
	tot_vocab  = len(word_to_freq)
	pun_vocab  = [w for w in punctuation if w in word_to_freq]
	tot_tokens = sum(word_to_freq.values())
	pun_tokens = sum(word_to_freq[w] for w in punctuation if w in word_to_freq)
	vocab_5    = sum(1 for f in word_to_freq.values() if f >= 5)
	vocab_10   = sum(1 for f in word_to_freq.values() if f >= 10)
	vocab_20   = sum(1 for f in word_to_freq.values() if f >= 20)
	
	save_stats([tot_sen,uni_sen,tot_vocab,pun_vocab,tot_tokens,pun_tokens,vocab_5,vocab_10,vocab_20])

# combine two dicts
def combineDict(word_to_freq,c_dict):
	com_dict   = Counter(word_to_freq) + Counter(c_dict)
	com_dict   = dict(sorted(com_dict.items(),key=lambda x:x[1],reverse=True))
	
	return com_dict

tot_sen   = 0				# Total num of sentences in all corpora
hash_sent = 0				# Total num of hashed sentences in all corpora
word_to_freq = {}			# Dict to store combined freq of a word in combined vocab

print('Starting combining vocabulary from different corporas...')
for c in corpora_dir:
	c_vocab = pd.read_csv(c +'word_to_freq.csv',header=None)
	c_dict  = dict(zip(list(c_vocab[0]),list(c_vocab[1])))
	print('Combining vocabulary from:  {} of vocab_size: {}'.format(c,len(c_dict)))
	word_to_freq = combineDict(word_to_freq,c_dict)
	print('After combining vocabulary, vocab_size: {}'.format(len(word_to_freq)))
	tot_sen = tot_sen + float(open(c+'stats.txt').readline().strip().split()[-1][0:-1])
	hash_sent = hash_sent + pd.read_csv(c +'hash_to_line.csv',header=None).shape[0]

pd.DataFrame.from_dict(data=word_to_freq,orient='index').to_csv(files_dir+'word_to_freq.csv',header=None)
word_doc.write('\n'.join(word_to_freq.keys()))
word_doc.write('\n')
word_doc.close()
print('Done!!')

print('Total nuber of sentences:{}M'.format(tot_sen))

tot_sen = tot_sen*pow(10,6)					
uni_sen = hash_sent			 # No overlapping sentences btw corpora (can be know by below commented snippet)
get_stats(word_to_freq,tot_sen,uni_sen)



# TO CHECK INTERSECTION OF SENTENCES BTW CORPORA
'''
over_sen = 0
remove_sen = {}
for i in range(0,len(corpora_dir)):
	del_l = []
	for j in range(i+1,len(corpora_dir)):
		c_i   = list(pd.read_csv(corpora_dir[i]+'hash_to_line.csv',header=None)[0])
		c_j   = list(pd.read_csv(corpora_dir[j]+'hash_to_line.csv',header=None)[0])
		c_ij  = list(set(c_i).intersection(c_j))
		del_l = list(set(del_l).union(c_ij))
	
	remove_sen[corpora_dir[i]] = del_l

for rs in remove_sen:
	over_sen = over_sen + len(remove_sen[rs])

print('Total nuber of overlapping sentences:{}'.format(over_sen))
'''