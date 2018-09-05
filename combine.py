import string
import numpy as np
import pandas as pd
from collections import Counter

wiki_dir  = '../../data/corpora/wiki/stats/'						# base directory for wiki articles
AJ_dir    = '../../data/corpora/andhrajyothy/stats/'				# base directory for andhrajyothy articles

files_dir = '../../data/corpora/combined_stats/'

word_doc  = open(files_dir+'vocab.txt','w')		# txt file to store vocabulary
punctuation = list(string.punctuation)			# list of special characters

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

def combine_vocab(AJ_vocab,wiki_vocab):
	AJ_dict    = dict(zip(list(AJ_vocab[0]),   list(AJ_vocab[1])))
	wiki_dict  = dict(zip(list(wiki_vocab[0]), list(wiki_vocab[1])))
	all_dict   = Counter(AJ_dict) + Counter(wiki_dict)
	all_dict   = dict(sorted(all_dict.items(),key=lambda x:x[1],reverse=True))
	pd.DataFrame.from_dict(data=all_dict,orient='index').to_csv(files_dir+'word_to_freq.csv',header=None)
	word_doc.write('\n'.join(all_dict.keys()))
	word_doc.write('\n')
	
	return all_dict

AJ_vocab     = pd.read_csv(AJ_dir  +'word_to_freq.csv',header=None)
wiki_vocab   = pd.read_csv(wiki_dir+'word_to_freq.csv',header=None)
word_to_freq = combine_vocab(AJ_vocab,wiki_vocab)
tot_sen = 74.34*pow(10,6)
uni_sen = 10.21*pow(10,6)
get_stats(word_to_freq,tot_sen,uni_sen)

# 1 -- andhrajyothy, 2 -- wiki
# sentences in both sources(1&2) are unique, checked this ...
# by comparing the hash values of each sentecne
# tot_sen = tot_sen1 + tot_sen2 
# uni_sen = uni_sen1 + uni_sen2 