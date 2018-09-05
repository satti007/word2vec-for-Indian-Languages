# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from polyglot.text import Text

wiki_dir  = '../../data/corpora/wiki/stats/'					# base directory for wiki articles
AJ_dir    = '../../data/corpora/andhrajyothy/stats/'			# base directory for andhrajyothy articles

files_dir =  wiki_dir											# AJ_dir (or) wiki_dir | processing one source at a time 
re_tel    = re.compile(u'[^\u0C00-\u0C7F]+')					# regex for only telugu 'utf-8' encoding

def save_dicts(bigram_to_freq):
	bigram_to_freq = dict(sorted(bigram_to_freq.items(),key=lambda x:x[1],reverse=True))
	pd.DataFrame.from_dict(data=bigram_to_freq,orient='index').to_csv(files_dir+'bigram_to_freq.csv',header=None)

text  = open(files_dir+'sentences.txt','r').read()
text  = Text(text)
words = text.words
bigram_to_freq = {}
print ('Total number of words: {}'.format(len(words)))

for i in range(0,len(words)-1):
	w1,w2 = words[i],words[i+1]
	if len(re_tel.sub(r'',w1)) or len(re_tel.sub(r'',w2)):
		if (w1,w2) not in bigram_to_freq:
			bigram_to_freq[(w1,w2)] = 1
		else:
			bigram_to_freq[(w1,w2)] += 1
	if (i+1)%100 == 0:
		print ('word_count:{}, size of bigrams: {}'.format(i+1,len(bigram_to_freq)))
	if (i+1)%10000 == 0:
		save_dicts(bigram_to_freq)

save_dicts(bigram_to_freq)

# word_to_freq.sort_values(1,ascending=False,inplace=True)