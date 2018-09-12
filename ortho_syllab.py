# -*- coding: utf-8 -*-

import pickle
import pandas as pd

INDIC_NLP_LIB_HOME  = '/DDP/src/indic_nlp_library'
INDIC_NLP_RESOURCES = '/DDP/data/indic_nlp_resources'

import sys
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp import loader
loader.load()

from indicnlp.syllable import  syllabifier

def getSyallables(word,lang):
	return syllabifier.orthographic_syllabify(word.decode('utf-8'),lang)

# getSyallables('నాలుగు','te')




# BELOW CODE IS FOR GETTING SYALLANLES FOR EACH WORD IN VOCAB
'''
INDIC_NLP_LIB_HOME  = '/home/satti/Documents/DDP/src/indic_nlp_library'
INDIC_NLP_RESOURCES = '/home/satti/Documents/DDP/data/indic_nlp_resources'
TELUGU_VOCAB        = '/home/satti/Documents/DDP/data/corpora/combined_stats/'

lang  = 'te'
vocab = open(TELUGU_VOCAB+'vocab.txt','r').readlines()
vocab = [v.strip() for v in vocab]

min_len     = 100
max_len     = 0
tot_syllables     = {}
word_to_syllables = {}
print 'Orthographic Syllabification for vocabulary...'
for idx,word in enumerate(vocab):
	print idx,word
	syllables = syllabifier.orthographic_syllabify(word.decode('utf-8'),lang)
	syllables = [syl.encode('utf-8') for syl in syllables]
	word_to_syllables[word] = syllables
	tot_syllables.update({syl:1 for syl in syllables if syl not in tot_syllables})
	if min_len>len(syllables):
		min_len = len(syllables)
	if max_len<len(syllables):
		max_len = len(syllables)
	if idx == 1000:
		break

print 'Done!'
print 'The total num of syllables are    : ',len(tot_syllables)
print 'Min number of syllables in a word : ',min_len
print 'Max number of syllables in a word : ',max_len

pd.DataFrame.from_dict(data=tot_syllables,orient='index').to_csv(TELUGU_VOCAB+'tot_syllables.csv',header=None)
pd.DataFrame.from_dict(data=word_to_syllables,orient='index').to_csv(TELUGU_VOCAB+'word_to_syllables.csv',header=None)
with open(TELUGU_VOCAB+'word_to_syllables.pickle', 'wb') as handle:
	pickle.dump(word_to_syllables, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(TELUGU_VOCAB+'word_to_syllables.pickle', 'rb') as handle:
    word_to_syllables = pickle.load(handle)
'''