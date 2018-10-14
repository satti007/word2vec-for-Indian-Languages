# -*- coding: utf-8 -*-
# BELOW CODE IS FOR GETTING SYALLANLES FOR EACH WORD(with count >= cut-off) IN VOCAB (to know min and max number of syllables)
################################################################################################################################
INDIC_NLP_LIB_HOME  = '../indic_nlp_library'										# path to the library source
INDIC_NLP_RESOURCES = '../../data/indic_nlp_resources'								# path to the resources neede by the library
files_dir           = '../../data/corpora/combined_stats/'							# path to vocabulary file

# loading the library
import sys
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp import loader
loader.load()

from indicnlp.syllable import  syllabifier

vocab        = pd.read_csv(files_dir +'word_to_freq.csv',header=None)				# load the word_to_freq file
word_to_freq = dict(zip(list(vocab[0])  ,list(vocab[1])))							# convert the DataFrame to a dcit(word:freq) 

cutoff = [1,2,5,10,20]					# cut-off for freq of words(i.e get syllables for only words with count>=cutoff)
lang   = 'te'							# parameter for indic_nlp library

for c in cutoff:
	min_syll_len,min_W   = 100,''		# min number of syllables of a word in vocab
	max_syll_len,max_W   = 0,''			# max number of syllables of a word in vocab
	num_words      = 0					# number of words with count >= cutoff 
	tot_syllables  = {}					# to get total number of unique-syllables in vocabulary...
										# ... using dict for fastlookup
	
	print ('Started syllabification  of words with freq>={} in vocabulary...'.format(c))
	for idx, word in enumerate(word_to_freq):
		if word_to_freq[word] < c: 		# ignore then word if it doesn't satisfy the cutoff
			continue
		syllables = syllabifier.orthographic_syllabify(word.decode('utf-8'),lang)		# syllabification of word
		tot_syllables.update({syl:1 for syl in syllables if syl not in tot_syllables})	# update the dict
		if min_syll_len>len(syllables):													# update the min_syll_len
			min_syll_len,min_W = len(syllables),word
		if max_syll_len<len(syllables):													# update the max_syll_len
			max_syll_len,max_W = len(syllables),word
		num_words = num_words + 1														# update num_words
		# if (idx+1)%10==0:
		# 	break
	
	print ('Min number of syllables in a word({}) :{} '.format(min_W,min_syll_len))
	print ('Max number of syllables in a word({}) :{} '.format(max_W,max_syll_len))
	print ('The total num of syllables are    : ',len(tot_syllables))
	print ('The total num of words syllabified: ',num_words)
	print ('Done!')
	print ('#####\n')
################################################################################################################################