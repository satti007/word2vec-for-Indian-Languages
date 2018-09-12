# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

files_dir = '../../data/corpora/combined_stats/'						# base directory for frequency(unigram & bigram) files

vocab      = pd.read_csv(files_dir +'word_to_freq.csv',header=None)
bigrams    = pd.read_csv(files_dir +'bigram_to_freq.csv',header=None)
phrases_doc    = open(files_dir+'phrases.txt','w')
word_to_freq   = dict(zip(list(vocab[0])  ,list(vocab[1])))					# dict - {word:count(word)}
bigram_to_freq = dict(zip(list(bigrams[0]),list(bigrams[1])))				# dict - {(w1,w2):count(w1,w2)}
tot_tokens     = sum(word_to_freq.values()) 								# total number of tokens

passes     = 1 				# No.of iterations over the training data to allow longer phrases formation
discount   = 5				# discard unigram & bigrams that appear < discount
threshold  = 50				# threshold for forming the phrases
decrement  = 10				# decrease threshold value by decrement after each pass to allow longer phrases formation

# check if unigrams(w1&w2) & bigrams appear < discount
def OVV(bigram,w1,w2):
	if bigram_to_freq[bigram]  < discount:
		return 1
	if word_to_freq[w1] < discount:
		return 1
	if word_to_freq[w2] < discount:
		return 1
	
	return 0

# caluclate score used for phrasing
def calScore(bigram,w1,w2):
	print ('bigram: ',bigram_to_freq[bigram])
	print ('w1: ',word_to_freq[w1])
	print ('w2: ',word_to_freq[w2])
	return ((float)(bigram_to_freq[bigram]/word_to_freq[w1])/word_to_freq[w2])*tot_tokens

# check if the the word is a int/float
def checkNum(word):
	W = word.replace('"','').replace(',','').replace('.','') 	# remove '"' ',' '.'  
	try:
		x = float(W)
		return 1 
	except ValueError:
		return 0

iter  = 0
bigram_count = 0
phrase_count = 0
while (iter < passes):
	print('Starting iteration {} with threshold {}...'.format(iter,threshold))
	for bigram in bigram_to_freq:
		bigram_count = bigram_count + 1
		if bigram_count%100000 == 0:
			print('bigrams_count: {}, phrase_count: {}'.format(bigram_count,phrase_count)) 
			break
		w1, w2 = bigram[1:-1].split(',')[0].strip()[1:-1],bigram[1:-1].split(',')[-1].strip()[1:-1] 
		if len(w1) and len(w2):
			print('bigram:{}, w1:{}, w2:{}'.format(bigram,w1,w2))
			if checkNum(w1):
				continue
			if checkNum(w2):
				continue
			if OVV(bigram,w1,w2):
				continue
			score = calScore(bigram,w1,w2)
			print (score)
			if score < threshold:
				continue
			phrase = w1 + '_' + w2 + '\n'
			phrases_doc.write(phrase)
			phrase_count = phrase_count + 1
	print('End of iteration {} with threshold {}'.format(iter,threshold))
	iter = iter+ 1
	threshold = threshold - decrement

phrases_doc.close()