# -*- coding: utf-8 -*-
# phrase learning on sentences formed after preprocessing the articles
# also phrases of only words, no special chars i.e phrase = w1_'.'  is not formed
# only one pass with threshold = 50
# later run this directly on the articles with decreasing threshold for 2-4 passes

import os
import re
import string
import numpy as np
import pandas as pd
from polyglot.text import Text

# files_dir = '../../data/corpora/combined_stats/'	   # base directory for frequency(unigram & bigram) files
files_dir = 'stats/'	  							   # base directory for frequency(unigram & bigram) files
re_tel    = re.compile(u'[^\u0C00-\u0C7F]+')		   # regex for only telugu characters

passes     = 1 			# No.of iterations over the training data to allow longer phrases formation
discount   = 5			# discard unigram & bigrams that appear < discount
threshold  = 500		# threshold for forming the phrases
decrement  = 10			# decrease threshold value by decrement after each pass to allow longer phrases formation

# add the phrase to the word2freq dict
def update_Word2freq(word_to_freq,phrase_words):
	for i in range(len(phrase_words)):
		w1,w2  = phrase_words[i][0],phrase_words[i][1] 
		phrase = w1 + '_' + w2 
		word_to_freq[phrase] = int((word_to_freq[w1]+word_to_freq[w2])/2)
	word_to_freq   = dict(sorted(word_to_freq.items()  ,key=lambda x:x[1],reverse=True))
	pd.DataFrame.from_dict(data=word_to_freq,orient='index').to_csv(files_dir+'word_to_freq.csv',header=None)

# caluclate score used for phrasing
def calScore(bigram,w1,w2,bigram_to_freq,word_to_freq):
	return ((float)(bigram_to_freq[bigram]/word_to_freq[w1])/word_to_freq[w2])*sum(word_to_freq.values())

# check if unigrams(w1&w2) & bigrams appear < discount
def OVV(bigram,w1,w2,bigram_to_freq,word_to_freq):
	try: 
		if bigram_to_freq[bigram] < discount:
			return 1
	except:
		return 1
	if word_to_freq[w1] < discount:
		return 1
	if word_to_freq[w2] < discount:
		return 1
	
	return 0

# learn phrases
def doPhrases(sentences,word_to_freq,bigram_to_freq):
	phrase_count  = 0													# no.of phrases formed
	phrase_words  = []													# list of tuples of words forming phrases
	sent_doc   = open(files_dir+'sentences_{}.txt'.format(iter+1),'w')	# txt file to write updated sentences(with phrases) 
	phrase_doc = open(files_dir+'phrases_{}.txt'.format(iter+1),'w')	# txt file to write phrases learnt after iter
	for idx, sent in enumerate(sentences):
		sent  = sent.strip()																# remove '\n'
		Words = Text(sent).words															# tokenize the sentence
		# Words = sent.split()																# tokenize the sentence
		sent_words = []
		for i in range(0,len(Words)-1):
			w1,w2 = Words[i],Words[i+1]
			if (w1,w2) in phrase_words:
				sent_words.append((w1,w2))
			else:
				if len(re_tel.sub(r'',w1)) and len(re_tel.sub(r'',w2)):
					if OVV((w1,w2),w1,w2,bigram_to_freq,word_to_freq):
						continue
					score = calScore((w1,w2),w1,w2,bigram_to_freq,word_to_freq)
					if score < threshold:
						continue
					sent_words.append((w1,w2))
					phrase_words.append((w1,w2))
					phrase = w1 + '_' + w2 + '\n'
					phrase_doc.write(phrase)
					phrase_count = phrase_count + 1
					print('For w1:{}, w2:{}, score:{}'.format(w1,w2,score))
			
		for j in range(0,len(sent_words)):
			phrase1 = sent_words[j][0] + ' ' + sent_words[j][1]
			phrase2 = sent_words[j][0] + '_' + sent_words[j][1]
			sent.replace(phrase1,phrase2)		
		sent_doc.write(sent+'\n')
		
		if (idx+1)%100000 == 0:
			print('Processed {} sentences, phrase_count: {}'.format((idx+1),phrase_count))
			# break 
	sent_doc.close()
	phrases_doc.close()
	update_Word2freq(word_to_freq,phrase_words)

# delete bigrams with count=1
def rmBigrams(bigram_to_freq):
	return {k:v for k,v in bigram_to_freq.items() if v>1}

# bigram_count
def updateCount(words,bigram_to_freq):
	for i in range(0,len(words)-1):
		w1,w2 = words[i],words[i+1]
		if len(re_tel.sub(r'',w1)) and len(re_tel.sub(r'',w2)):
			if (w1,w2) not in bigram_to_freq:
				bigram_to_freq[(w1,w2)] = 1
			else:
				bigram_to_freq[(w1,w2)] += 1
	return bigram_to_freq

# given sentences, return bigram_to_freq dict 
def bigramCount(sentences):
	bigram_to_freq = {}
	for idx, sent in enumerate(sentences):
		sent  = sent.strip()											# remove '\n'
		Words = Text(sent).words										# tokenize the sentence
		# Words = sent.split()											# tokenize the sentence
		updateCount(Words,bigram_to_freq)								# call the fucntion for bigram updates
		if (idx+1)%100000 == 0	:										# after every 1M sentences
			bigram_to_freq = rmBigrams(bigram_to_freq)					# delete bigrams with count=1 (due to space constraint)
			print('Processed {} sentences, bigram_count: {}'.format(idx+1,len(bigram_to_freq)))
			# break
	bigram_to_freq = rmBigrams(bigram_to_freq)
	
	return bigram_to_freq

iter  = 0
while (iter < passes):
	if iter == 0:
		extn = ''
	else:
		extn = '_'+str(iter)
	word2freq_df   = pd.read_csv(files_dir +'word_to_freq.csv',header=None)	# unigrams, freq csv file
	word_to_freq   = dict(zip(list(word2freq_df[0]) ,list(word2freq_df[1])))				# dict - {word:count(word)}
	sentences      = open(files_dir+'sentences{}.txt'.format(extn),'r').readlines()			# txt file to write sentences
	
	print('Getting bigrams for starting iteration {} ...'.format(iter))
	bigram_to_freq = bigramCount(sentences)													# dict - (w,w2):freq
	print('Done processing sentences, bigram_count: {}'.format(len(bigram_to_freq)))
	
	print('Starting iteration {} with threshold {}...'.format(iter,threshold))
	doPhrases(sentences,word_to_freq,bigram_to_freq)
	print('End of iteration {} with threshold {}'.format(iter,threshold))
	iter = iter+ 1
	threshold = threshold - decrement



'''
for idx, bigram in enumerate(bigram_to_freq):
	w1, w2 = bigram[0],bigram[1] 
	if len(w1) and len(w2):
		if OVV(bigram,w1,w2):
			continue
		score = calScore(bigram,w1,w2)
		if score < threshold:
			continue
		print('For bigram:{}, w1:{}, w2:{}, score:{}'.format(bigram,w1,w2,score))
		phrase = w1 + '_' + w2 + '\n'
		phrases_doc.write(phrase)
		phrase_count = phrase_count + 1
	if (idx+1)%100000 == 0:
		print('Processed {} bigrams, phrase_count: {}'.format((idx+1),phrase_count)) 
'''