'''
* @file dataPreprocessing.py
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Mon Aug 13 14:02:04 IST 2018
* @Contains data cleaning and preprocessing functions
'''

import os
import sys
sys.path.insert(0, '../utilities_files/')

import re
import time
import string
import pandas as pd
import commonFuctions
from   polyglot.text import Text

start_time = time.time()

wiki_dir  = '../../../data/corpora/wiki/'					# base directory for wiki articles
WD_dir    = '../../../data/corpora/webdunia/'				# base directory for webdunia articles
AJ_dir    = '../../../data/corpora/andhrajyothy/'			# base directory for andhrajyothy articles
small_dir = '../../../data/corpora/small_data/'				# base directory for small data articles
sample_dir= '../../../data/corpora/sample_data/'			# base directory for sample data articles

files_dir =  sample_dir 									# AJ_dir (or) wiki_dir (or) WD_dir

folders   = sorted(os.listdir(files_dir+'articles/'))		# folders with articles
word_doc  = open(files_dir+'stats/vocab.txt','wb',0)		# txt file to write vocabulary
sent_doc  = open(files_dir+'stats/sentences.txt','wb',0)	# txt file to write sentences

punctuation = list(string.punctuation)						# list of special characters
re_tel      = re.compile(u'[^\u0C00-\u0C7F]+')				# regex for only telugu characters
re_num      = re.compile(u'[^\u0030-\u0039]+')				# regex for only numerical digits

# regex for {telugu, number}
# to consider only telugu words and numericals as valid words
re_word     = re.compile(u'[^\u0C00-\u0C7F\u0030-\u0039]+')

# regex for {telugu, number, puntucation-marks}
# mainly useful for  cleaning the wiki dump
re_valid    = re.compile(u'[^\u0C00-\u0C7F\u0020-\u0040\u005B-\u0060\u007B-\u007E\u000A]+')

# A mapping of digits to corresponding telugu word
digits_to_spell = {
	
	'0':' సున్నా',
	'1':' ఒకటి',
	'2':' రెండు',
	'3':' మూడు',
	'4':' నాలుగు',
	'5':' ఐదు',
	'6':' ఆరు',
	'7':' ఏడు',
	'8':' ఎనిమిది',
	'9':' తొమ్మిది'
}

# save dicts (freq_count,line2hash -- useful if you are preprocessing the data in chunks)
def save_dicts(word2freq,line2hash):
	word2freq   = dict(sorted(word2freq.items()  ,key=lambda x:x[1],reverse=True))
	pd.DataFrame.from_dict(data=word2freq,orient='index').to_csv(files_dir+'stats/word2freq.csv',header=None)
	pd.DataFrame.from_dict(data=line2hash,orient='index').to_csv(files_dir+'stats/line2hash.csv',header=None)

# update digits(it's tel word) count
# Example: nums = ['ఒకటి' రెండు మూడు','రెండు మూడు']
def update_digits_count(nums,word2freq):
	for num in nums:
		for d in num.split(' '):
			if d not in word2freq:
				word_doc.write((d+'\n').encode('utf-8'))
				word2freq[d]  = 1
			else:
				word2freq[d] += 1
	
	return word2freq

# update words count
def update_vocab_count(nums,words,sent,word2freq,line2hash,uni_sen):
	h_sent = re_tel.sub(r'',sent)							# put only telugu chars in the sentence and ...
	if hash(h_sent) not in line2hash:					# ... check if sentence is unique
		line2hash[hash(h_sent)] = uni_sen
		sent_doc.write((sent+'\n').encode('utf-8'))
		word2freq = update_digits_count(nums,word2freq)
		for w in words:
			if w not in word2freq:
				word_doc.write((w+'\n').encode('utf-8'))
				word2freq[w]  = 1
			else:
				word2freq[w] += 1
		uni_sen += 1
	
	return word2freq,line2hash,uni_sen

# spell the number(if present) in the word, and seprate it
# Ex1: w = '1310లుగా', return 'ఒకటి  మూడు ఒకటి  సున్నా', 'లుగా'
# Ex2: w = '1310'      , return 'ఒకటి  మూడు ఒకటి  సున్నా', ''
# Ex3: w = 'లుగా'    , return '', 'లుగా'
def spellDigits(w):
	W = w.replace('"','').replace(',','').replace('.','') 	# remove '"' ',' '.'
	n = re_num.sub(r'',W)									# extract num from the word
	W = W.replace(n,'')										# remove  num from the word(to get chars only)
	if len(n):
		for d in digits_to_spell:
			n = n.replace(d,digits_to_spell[d]) 			# replace each digit by it's tel word
		n = n.strip()
		return n,W
	return n,w

tot_sen    = 0				# total number of sentences
uni_sen    = 0				# unique number of sentences
file_count = 0				# no.of articles preprocessed
time_count = 0				# count for displaying time
word2freq  = {}				# dict to store word frequency
line2hash  = {}				# dict to store has value of a sentence to check if it is unique

for F in folders:				# loop through all article folders
	files = sorted(os.listdir(files_dir+'articles/'+F))
	for f in files:				# loop through all articles
		text = open(files_dir+'articles/'+F+'/'+f,'r').read()	# load the content of the article
		text = re_valid.sub(r'',text)							# keep only {telugu chars, number, puntucation-marks}
		text = Text(text)										# get a text object from polyglot
		try:
			sentences = text.sentences							# try to do sentence segmentation...
		except:													# ... if not possible (maybe due to illegal chars in the text)...
			continue 											# ... ignore it
		for i,s in enumerate(sentences):						
			tot_sen += 1
			sent  = ' '
			nums  = []											# list of telugu-digits in the article
			words = []											# list of   valid-words in the article
			for w in s.words:									# tokenize each sentence
				w = w.strip()									# strip of spaces (at first & last)
				w = re_word.sub(r'',w)							# keep only telugu chars & numbers
				if len(w) ==0:									# if len=0, after striping spaces, ignore the word
					continue
				n,w = spellDigits(w)							# spell the number(if present) in the word, and seprate it
				if len(n):										# if the word has numnber ...
					nums.append(n)								# ... append them to this list
					words.append(n)								# append the num to this list for forming sentence
				if len(w):
					words.append(w)
			sent = sent.join(words).strip() 					# join the valid words by space to form the sentence
			if len(sent):
				words = list(set(words)-set(nums))
				word2freq,line2hash,uni_sen = update_vocab_count(nums,words,sent,word2freq,line2hash,uni_sen)
		file_count += 1
		time_count += 1
		if file_count == 10000:
			print ('Time taken to process 10K files is : {}'.format(time.time() - start_time))
			start_time = time.time()
			time_count = 0
		print ('{}/{} done, file_count:{}, size of vocab: {}, tot_sen: {}, uni_sen: {}'
							.format(F,f,file_count,len(word2freq),tot_sen,uni_sen))
		if file_count%100 == 0:
			save_dicts(word2freq,line2hash)
		# if file_count%100 == 0:
		# 	break

print ('file_count:{}, size of vocab: {}, tot_sen: {}, uni_sen: {}'.format(file_count,len(word2freq),tot_sen,uni_sen))
save_dicts(word2freq,line2hash)
commonFuctions.get_stats(word2freq,tot_sen,uni_sen,files_dir,punctuation)