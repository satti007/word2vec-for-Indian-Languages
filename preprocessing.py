# -*- coding: utf-8 -*-

import os
import re
import string
import pandas as pd
from polyglot.text import Text

wiki_dir = '../../data/corpora/wiki/'						# base directory for wiki articles
AJ_dir   = '../../data/corpora/andhrajyothy/'				# base directory for andhrajyothy articles

files_dir =  AJ_dir 										# AJ_dir (or) wiki_dir

folders   = sorted(os.listdir(files_dir+'articles/'))		# folders with articles
word_doc  = open(files_dir+'stats/vocab.txt','wb',0)		# txt file to write vocabulary
sent_doc  = open(files_dir+'stats/sentences.txt','wb',0)	# txt file to write sentences

# regex for {telugu, number, puntucation-marks} 'utf-8' encoding
re_valid    = re.compile(u'[^\u0C00-\u0C7F\u0020-\u0040\u005B-\u0060\u007B-\u007E\u000A]+')
re_tel      = re.compile(u'[^\u0C00-\u0C7F]+')				# regex for only telugu 'utf-8' encoding
punctuation = list(string.punctuation)						# list of special characters

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

# spelling digits in a number
def spellDigits(w):
	W = w.replace('"','').replace(',','').replace('.','') 	# remove '"' ',' '.'  
	try:
		x = float(W)				
		for d in digits_to_spell:
			W = W.replace(d,digits_to_spell[d]) 			# replace each digit by it's tel word
		W = W.strip()
		return W,1
	except ValueError:										# if the word not a int/float
		return w,0

# update digits(it's tel word) count
# Example: num = [['ఒకటి' రెండు మూడు'],['రెండు మూడు']]
def update_digits_count(nums,word_to_freq):
	for num in nums:
		for d in num.split(' '):
			if d not in word_to_freq:
				word_doc.write((d+'\n').encode('utf-8'))
				word_to_freq[d]  = 1
			else:
				word_to_freq[d] += 1
	
	return word_to_freq

# update words count
def update_vocab_count(nums,words,sent,word_to_freq,hash_to_line,uni_sen):
	h_sent = re_tel.sub(r' ',sent).strip()					# put only telugu chars in the sentence and ...
	if hash(h_sent) not in hash_to_line:					# ... check if sentence is unique
		uni_sen += 1
		sent_doc.write((sent+'\n').encode('utf-8'))
		word_to_freq = update_digits_count(nums,word_to_freq)
		hash_to_line[hash(h_sent)] = uni_sen
		for w in words:
			if w not in word_to_freq:
				word_doc.write((w+'\n').encode('utf-8'))
				word_to_freq[w]  = 1
			else:
				word_to_freq[w] += 1
	
	return word_to_freq,hash_to_line,uni_sen

# bigram_count
def get_bigran_count(words,bigram_to_freq):
	for i in range(0,len(words)-1):
		w1,w2 = words[i],words[i+1]
		if len(re_tel.sub(r'',w1)) and len(re_tel.sub(r'',w2)):
			if (w1,w2) not in bigram_to_freq:
				bigram_to_freq[(w1,w2)] = 1
			else:
				bigram_to_freq[(w1,w2)] += 1
	
	return bigram_to_freq

# delete bigrams with count=1 
def clrBigrams_1(bigram_to_freq):
	return {k:v for k,v in bigram_to_freq.items() if v>1}

# save dicts (freq_count,bigram_count,hash_to_line -- useful if you are preprocessing the data in chunks)
def save_dicts(word_to_freq,bigram_to_freq,hash_to_line):
	word_to_freq   = dict(sorted(word_to_freq.items()  ,key=lambda x:x[1],reverse=True))
	bigram_to_freq = dict(sorted(bigram_to_freq.items(),key=lambda x:x[1],reverse=True))
	pd.DataFrame.from_dict(data=word_to_freq,orient='index').to_csv(files_dir+'stats/word_to_freq.csv',header=None)
	pd.DataFrame.from_dict(data=hash_to_line,orient='index').to_csv(files_dir+'stats/hash_to_line.csv',header=None)
	pd.DataFrame.from_dict(data=bigram_to_freq,orient='index').to_csv(files_dir+'stats/bigram_to_freq.csv',header=None)

# write stats to a file
def save_stats(stats):
	file = open(files_dir+'stats/stats.txt','w')
	file.write('Number of sentences:           {}M\n'.format(round(stats[0]/pow(10.0,6),2)))
	file.write('Number of unique sentences:    {}M\n'.format(round(stats[1]/pow(10.0,6),2)))
	file.write('Number of words     in vocabulary: {}M\n'.format(round(stats[2]/pow(10.0,6),2)))
	file.write('Number of sp.chars  in vocabulary: {}\n'.format(len(stats[4])))
	file.write('Number of bigrams   in corpora   : {}M\n'.format(round(stats[3]/pow(10.0,6),2)))
	file.write('Number of tokens    in corpora:   {}M\n'.format(round(stats[5]/pow(10.0,6),2)))
	file.write('Number of sp.tokens in corpora:   {}M\n'.format(round(stats[6]/pow(10.0,6),2)))
	file.write('Number of words with freq>=5 in vocabulary:  {}K\n'.format(round(stats[7]/pow(10.0,3),2)))
	file.write('Number of words with freq>=10 in vocabulary: {}K\n'.format(round(stats[8]/pow(10.0,3),2)))
	file.write('Number of words with freq>=20 in vocabulary: {}K\n'.format(round(stats[9]/pow(10.0,3),2)))
	file.write('\n\nsp.chars = sp.tokens:\n{}'.format(stats[4]))
	file.close()

# caluclate the stats
def get_stats(word_to_freq,bigram_to_freq,tot_sen,uni_sen):
	word_to_freq = dict(sorted(word_to_freq.items(),key=lambda x:x[1],reverse=True))
	tot_vocab    = len(word_to_freq)
	bigram_count = len(bigram_to_freq)
	pun_vocab    = [w for w in punctuation if w in word_to_freq]
	tot_tokens   = sum(word_to_freq.values())
	pun_tokens   = sum(word_to_freq[w] for w in punctuation if w in word_to_freq)
	vocab_5      = sum(1 for f in word_to_freq.values() if f >= 5)
	vocab_10     = sum(1 for f in word_to_freq.values() if f >= 10)
	vocab_20     = sum(1 for f in word_to_freq.values() if f >= 20)
	
	save_stats([tot_sen,uni_sen,tot_vocab,bigram_count,pun_vocab,tot_tokens,pun_tokens,vocab_5,vocab_10,vocab_20])

tot_sen 	 = 0
uni_sen 	 = 0
file_count   = 0
word_to_freq   = {}
hash_to_line   = {}
bigram_to_freq = {}

for F in folders:
	files = sorted(os.listdir(files_dir+'articles/'+F))
	for f in files:
		text = open(files_dir+'articles/'+F+'/'+f,'r').read()
		text = re_valid.sub(r'',text)							# keep only {telugu chars, number, puntucation-marks}
		text = Text(text)
		try:
			sentences = text.sentences
		except:
			continue 
		for i,s in enumerate(sentences):
			tot_sen += 1
			sent  = ' '
			nums  = []
			words = []
			for w in s.words:
				w   = w.strip()
				w,c = spellDigits(w)
				if c == 1:
					nums.append(w)
					words.append(w)
				else:
					if len(w) > 0:
						words.append(w)
			sent = sent.join(words)
			if len(sent.strip()) > 0:
				words = list(set(words)-set(nums))
				txt_words = Text(sent).words
				bigram_to_freq = get_bigran_count(txt_words,bigram_to_freq)
				word_to_freq,hash_to_line,uni_sen = update_vocab_count(nums,words,sent,word_to_freq,hash_to_line,uni_sen)
		file_count += 1
		print ('{}/{} done, file_count:{}, size of vocab: {}, size of bigrams: {}, tot_sen: {}, uni_sen: {}'
							.format(F,f,file_count,len(word_to_freq),len(bigram_to_freq),tot_sen,uni_sen))
		if file_count%10000 == 0:
			save_dicts(word_to_freq,bigram_to_freq,hash_to_line)
		if file_count%100000 == 0:
			bigram_to_freq = clrBigrams_1(bigram_to_freq)

bigram_to_freq = clrBigrams_1(bigram_to_freq)
save_dicts(word_to_freq,bigram_to_freq,hash_to_line)
print ('file_count:{}, size of vocab: {}, size of bigrams: {}, tot_sen: {}, uni_sen: {}'
				.format(file_count,len(word_to_freq),len(bigram_to_freq),tot_sen,uni_sen))
get_stats(word_to_freq,bigram_to_freq,tot_sen,uni_sen)