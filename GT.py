import os
import glob
import numpy as np
import pandas as pd

files_dir = '../../data/synset_maps/fastext/'
save_dir1 = '../../data/synset_maps/GT/'
save_dir  = '../../data/synset_maps/GT/helper_files'

def save_list(file_name,list_W):
	with open(save_dir + file_name, 'w') as f:
		for word in list_W:
			f.write("{}\n".format(word))

def unionOflists(file_name,engWords_union_set):
	en_to_lang         = pd.read_csv(files_dir+file_name,sep=' ',header=None)
	engWords_in_lang   = list(set(en_to_lang[0].tolist()))
	engWords_union_set = list(set(engWords_union_set+engWords_in_lang))
	
	return engWords_union_set

def get_otherLang_words(file_name):
	en_to_lang      = pd.read_csv(files_dir+file_name,sep=' ',header=None)
	for i in range(0,2):
		en = ''
		if i==0:
			en = '_en' 
		otherLang_words = en_to_lang[i].tolist()
		words_file      = '{}_{}{}.txt'.format(file_name.split('.')[0].split('-')[1],file_name.split('.')[1],en)
		save_list(words_file,otherLang_words)

en_to_lang_files     = os.listdir(files_dir)  
engWords_union_test  = []
engWords_union_train = []
for f in en_to_lang_files:
	if '.Train.txt' in f:
		engWords_union_train = unionOflists(f,engWords_union_train)
	if '.Test.txt' in f:
		engWords_union_test  = unionOflists(f,engWords_union_test)
	get_otherLang_words(f)

engWords_in_test = list(set(engWords_union_train) -set(engWords_union_test))
engWords_union_test.sort()
engWords_union_train.sort()

save_list('TestengWords_hi_union_ta.txt',engWords_union_test)
save_list('TrainengWords_hi_union_ta.txt',engWords_union_train)

def clean_file(file_name):
	lines = open(file_name).readlines()
	file  = open(file_name,'w')
	for line in lines:
		if '	' in line:
			line = line.replace('	',' ')
			if len(line.split()) == 2:
				file.write(line)
	file.close()

def upperToLower(file_name):
	data = open(file_name,'r').read()
	data = data.lower()
	file = open(file_name,'w')
	file.write(data)
	file.close()
	clean_file(file_name)

files = glob.glob(save_dir1+'/*.txt')
for f in files:
	upperToLower(f)
