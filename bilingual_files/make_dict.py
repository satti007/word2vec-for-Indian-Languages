import os
import numpy as np
import pandas as pd
from mapping import get_il_synsets

files_dir = '../../data/synset_maps/'

def clean_file(file_name):
	lines = open(files_dir1+file_name).readlines()
	file  = open(files_dir1+file_name,'w')
	for line in lines:
		if '	' in line:
			line = line.replace('	',' ')
		file.write(line)
	file.close()

def get_syn_idx(word,lang_syn_dict):
	syn_idxs = []
	for idx in lang_syn_dict:
		if word in lang_syn_dict[idx]:
			syn_idxs.append(idx)
	return syn_idxs

tel_syn_dict,temp  = get_il_synsets('telugu')
hin_syn_dict,temp  = get_il_synsets('hindi')
enTohin_files = ['en-hi.Train.txt','en-hi.Test.txt']

for file in enTohin_files:
	en_to_hin     = pd.read_csv(files_dir+file,header=None,sep=' ')
	engTotel_file = open(files_dir+'en-tel.{}.csv'.format(file.split('.')[1]),'wb',0)
	for index, row in en_to_hin.iterrows():
		eng_w,hin_w = row[0],row[1]
		syn_idxs    = get_syn_idx(hin_w,hin_syn_dict)
		if len(syn_idxs) > 0:
			for idx in syn_idxs:
				try:
					tel_syns = tel_syn_dict[idx]
				except:
					continue 
				for syn in tel_syns:
					engTotel_file.write((eng_w+'	'+syn+'\n').encode('utf-8'))
	engTotel_file.close()

# def unionOflists(file_name,engWords_union_train):
# 	clean_file(file_name)
# 	en_to_lang           = pd.read_csv(files_dir1+file_name,sep=' ',header=None)
# 	engWords_in_lang     = list(set(en_to_lang[0].tolist()))
# 	engWords_union_train = list(set(engWords_union_train+engWords_in_lang))
	
# 	return engWords_union_train


# en_to_lang_files     = os.listdir(files_dir1)  
# engWords_union_test  = []
# engWords_union_train = []
# for f in en_to_lang_files:
# 	if 'en-' in f: 
# 		if '.0-5000.txt' in f:
# 			engWords_union_train = unionOflists(f,engWords_union_train)
# 		if '.5000-6500.txt' in f:
# 			engWords_union_test  = unionOflists(f,engWords_union_test)

# engWords_union_test.remove(engWords_union_test[0])
# engWords_union_test.sort()
# engWords_union_train.sort()

# en_to_tel         = pd.read_csv(files_dir2+'english_telugu_direct.csv')
# engWords_in_dict  = list(set(en_to_tel['english'].tolist()))
# engWords_in_both  = list(set(engWords_in_dict).intersection(engWords_union_train))
# engWords_in_both1 = list(set(engWords_in_dict).intersection(engWords_union_test))
# engWords_in_both1 = list(set(engWords_in_both1) -set(engWords_in_both))
# engWords_in_both.sort()
# engWords_in_both1.sort()
# print ('No.of en words in train_file: {}'.format(len(engWords_in_both)))
# print (' No.of en words in test_file: {}'.format(len(engWords_in_both1)))

# train_file = 'en-tel_trainDict.csv'
# test_file  = 'en-tel_testDict.csv'
# train_dict =  pd.merge(pd.DataFrame(engWords_in_both,columns=['english']),en_to_tel)
# test_dict  =  pd.merge(pd.DataFrame(engWords_in_both1,columns=['english']),en_to_tel)
# train_dict.to_csv(files_dir2+train_file,index=False,header=False,encoding='utf-8',sep=' ')
# test_dict.to_csv(files_dir2+test_file,index=False,header=False,encoding='utf-8',sep=' ')
# print ('No.of en-tel pairs in {}: {}'.format(train_file,train_dict.shape[0]))
# print ('No.of en-tel pairs in {}: {}'.format(test_file,test_dict.shape[0]))
# print ('Files {},{} saved to: {}'.format(train_file,test_file,files_dir2))