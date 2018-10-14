'''
il: Indian Language
'''
import pandas as pd
from pyiwn import pyiwn
from nltk.corpus import wordnet as ewn

files_dir = '../../data/synset_maps/'

def get_il_synsets(lang):
	iwn  = pyiwn.IndoWordNet(lang)
	syns = iwn.all_synsets()
	il_syn_dict = {}
	for syn in syns: 
		syn_id = syn.synset_id()
		il_syn_dict[syn_id] = [lemma.name() for lemma in syn.lemmas()]
	il_syn_df = pd.DataFrame(list(il_syn_dict.items()),columns=['{}_id'.format(lang),'{}_lemmas'.format(lang[0:2])])
	
	return il_syn_dict,il_syn_df

def get_eng_synsets():
	en_syns = list(ewn.all_synsets())
	en_syn_dict = {}
	for esyn in en_syns: 
		esyn.offset()
		en_syn_dict[esyn.offset()]=esyn.lemma_names()
	en_syn_df=pd.DataFrame(list(en_syn_dict.items()),columns=['english_id','en_lemmas']) 
	
	return en_syn_dict,en_syn_df

def get_direct_synsets(src_lang,tgt_lang,merged_df,mapping):
	synset_pairs = []
	for r in merged_df.iterrows():
		src_l = r[1]['{}_lemmas'.format(src_lang[0:2])]
		tgt_l = r[1]['{}_lemmas'.format(tgt_lang[0:2])]
		for s in src_l: 
			for t in tgt_l: 
				synset_pairs.append('{}^{}'.format(s,t))
	synset_pairs_set = list(map(lambda p: p.split('^'), set(synset_pairs)))
	file_name = '{}_{}_{}.csv'.format(src_lang,tgt_lang,mapping)
	pd.DataFrame(synset_pairs_set,columns=[src_lang,tgt_lang]).to_csv(files_dir+file_name,index=False,encoding='utf-8')
	print ('No.of {}-{} {}-synset mappings are: {}'.format(src_lang,tgt_lang,mapping,len(synset_pairs_set)))
	print ('{} saved in {}'.format(file_name,files_dir))

def get_all_synsets(src_lang,tgt_lang,src_syn_df,tgt_syn_df,synset_mapping_fname,direct=None):
	synset_mapping = pd.read_csv(files_dir+synset_mapping_fname)
	merged_df = pd.merge(pd.merge(tgt_syn_df,synset_mapping),src_syn_df)
	if direct:
		get_direct_synsets(src_lang,tgt_lang,merged_df[merged_df.mapping_type=='Direct'],'direct')
	else:
		get_direct_synsets(src_lang,tgt_lang,merged_df,'all')


# src_lang = 'english'
# tgt_lang = 'telugu'

# if src_lang == 'english':
# 	print('Getting english synsets...')
# 	src_syn_dict,src_syn_df = get_eng_synsets()
# 	print ('No.of english synsets are: {}'.format(src_syn_df.shape[0]))
# else:
# 	print('Getting {} synsets...'.format(src_lang))
# 	src_syn_dict,src_syn_df = get_il_synsets(src_lang)
# 	print ('No.of {} synsets are: {}'.format(src_lang,src_syn_df.shape[0]))

# print('Getting {} synsets...'.format(tgt_lang))
# tgt_syn_dict,tgt_syn_df = get_il_synsets(tgt_lang)
# print ('No.of {} synsets are: {}'.format(tgt_lang,tgt_syn_df.shape[0]))

# synset_mapping_fname = 'english_telugu_id_mapping.csv'
# get_all_synsets(src_lang,tgt_lang,src_syn_df,tgt_syn_df,synset_mapping_fname,direct=None)
# get_all_synsets(src_lang,tgt_lang,src_syn_df,tgt_syn_df,synset_mapping_fname,True)