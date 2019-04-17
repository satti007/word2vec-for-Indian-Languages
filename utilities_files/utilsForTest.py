'''
* @file utils_test.py
* @author Satish Golla
* @date 22-03-2019
* Contains helper functions to run test.py
'''

import re
import argparse
import numpy as np
import pandas as pd
import commonFuctions
import syllabification

np.random.seed(1234)
re_tel = re.compile(u'[^\u0C00-\u0C7F]+') # regex for only telugu characters

##  function to parse the argumnets needed ny test.py
def parseArguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--minlen',type=int,default=1)
	ap.add_argument('--maxlen',type=int)
	
	ap.add_argument('--models_dir' ,type=str)
	ap.add_argument('--model_type' ,type=str,choices=['CNN_syl/','CNN_char/','Mean_syl/','CNN_hyb/'])
	ap.add_argument('--weights_dir',type=str)
	
	ap.add_argument('--language'     , type=str)
	ap.add_argument('--data_dir'     , type=str)
	ap.add_argument('--sim_file_path', type=str)
	
	print ('Parsing the Arguments')
	args = vars(ap.parse_args())
	
	minlen = args['minlen']
	maxlen = args['maxlen']
	models_dir  = args['models_dir']
	model_type  = args['model_type']
	weights_dir = args['weights_dir']
	data_dir      = args['data_dir']
	language      = args['language']
	sim_file_path = args['sim_file_path']
	
	print ('Arguments Parsing Done!')
	print ('Arguments details : ')
	print ('minlen, maxlen : ', minlen,maxlen)
	print ('models_dir     : ', models_dir)
	print ('model_type     : ', model_type)
	print ('weights_dir    : ', weights_dir)
	print ('language       : ', language)
	print ('data_dir       : ', data_dir)
	print ('sim_file_path  : ', sim_file_path)
	
	return minlen,maxlen,models_dir,model_type,weights_dir,language,data_dir,sim_file_path

def checkOOV(w,w_unit,unit2idx,maxlen,padUnit_idx,OOV,OOV_units):
	units     = []
	wordUnits = []
	for unit in w_unit:
		if unit in unit2idx:
			wordUnits.append(unit2idx[unit])
		else:
			OOV  = True
			units.append(unit)
			wordUnits.append(padUnit_idx)
	
	if OOV:
		print ('This word {} has OOV units'.format(w))
		print ('They are : ',units)
		OOV_units.extend(units)
	wordUnits.extend([padUnit_idx]*(maxlen-len(w_unit)))
	return wordUnits,OOV,list(set(OOV_units))

def checkValidity(w1_unit,w2_unit,minlen,maxlen):
	if len(w1_unit) >= minlen and len(w1_unit) <= maxlen:
		if len(w2_unit) >= minlen and len(w2_unit) <= maxlen:
			return True
		else:
			return False
	else:
		return False

def get_hybridUnits(word,unit2idx,lang,OOV,OOV_units):
	hyb   = []
	units = []
	syllables = syllabification.getSyllables(word,lang)
	for syl in syllables:
		if syl in unit2idx:
			hyb.append(unit2idx[syl])
		else:
			syl_ch = list(syl)
			for ch in syl_ch:
				if ch in unit2idx:
					hyb.append(unit2idx[ch])
				else:
					OOV = True
					units.append(ch)
					hyb.append(len(unit2idx))
	if OOV:
		print ('This word {} has OOV units'.format(w))
		print ('They are : ',units)
		OOV_units.extend(units)
	
	return hyb,OOV,list(set(OOV_units))

def get_wordUnits(minlen,maxlen,language,model_type,data_dir,sim_file_path):
	data         = open(sim_file_path).readlines()
	unit         = model_type.split('_')[-1].strip('/')
	unit2idx     = commonFuctions.load_pickleFile(data_dir + unit +'2idx.pkl')
	padUnit_idx  = len(unit2idx)
	OOV_units    = []
	pair2score   = {}
	word2Unitidx = {}

	for idx,l in enumerate(data):
		OOV1,OOV2 = False,False
		l       = l.strip().split(',')
		w1,w2,s = l[0].strip(),l[1].strip(),float(l[2].strip())
		
		if unit == 'syl':
			w1_unit = syllabification.getSyllables(w1,language[:2])
			w2_unit = syllabification.getSyllables(w2,language[:2])
		elif unit == 'char':
			w1,w2   = re_tel.sub(r'',w1),re_tel.sub(r'',w2)
			w1_unit = list(w1)
			w2_unit = list(w2)
		else:
			w1,w2   = re_tel.sub(r'',w1),re_tel.sub(r'',w2)
			w1_unit,OOV1,OOV_units = get_hybridUnits(w1,unit2idx,language[:2],OOV1,OOV_units)
			w2_unit,OOV2,OOV_units = get_hybridUnits(w2,unit2idx,language[:2],OOV2,OOV_units)
			if not OOV1 and not OOV2:
				if checkValidity(w1_unit,w2_unit,minlen,maxlen):
					w1_unit.extend([padUnit_idx]*(maxlen-len(w1_unit)))
					w2_unit.extend([padUnit_idx]*(maxlen-len(w2_unit)))
					word2Unitidx[w1]    = w1_unit
					word2Unitidx[w2]    = w2_unit
					pair2score[(w1,w2)] = s
				else:
					print ('Invalid pair: {}, {}'.format(w1,w2))
			else:
				print ('Invalid pair: {}, {}'.format(w1,w2))
		
		if unit != 'hyb':
			if checkValidity(w1_unit,w2_unit,minlen,maxlen):
				w1_unit,OOV1,OOV_units = checkOOV(w1,w1_unit,unit2idx,maxlen,padUnit_idx,OOV1,OOV_units)
				w2_unit,OOV2,OOV_units = checkOOV(w2,w2_unit,unit2idx,maxlen,padUnit_idx,OOV2,OOV_units)
				if not OOV1 and not OOV2:
					word2Unitidx[w1]    = w1_unit
					word2Unitidx[w2]    = w2_unit
					pair2score[(w1,w2)] = s
				else:
					print ('Invalid pair: {}, {}'.format(w1,w2))
			else:
				print ('Invalid pair: {}, {}'.format(w1,w2))
	
	if len(OOV_units) > 0:
		OOV_units = list(set(OOV_units))
		print ('')
		print ('Total number of OOV units: ',len(OOV_units))
		print ('They are : ',OOV_units)
	
	print ('')
	print ('Total number of valid pairs are: ',len(pair2score))
	
	return pair2score,word2Unitidx,len(unit2idx)+1


def remove_zeroScores(human_score,embd_score):
	zero_ids = []
	for idx,s in enumerate(embd_score):
		if s == 0:
			zero_ids.append(idx)
	
	print (zero_ids)
	embd_score1  = []
	human_score1 = []
	for i in range(len(human_score)):
		if i not in zero_ids:
			embd_score1.append(embd_score[i])
			human_score1.append(human_score[i])
	
	return human_score1,embd_score1

def writeTocsv(rho,rho_zero,save_dir,dim):
	df = pd.DataFrame()
	df['epoch'] = np.arange(1,len(rho_zero)+1).tolist() 
	df['rho']   = rho
	df['rho_zero'] = rho_zero
	df.to_csv(save_dir+'test_log_{}.csv'.format(dim),index=False)