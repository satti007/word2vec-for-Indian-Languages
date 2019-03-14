import numpy as np

# function to save the data as file_name.npy
def saveTofile(save_dir,file_name,data):
	np.save(save_dir+'{}.npy'.format(file_name),data)
	print ('Saved {}.npy'.format(file_name))

# function to load the data from file_name.npy
def loadFile(file_name,isDict):
	print ('Loaded {}'.format(file_name))
	if isDict:
		return np.load(file_name).item()
	
	return np.load(file_name)

def padSyllables_toMax(word2Sylidx,syl2idx,Min,Max):
	pad_syl      = len(syl2idx)
	syl_MintoMax = []
	word2Sylidx_MintoMax = {}
	for word in word2Sylidx:
		if len(word2Sylidx[word]) >= Min and len(word2Sylidx[word]) <= Max:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(Max-len(word2Sylidx[word])))
			syl_MintoMax.extend(syllables)
			syl_MintoMax = list(set(syl_MintoMax))
			word2Sylidx_MintoMax[word] = syllables
	
	syl_MintoMax.append(pad_syl)
	return word2Sylidx_MintoMax,syl_MintoMax

data_dir    = '../../../data/corpora/andhrajyothy/stats/'
Min, Max    = 2,6
syl2idx     = loadFile(data_dir+'syl2idx.npy',1)
word2Sylidx = loadFile(data_dir+'word2Sylidx.npy',1)
train_words = loadFile(data_dir+'train_words.npy',0)
word2Sylidx_MintoMax,syl_MintoMax = padSyllables_toMax(word2Sylidx,syl2idx,Min,Max)
train_words_MintoMax = [w for w in train_words if w in word2Sylidx_MintoMax]
print ('The number of syllables in words  with num of syll {}to{} are : {}'.format(Min,Max,len(syl_MintoMax)))
print ('The number of words in vocabulary with num of syll {}to{} are : {}K'.format(Min,Max,len(word2Sylidx_MintoMax)//pow(10,3)))
print ('The number of tokens in corpora   with num of syll {}to{} are : {}M'.format(Min,Max,len(train_words_MintoMax)//pow(10,6)))
saveTofile(data_dir,'word2Sylidx_{}to{}'.format(Min,Max),word2Sylidx_MintoMax)
saveTofile(data_dir,'train_words_{}to{}'.format(Min,Max),np.asarray(train_words_MintoMax))


'''
# function to pad the syllables to make a batch of words have same no.of.syllables
# words with 1,2,3     num of syllables, pad them so that all of them become 3 syllabi words
# words with 4,5,6,7   num of syllables, pad them so that all of them become 7 syllabi words
# words with 8,9,10,11 num of syllables, pad them so that all of them become 11 syllabi words
def padSyllables_3c7c11(word2Sylidx,syl2idx):
	pad_syl = len(syl2idx)
	word2Sylidx_3  = {}
	word2Sylidx_7  = {}
	word2Sylidx_11 = {}
	for word in word2Sylidx:
		if len(word2Sylidx[word]) <= 3:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(3-len(word2Sylidx[word])))
			word2Sylidx_3[word] = syllables
		elif len(word2Sylidx[word]) <= 7:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(7-len(word2Sylidx[word])))
			word2Sylidx_7[word] = syllables
		else:
			syllables = np.copy(word2Sylidx[word]).tolist()
			syllables.extend([pad_syl]*(11-len(word2Sylidx[word])))
			word2Sylidx_11[word] = syllables
		
	return word2Sylidx_3,word2Sylidx_7,word2Sylidx_11

word2Sylidx_3,word2Sylidx_7,word2Sylidx_11 = padSyllables_3c7c11(word2Sylidx,syl2idx)
saveTofile(save_dir,'word2Sylidx_3' ,word2Sylidx_3)
saveTofile(save_dir,'word2Sylidx_7' ,word2Sylidx_7)
saveTofile(save_dir,'word2Sylidx_11',word2Sylidx_11)
'''