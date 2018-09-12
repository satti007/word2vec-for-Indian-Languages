import io
import math
import numpy as np

# Function to load word embeddings saved in a txt file
# The format of embeddings file should be as follows,
# "words" dimension               * FIRST LINE (has the dimension of the embeddings)
# word1 x1 x2 ......... xd        * SECOND LINE(word1 follwoed by space separated vector)
# word2 y1 y2 ......... yd
def read_txt_embeddings(emb_path,full_vocab=False,max_vocab=200000,emb_dim=300):
    word2id = {}                  # dictionary to store the index of word 'W' as word2id['W'] = id
    vectors = []                  # list of vectors such that embedding of word 'W' is vectors[word2id['W']]
    
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:                                      # ignore the first line
                split = line.split()
                assert len(split) == 2
                assert emb_dim == int(split[1])             
            else:
                word, vect = line.rstrip().split(' ', 1)    # split the line into word and vector
                vect = np.fromstring(vect, sep=' ')         # form a vector from space separated string
                if np.linalg.norm(vect) == 0:               # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:                           # just throws a warning
                        print("Word '%s' found twice in %s embedding file" % (word, emb_path))
                else:
                    if not vect.shape == (emb_dim,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i" % (vect.shape[0], emb_path, word, i))
                        continue
                    assert vect.shape == (emb_dim,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
                break
    
    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))
    
    id2word    = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)
    
    return embeddings,word2id,id2word


# return vector for a given word
def get_wordEmbedding(word,word2id,embeddings,OOV_count,OOV_words):
    try:
        vec = embeddings[word2id[word]]                    # get the embedding if it is in vocabulary
    except:
        vec = np.zeros([300])                              # return zero vector if OOV
        OOV_words.append(word.encode('utf-8'))
        OOV_count = OOV_count + 1
        print ('OOV word: {}'.format(word.encode('utf-8')))
    
    return vec,OOV_count,OOV_words

# caluclate cosine similarity
def get_wordSim(vec1,vec2):
    return round(10*(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))),3)

# sum of euclidean distance between corresponding elements in two vectors
# if vec1 = [x1,x2,x3] & vec2 = [y1,y2,y3] return sqrt((x1-y1)^2+(x2-y2)^2+(x3-y3)^2)
def euclidean(vect_list1,vect_list2):
    return sum([(a - b)**2 for a, b in zip(vect_list1,vect_list2)])

# caluclate spearman's correlation btw human_score and cosine similarity
def get_rho(human_score,emb_score):
    d = euclidean(human_score,emb_score)
    sp = (6*d)/19656.0
    
    return round(100*(1 - sp),3)

emb_path = '../../data/embds/WordVec/cc.te.300.vec'       # path to embeddings (text)file
embeddings,word2id,id2word = read_txt_embeddings(emb_path,False,200000)

sim_file = '../../data/WordSim/Telugu-WS.txt'            # path to WordSim dataset
lines    = open(sim_file).readlines()
word1    = []
word2    = []
emb_score   = []
human_score = []
OOV_count = 0
OOV_words = []
for l in lines:
    l       = l.strip().decode('utf-8')
    w1,w2,s = l.split(',')[0],l.split(',')[1],float(l.split(',')[2]) # split the line to get pair of words and their human_score
    v1,OOV_count,OOV_words = get_wordEmbedding(w1,word2id,embeddings,OOV_count,OOV_words) # get vector for word 'W1'
    v2,OOV_count,OOV_words = get_wordEmbedding(w2,word2id,embeddings,OOV_count,OOV_words) # get vector for word 'W2'
    e_sim        = get_wordSim(v1,v2)                   # cosine similarity btw v1 & v2
    word1.append(w1)
    word2.append(w2)
    human_score.append(s)
    emb_score.append(e_sim)
    print w1.encode('utf-8')+' '+w2.encode('utf-8')+' '+str(s)+' '+str(e_sim)

print '\nTotal OOV words: ',OOV_count
print 'They are: '
for w in OOV_words:
    print w + ' ',
print ''

emb_score = [0 if math.isnan(x) else x for x in emb_score]
print 'rho is with OVV words: ',get_rho(human_score,emb_score)

OVV_idxs     = [i for i, x in enumerate(emb_score) if x == 0]
emb_score1   = [i for i in emb_score if i!=0]
human_score1 = [s for idx,s in enumerate(human_score) if idx not in OVV_idxs ]
print 'rho is with out OVV words: ',get_rho(human_score1,emb_score1)