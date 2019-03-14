python3  -u train.py 	--lr 0.0005 --dim 300\
						--max_syl 6 --rep_dim 300\
						--ws      5 --neg 10\
						--epochs 20 		--batch_size 1000\
						--pretrain false 	--state 0\
						--save_dir		weights\
						--idx2syl		../../../data/corpora/andhrajyothy/stats/idx2syl.npy\
						--idx2word		../../../data/corpora/andhrajyothy/stats/idx2word.npy\
						--train_words	../../../data/corpora/andhrajyothy/stats/train_words_2to6.npy\
						--word2Sylidx	../../../data/corpora/andhrajyothy/stats/word2Sylidx_2to6.npy\

# lr  = 0.01
# ws  = 5
# neg = 10
# dim = 300
# lang       = 'te'
# state      = 0
# epochs     = 10
# pretrain   = False
# doSampling = True
# save_dir   = 'weights'
# idx2syl     = '../../../data/corpora/webdunia/stats/idx2syl.npy'
# idx2word    = '../../../data/corpora/webdunia/stats/idx2word.npy'
# train_words      = '../../../data/corpora/webdunia/stats/train_words.npy'
# word2Sylidx_3    = '../../../data/corpora/webdunia/stats/word2Sylidx_3.npy'
# word2Sylidx_7    = '../../../data/corpora/webdunia/stats/word2Sylidx_7.npy'
# word2Sylidx_11   = '../../../data/corpora/webdunia/stats/word2Sylidx_11.npy'
# word2Sylidx_3to7 = '../../../data/corpora/webdunia/stats/word2Sylidx_3to7.npy'
# train_words_3to7 = '../../../data/corpora/webdunia/stats/train_words_3to7.npy'
# sampled_words    = '../../../data/corpora/webdunia/stats/sampledTrain_words.npy'