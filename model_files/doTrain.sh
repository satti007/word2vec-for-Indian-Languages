python3 -W ignore -u train.py --ws 5    --neg 10\
						 	  --lr 0.005 --dim 300\
							  --lang te --epochs 300\
							  --doSampling True --save_dir weights\
							  --pretrain False  --state 0 --alpha 0.001\
							  --idx2syl 		../../../data/corpora/webdunia/stats/idx2syl.npy\
							  --idx2word 		../../../data/corpora/webdunia/stats/idx2word.npy\
							  --train_words   	../../../data/corpora/webdunia/stats/train_words.npy\
							  --word2Sylidx_3   ../../../data/corpora/webdunia/stats/word2Sylidx_3.npy\
							  --word2Sylidx_7   ../../../data/corpora/webdunia/stats/word2Sylidx_7.npy\
							  --word2Sylidx_11  ../../../data/corpora/webdunia/stats/word2Sylidx_11.npy\
							  --word2Sylidx_3to7 ../../../data/corpora/webdunia/stats/word2Sylidx_3to7.npy\
							  --train_words_3to7 ../../../data/corpora/webdunia/stats/train_words_3to7.npy\
							  --sampled_words 	 ../../../data/corpora/webdunia/stats/sampledTrain_words.npy\


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