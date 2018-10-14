python3 -W ignore -u train.py --word2idx stats/word2idx.npy --idx2word stats/idx2word.npy\
							  --syl2idx  stats/syl2idx.npy  --idx2syl stats/idx2syl.npy\
							  --doSampling True --save_dir weights\
							  --word2Sylidx   stats/word2Sylidx.npy\
							  --train_words   stats/train_words.npy\
							  --sampled_words stats/sampledTrain_words.npy\
							  --pretrain False --state 0\
							  --lang te --epochs 10\
						 	  --lr 0.01 --dim 300\
							  --ws 5    --neg 10\