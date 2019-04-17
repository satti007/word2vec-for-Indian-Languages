: '
* @file runTrain.sh
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Tue Apr  2 23:04:29 IST 2019
* @Contains command to run train.py 
'

python3 -u train.py 	--unit syl 			--model CNN\
						--lr 1e-4 			--ws 5\
						--neg 10 			--epochs 20\
						--maxlen 4 			--num_layers 4\
						--l2reg False 		--beta 1e-3\
						--dropOut True      --prob 0.5\
						--pretrain False 	--state  0\
						--embd_dim 300 		--rep_dim 200\
						--batch_size 20000 	--save_dir weights/\
						--idx2unit ../../../data/corpora/all_combined/stats/idx2hyb.npy\
						--idx2word ../../../data/corpora/all_combined/stats/idx2word.npy\
						--word2Unitidx ../../../data/corpora/all_combined/stats/word2Sylidx_1to4.npy\
						--train_words ../../../data/corpora/all_combined/stats/train_words_1to4.npy\