python3 -u -W ignore supervised.py 			\
		--exp_id union --cuda False 		\
		--src_lang en  --tgt_lang te 		\
		--max_vocab 40000 --n_refinement 5 	\
		--tgt_emb ../../data/embds/cc.te.300.vec					\
		--src_emb ../../data/embds/wiki-news-300d-1M.vec 			\
		--dico_eval ../../data/synset_maps/GT/enU-tel.Test.txt 		\
		--dico_train ../../data/synset_maps/GT/enU-tel.Train.txt 	\
		--normalize_embeddings center