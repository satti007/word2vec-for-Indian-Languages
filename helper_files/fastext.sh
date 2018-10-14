./fasttext skipgram -input ../stats/sentences.txt\
					 -output ../embds\
					 -verbose 2 -minCount 5 -wordNgrams 14\
					 -bucket 200000 -minn 3 -maxn 6 -t 1e-5\
					 -lr 0.05 -dim 300 -ws 5 -epoch 10 -neg 10 -loss ns -thread 8