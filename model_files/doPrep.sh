python3 -W ignore -u data_prep.py --data_file  ../../../data/corpora/all_combined/stats/sentences.txt\
								  --freq_file  ../../../data/corpora/all_combined/stats/word_to_freq.csv\
								  --save_dir   ../../../data/corpora/all_combined/stats/\
								  --minCount   5    --wordMaxlen  14\
								  --doSampling True --threshold   1e-5\
								  --lang  te

freq_file  = '../../data/stats/word_to_freq.csv'
data_file  = '../../data/stats/sentences.txt'
save_dir   = '../../data/stats/'
lang       = 'te'
minCount   = 5
wordMaxlen = 14
doSampling = True
threshold  = 1e-5
