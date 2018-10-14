python3 -W ignore -u data_prep.py --data_file  stats/sentences.txt\
								  --freq_file  stats/word_to_freq.csv\
								  --minCount   5    --wordMaxlen  14\
								  --doSampling True --threshold   1e-5\
								  --lang  te

# freq_file  = 'stats/word_to_freq.csv'
# data_file  = 'stats/sentences.txt'
# lang       = 'te'
# minCount   = 5
# wordMaxlen = 14
# doSampling = True
# threshold  = 1e-5
