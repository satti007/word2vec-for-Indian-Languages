: '
* @file rundataPrep.sh
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Sat Mar 23 08:36:31 IST 2019
* @Contains command to run dataPrepForTraining.py
'

python3 -u dataPrepForTraining.py --data_file  ../../../data/corpora/small_data/stats/sentences.txt\
								--freq_file  ../../../data/corpora/small_data/stats/word2freq.csv\
								--save_dir   ../../../data/corpora/small_data/stats/\
								--minCount   5    --wordMaxlen  11\
								--doSampling True --threshold   1e-5\

python3 -u sylPrepForTraining.py  small_data train_words 1 4
python3 -u charPrepForTraining.py small_data train_words 4 10