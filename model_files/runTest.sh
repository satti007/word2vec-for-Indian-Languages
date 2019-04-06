: '
* @file runTest.sh
* @author Satish Golla <gsatishkumaryadav@gmail.com>
* @date Fri Mar 22 20:36:31 IST 2019
* @Contains command to run test.py
'

python3 -u -W ignore test.py \
			--minlen 1 --maxlen 4\
			--models_dir models/SD/ --model_type CNN_syl/ \
			--weights_dir weights_200/ --language telugu \
			--sim_file_path ../../../data/WordSim/Telugu-WS.txt\
			--data_dir   ../../../data/corpora/andhrajyothy/stats/