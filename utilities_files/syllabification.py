'''
* @file syllabification.py
* @author Satish Golla
* @date 22-03-2019
* Contains function to syllabify words
'''

import sys

lib_dir             = '../../../src/'						# path to directory conataining library and resources
INDIC_NLP_LIB_HOME  = lib_dir + 'indic_nlp_library'			# path to the library source (needed for syllabification)
INDIC_NLP_RESOURCES = lib_dir + 'indic_nlp_resources'		# path to the resources needed by the above library

## loading the library for syllabification
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))		
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.syllable import  syllabifier

## function to get syllables of a word in an language
def getSyllables(word,lang):
	return syllabifier.orthographic_syllabify(word,lang)