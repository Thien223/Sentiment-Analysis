#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_arguments():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--text', type=str, default='', help='input text to find similar words')
	args = parser.parse_args()
	return args


if __name__=='__main__':
	from gensim.models import KeyedVectors
	args=get_arguments()
	keyword = args.text
	pretrained_fasttext = KeyedVectors.load('source/models/fasttext_gensim.model')

	print('## {}와 관련한 keywords ##')
	for similar_word in pretrained_fasttext.similar_by_word(keyword, topn=15):
		print("Word: {0}, -------- : {1:.2f}".format(similar_word[0], similar_word[1]))