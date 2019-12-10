import string

from gensim.utils import tokenize
from kor2vec import Kor2Vec
import re

from tqdm import tqdm


def get_arguments():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_embedding_model', type=bool, default=True, help='Whether embedding model will be trained')
	args = parser.parse_args()
	return args

def clean_training_data(input_path, output_path, index_ = 0, text_index=1, label_index=2, mode='a'):
	'''
	For data that has been formatted as [id, text, text_label] each line, read each line of input data, write down only text to new file
	for data is an document (KAIST corpus, refer to clean_training_document_data function
	:param input_path: input texts
	:param output_path: output texts
	:param index_: index of id column
	:param text_index: column index of text
	:param label_index: column index of text label
	:return:
	'''
	import re
	remove_special_char = re.compile('r[^a-z\d]', re.IGNORECASE)
	with open(input_path, 'r', encoding='utf8') as f:
		with open(output_path, mode, encoding='utf-8') as f_:
			for line in f:
				try:
					_line = remove_special_char.sub('', line).split('\t')
					print(_line)
					index_ = _line[index_]
					text = _line[text_index]
					label = _line[label_index]
					f_.write(index_+'\t'+text+'\t'+label)
					# print(unicode(line_).encode('utf-8'))
				except IndexError as e:
					print(line)
					continue

def clean_training_document_data(input_path, output_path, mode='a'):
	'''
	for data is an document (KAIST corpus) read whole file, remove redundant line, separate document to
	:param input_path: input texts
	:param output_path: output texts
	:param mode: write output file as new file or append to old one
	:return:
	'''
	# input_path='data/kaist_text_corpus/mass-media/magazine/kaistcorpus_written_raw_or_mass-media_magazine_na0301_1.txt'
	# input_path='data/kaist_text_corpus/mass-media/news/kaistcorpus_written_raw_or_mass-media_news_kbs8-1_6.txt'
	remove_special_char = re.compile('r[^a-z\d]', re.IGNORECASE)
	doc = []
	### with each line, read and exclude line start with <, * and # (description, comment lines)
	with open(input_path, 'r', encoding='cp949') as f:
		for line in f.readlines(): #### KAIST corpus has first 11 line as document's description, exclude them
			if line.strip() =='' or line[0] in ['<', '*','#','⊙']:
				pass
			else:
				_line = remove_special_char.sub('', line).strip().split('\t')
				#### add text line to document list
				doc.append(_line[0])
	doc = ' '.join(doc)

	import konlpy
	### using konlpy library to split document to sentences
	processor = konlpy.tag.Kkma()
	sentences = processor.sentences(doc)
	### write sentence to output file, each sentence one line
	with open(output_path, mode='a', encoding='utf-8') as f_:
		for sentence in sentences:
			f_.write(sentence+'\n')
	return

def clean_text(text):
	text = re.sub('<.*?>', '', text).strip()
	text = re.sub('(\s)+', r'\1', text)
	return text

def sentence_segment(text):
	sents = re.split("([.?!])?[\n]+|[.?!] ", text)
	return sents

def word_segment(sent):
	sent = tokenize(sent.decode('utf-8'))
	return sent


def normalize_text(text):
	listpunctuation = string.punctuation.replace('_', '')
	for i in listpunctuation:
		text = text.replace(i, ' ')
	return text.lower()


def load_vectors(fname):
	'''
	load pretrained embedding vector of words from txt file (often trained with word2vec or fasttext model)
	:param fname: file path
	:return: dictionary of [word: word_vector]
	'''
	with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
		data = {}
		for line in tqdm(fin.readlines()[1:]):
			tokens = line.rstrip().split(' ')
			data[tokens[0]] = map(float, tokens[1:])
	return data

# aaa = load_vectors(fname='f:/cc.ko.300.vec')

def train_kor2vec(input_path,output_path='models/k2v.model', embedding_size=300, batch_size=128):
	'''
	:param input_path: text corpus
	:param output_path: file path to save the model
	:param embedding_size: size of embedding table
	:param batch_size: batch size
	:return: Nothing, just export model to file and store in folder
	'''

	k2v = Kor2Vec(embed_size=embedding_size)
	k2v.train_(input_path, batch_size=batch_size)  # takes some time
	k2v.save(path=output_path)  # saving embedding
	print('===== trained kor2vec model =====')
	print('===== outputed as {} ====='.format(output_path))
	return k2v


def train_fasttext(input_path, embedding_size=300, window_size=40, min_word = 5, down_sampling = 1e-2):
	'''
	:param input_path: text corpus
	:param window_size: fasttext param
	:param embedding_size: fasttext param
	:param min_word: fasttext param
	:param down_sampling: fasttext param
	:return: Nothing, just export model to file and store in folder
	'''
	from gensim.models import KeyedVectors
	from gensim.models.fasttext import FastText
	import konlpy
	pretrained_fasttext = KeyedVectors.load_word2vec_format('f:/cc.ko.300.vec', encoding='utf-8', unicode_errors='ignore')

	# Getting the tokens
	words = []
	for word in pretrained_fasttext.vocab:
		words.append(word)

	fasttext_model = FastText(size=300,window=50,min_count=7,sample= 1e-2,sg=1,iter=100)
	
	fasttext_model.train(words, total_examples=fasttext_model.corpus_count, epochs=fasttext_model.iter)

	# Pick a word


	keywords = list(categories)
	keyword = '애니메이션'
	for keyword in keywords:
		# Finding out similar words [default= top 10]
		print('## {}와 관련한 keywords ##')
		for similar_word in pretrained_fasttext.similar_by_word(keyword):
			print("Word: {0}, Similarity: {1:.2f}".format(similar_word[0], similar_word[1]))


	### using konlpy library to split sentences to words
	processor = konlpy.tag.Kkma()
	input_path='data/processed/train_embedding.txt'
	new_embedding_data = []
	i=0
	with open(input_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as file_input:
		for sentence in file_input:
			i+=1
			new_words = processor.nouns(sentence)
			new_embedding_data.extend(new_words)



	return k2v

if __name__ == '__main__':
	args = get_arguments()
	### test korean to vec model
	k2v = None
	#### train embedding model
	input_path = 'data/processed/all_text.txt'
	train_embedding_model=args.train_embedding_model
	if train_embedding_model==False:
		try:
			k2v=Kor2Vec.load('source/models/k2v.model')
		except FileNotFoundError as e:
			raise('Pretrained embedding model is not found, check if the file is exist, or train new one..')
	else:
		assert input_path is not None, 'To train embedding model, we need input corpus'
		train_kor2vec(input_path, output_path='source/models/k2v.model', embedding_size=512, batch_size=128)





	#### preprocessing document text data
	folder = ['data/kaist_text_corpus/utility/health/',
			  'data/kaist_text_corpus/literature/autobiography/',
			  'data/kaist_text_corpus/literature/biography/',
			  'data/kaist_text_corpus/literature/criticism/',
			  'data/kaist_text_corpus/literature/diary/',
			  'data/kaist_text_corpus/literature/essay/',
			  'data/kaist_text_corpus/literature/juvenileAndfable/',
			  'data/kaist_text_corpus/literature/novel/',
			  'data/kaist_text_corpus/literature/poem/',
			  'data/kaist_text_corpus/literature/theatre/',
			  'data/kaist_text_corpus/law/96law/',
			  'data/kaist_text_corpus/law/97law/',
			  'data/kaist_text_corpus/generality/art97/',
			  'data/kaist_text_corpus/generality/etc95/',
			  'data/kaist_text_corpus/generality/explanation95/',
			  'data/kaist_text_corpus/generality/memoirs95/',
			  'data/kaist_text_corpus/generality/mungobon94/',
			  'data/kaist_text_corpus/generality/science96/',
			  'data/kaist_text_corpus/generality/social95/',
			  'data/kaist_text_corpus/generality/social96/',
			  'data/kaist_text_corpus/generality/social97/',
			  'data/kaist_text_corpus/generality/speech95/',
			  'data/kaist_text_corpus/generality/travel95/',
			  'data/kaist_text_corpus/mass-media/magazine/',
			  'data/kaist_text_corpus/mass-media/newspaper/',
			  'data/kaist_text_corpus/religion/bible/',
			  'data/kaist_text_corpus/religion/bouddism/',
			  'data/kaist_text_corpus/textbook/textbook/']
	import os
	for f in folder:
		files = [f + file for file in os.listdir(f)]
		for file in files:
			try:
				clean_training_document_data(input_path=file, output_path='data/processed/train_embedding.txt', mode='a')
			except Exception as e:
				print(e)
				break