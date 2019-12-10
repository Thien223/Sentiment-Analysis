#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, Input, Bidirectional, LSTM, Dropout, Activation
from keras.models import Model
from keras.preprocessing import text, sequence
from kor2vec import Kor2Vec
from tqdm import tqdm
from sklearn import preprocessing
from source.TextProcessing.TextPreprocessing import clean_training_data
from source.modules.Attention import Attention
from source.utils import connect_to_data_base

### global variables
db = connect_to_data_base()
cursor = db.cursor()



def _plot_bar(x, title='Number observations of each class'):
	'''
	plot count of each column in x (x is a numpy array)
	:param x:
	:param title: title of image
	:return: nothing
	'''
	x=output_train.reshape(-1,)
	import matplotlib.pyplot as plt
	x, y = np.unique(x, return_counts=True)
	x_lab = [str(lab) for lab in x]
	plt.bar(x_lab, y)
	plt.xlabel('Class label')
	plt.ylabel('Number observations')
	plt.title(title)
	for i in range(len(x)):  # your number of bars
		plt.text(x=x[i] - 0.5,  # takes your x values as horizontal positioning argument
				 y=y[i] + 1,  # takes your y values as vertical positioning argument
				 s=y[i],  # the labels you want to add to the data
				 size=10)




def load_data_from_database(train = False):
	### save data for latter testing (because with csv, Hangul results unreadable characters, we will save as xlsx file)
	from sklearn import preprocessing  ### for categorical output, use label encode of sklearn to convert to one hot vector

	if train:
		query = 'select * from google_keywords order by insert_date DESC;'
		cursor.execute(query)
		data = cursor.fetchall()
		cols = [field[0] for field in cursor.description]
		data=pd.DataFrame(data, columns=cols)
		data.to_excel('data/keyword_data_.xlsx', encoding='utf-8')
	else:
		data = pd.read_excel('data/keyword_data_.xlsx', encoding='utf-8')

	### randomly shuffle data
	data = data.sample(frac=1)

	categories = copy.deepcopy(data['category'])
	for i, cate in enumerate(categories):
		cate_tokens = cate.split(' ')
		new_cate = cate_tokens[-1]
		categories[i] = new_cate
	input_data = copy.deepcopy(data['keyword'].astype(str))
	output_data = categories



	## calculate max sentences
	max_sentences = len(input_data)

	## use tokenizer class of keras to transform text to sequences
	tokenizer = text.Tokenizer(num_words=max_sentences)
	tokenizer.fit_on_texts(list(input_data))
	list_tokenized_input_all = tokenizer.texts_to_sequences(input_data)
	### pad text sequences to unique length
	label_encoder = preprocessing.OneHotEncoder()

	output_all = label_encoder.fit_transform(np.asarray(output_data).reshape(-1,1)).toarray()

	mapper={}
	for idx,label in zip(np.argmax(output_all, axis=1), output_data):
		mapper[label] = idx

	pointer = int(len(input_data) * 0.8)
	#### train-test split
	input_all = sequence.pad_sequences(list_tokenized_input_all, maxlen=50)
	input_train = input_all[:pointer,:]
	input_test = input_all[pointer:,:]
	# ### reshape output due to number of output classes (for output that has form of numeric)
	# output_train = np.asarray(list_label_train, dtype=int).reshape(-1,output_classes_dim)
	# output_test = np.asarray(list_label_test, dtype=int).reshape(-1,output_classes_dim)
	# output_all = mlb.fit_transform(output_data)

	#### with texture output, convert to sequence
	output_train =output_all[:pointer,:]
	output_test = output_all[pointer:,:]

	output_data_ = categories[:pointer]
	from sklearn.utils import class_weight
	class_w = class_weight.compute_class_weight('balanced', np.unique(output_data_.values), output_data_.values.reshape(-1, ))

	return input_train, output_train, input_test, output_test, mapper, class_w



def get_arguments():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_path', type=str, default='source/models/lstm_attention_best_model.h5', help='Whether embedding model will be trained')
	parser.add_argument('--preprocess', type=bool, default=False, help='Whether clean data before training')
	args = parser.parse_args()
	return args

def lstm_attention_model(max_input_sequence_length,output_dimensions, pretrained_embedding_sequences,activation_func='softmax', lstm_units=320, dense_units=256, dropout_rate = 0.45):
	'''
	prepare model graph,
	:param max_input_sequence_length: max length of input sequences tensor
	:param output_dimensions: number of classification classes
	:param pretrained_embedding_sequences: for using separate embedding model, pass the pretrained embedding model's weight
	:param activation_func: activation function for final output layer
	:param lstm_units: lstm hidden units
	:param dense_units: dense layer hidden unit
	:param dropout_rate: dropout rate use for both lstm and dense layer
	:return: model, early_stopping, model_checkpoint (for using when fitting model latter)
	'''
	### todo: remake
	### extract pretrained embedding model weight
	pretrained_embedding_matrix = np.zeros((pretrained_embedding_sequences.shape[0], 512))
	for i, vec in enumerate(pretrained_embedding_sequences):
		pretrained_embedding_matrix[i] = vec
	### create layers
	lstm_layer = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))
	### pass pretrained_embedding_weight matrix to Embedding layer. trainable=True allows weight to be updated, else the weight will me keep remain.
	### trainable = True: allow embedding_matrix weight to be updated while training
	embedding_layer = Embedding(input_dim=pretrained_embedding_sequences.shape[0], output_dim=512, weights=[pretrained_embedding_matrix], input_length=max_input_sequence_length, trainable=True)
	input_ = Input(shape=(max_input_sequence_length,))
	embedding_sequences = embedding_layer(input_)
	lstm_layer_1_output = lstm_layer(embedding_sequences)
	dropout_layer_output = Dropout(dropout_rate)(lstm_layer_1_output)
	attention = Attention(step_dim=max_input_sequence_length)
	with_attention_output = attention(dropout_layer_output)
	dense_layer_output = Dense(dense_units, activation='relu')(with_attention_output)
	dropout_layer_2_output = Dropout(dropout_rate)(dense_layer_output)
	final_output = Dense(output_dimensions, activation=activation_func)(dropout_layer_2_output)
	model = Model(inputs=input_, outputs=final_output)
	return model



def _process_text_data(path = 'data/processed/ratings_train.txt'):
	'''
	embed text to sequence using kor2vec model (pretrained)
	text sequences will be saved in data/processed/text_squence.npy
	text label will be saved in data/processed/text_label.npy
	these files will be used to extract embedding model weight to pass to embedding layer of keras model
	:param path: path to text (and label) file
	:return: sequence and corressponding label
	### this might take time
	'''
	texts = []
	labels = []
	k2v = Kor2Vec.load('source/models/k2v.model')
	with open(path,'r', encoding='utf-8') as f:
		for line in tqdm(f):
			line_ = line.replace('\n','')
			text_id, text, text_label = line_.split('\t')
			labels.append(text_label)
			texts.append(text)
	text_seqs = k2v.embedding(texts,seq_len=1, numpy=True)
	label_seqs = np.array(labels, dtype=np.int)
	np.save('data/processed/text_label_.npy', label_seqs.astype(np.int), allow_pickle=False)
	np.save('data/processed/text_squence_.npy', text_seqs.astype(np.float32), allow_pickle=False)
	return text_seqs, label_seqs


def prepare_data(train_path='data/processed/ratings_train.txt', test_path='data/processed/ratings_test.txt',max_text_length=50):
	'''
	preparing data into train input, output and validation input, output.
	:param train_path: path to training text file
	:param test_path: path to testing text file
	:param max_text_length: max length of text, text which is shorter than this, will be padded to unique length
	:param output_classes_dim: number of output classes (2 with binary classification)
	:return:
	'''

	from sklearn.preprocessing import MultiLabelBinarizer
	mlb = MultiLabelBinarizer()
	list_sentences_train = []
	list_label_train = []
	list_sentences_test = []
	list_label_test = []
	### put train texts and labels to lists
	with open(train_path, encoding='utf8') as train_file:
		for line in train_file:
			list_sentences_train.append(line.replace('\n','').split('\t')[1])
			list_label_train.append(line.replace('\n','').split('\t')[2])
	del line

	### put test texts and labels to lists
	with open(test_path, encoding='utf8') as test_file:
		for line in test_file:
			list_sentences_test.append(line.replace('\n','').split('\t')[1])
			list_label_test.append(line.replace('\n','').split('\t')[2])
	del line
	## calculate max sentences
	max_sentences = len(list_sentences_train) + len(list_sentences_test)


	## use tokenizer class of keras to transform text to sequences
	tokenizer = text.Tokenizer(num_words=max_sentences)
	tokenizer.fit_on_texts(list(list_sentences_train))
	list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
	list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

	### pad text sequences to unique length
	input_train = sequence.pad_sequences(list_tokenized_train, maxlen=max_text_length)
	input_test = sequence.pad_sequences(list_tokenized_test, maxlen=max_text_length)
	# ### reshape output due to number of output classes
	# output_train = np.asarray(list_label_train, dtype=int).reshape(-1,output_classes_dim)
	# output_test = np.asarray(list_label_test, dtype=int).reshape(-1,output_classes_dim)
	output_train =  mlb.fit_transform(list_label_train)
	output_test =  mlb.fit_transform(list_label_test)
	return input_train, output_train, input_test, output_test


if __name__=='__main__':

	import keras
	import tensorflow as tf

	config = tf.ConfigProto(device_count={'GPU': 1})
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)


	args = get_arguments()
	if args.preprocess:
		clean_training_data(input_path='data/ratings_train.txt', output_path='data/processed/ratings_train.txt')
		clean_training_data(input_path='data/ratings_test.txt', output_path='data/processed/ratings_test.txt')
	### preparing train data
	## load korean to vector model
	### open train data file

	text_label = np.load('data/processed/text_label_.npy')
	embedded_sequence = np.load('data/processed/text_squence_.npy')
	input_train, output_train, input_test, output_test, _, class_w = load_data_from_database(train=True)
	# # del embedded_sequence, text_sequences
	# # del text_sequences
	model = lstm_attention_model(max_input_sequence_length=50, output_dimensions=output_train.shape[1], pretrained_embedding_sequences=embedded_sequence, activation_func='softmax')
	### prepare data here
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())
	early_stopping = EarlyStopping(patience=2)
	###
	#### check model in-out shape and real in-out data shape
	print('model input shape: {}'.format(model.input_shape))
	print('model output shape: {}'.format(model.output_shape))
	print('input data shape: {}'.format(input_train.shape))
	print('input data shape: {}'.format(output_train.shape))

	### fit model
	hist = model.fit(x=input_train,
					 y=output_train,
					 epochs=200,
					 batch_size=256,
					 shuffle=True,
					 callbacks=[early_stopping],
					 class_weight=class_w,
					 verbose=1)
	## save model
	model.save('source/models/keyword_classification.h5')





############# backup

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import copy
#
# import numpy as np
# import pandas as pd
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers import Dense, Embedding, Input, Bidirectional
# from keras.layers import LSTM, Dropout
# from keras.models import Model
# from keras.preprocessing import text, sequence
# from konlpy.tag import Kkma
# from kor2vec import Kor2Vec
# from tqdm import tqdm
#
# from source.TextProcessing.TextPreprocessing import clean_training_data
# from source.modules.Attention import Attention
# from source.utils import connect_to_data_base
#
# ### global variables
# db = connect_to_data_base()
# cursor = db.cursor()
#
# def load_data_from_database():
# 	from sklearn import preprocessing ### for categorical output, use label encode of sklearn to convert to one hot vector
#
# 	query = 'select * from google_keywords order by insert_date DESC;'
# 	cursor.execute(query)
# 	data = cursor.fetchall()
# 	cols = [field[0] for field in cursor.description]
# 	data=pd.DataFrame(data, columns=cols)
#
#
# 	categories = copy.deepcopy(data['category'])
# 	for i, cate in enumerate(categories):
# 		cate_tokens = cate.split(' ')
# 		new_cate = cate_tokens[-1]
# 		categories[i] = new_cate
# 		len(set(categories))
# 	input_data = copy.deepcopy(data['keyword'])
# 	output_data = categories
#
# 	## calculate max sentences
# 	max_sentences = len(input_data)
#
# 	## use tokenizer class of keras to transform text to sequences
# 	tokenizer = text.Tokenizer(num_words=max_sentences)
# 	tokenizer.fit_on_texts(list(input_data))
# 	list_tokenized_input_all = tokenizer.texts_to_sequences(input_data)
# 	### pad text sequences to unique length
# 	label_encoder = preprocessing.OneHotEncoder()
# 	label_encoder.fit(np.asarray(output_data).reshape(-1,1))
# 	output_all = label_encoder.transform(output_data.values.reshape(-1,1)).toarray()
#
# 	pointer = int(len(input_data) * 0.8)
# 	#### train-test split
# 	input_all = sequence.pad_sequences(list_tokenized_input_all, maxlen=50)
# 	input_train = input_all[:pointer,:]
# 	input_test = input_all[pointer:,:]
# 	# ### reshape output due to number of output classes (for output that has form of numeric)
# 	# output_train = np.asarray(list_label_train, dtype=int).reshape(-1,output_classes_dim)
# 	# output_test = np.asarray(list_label_test, dtype=int).reshape(-1,output_classes_dim)
# 	# output_all = mlb.fit_transform(output_data)
#
# 	#### with texture output, convert to sequence
# 	output_train =output_all[:pointer,:]
# 	output_test = output_all[pointer:,:]
# 	return input_train, output_train, input_test, output_test
#
#
#
# def get_arguments():
# 	import argparse
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--checkpoint_path', type=str, default='source/models/lstm_attention_best_model.h5', help='Whether embedding model will be trained')
# 	parser.add_argument('--preprocess', type=bool, default=True, help='Whether clean data before training')
# 	args = parser.parse_args()
# 	return args
#
# def lstm_attention_model(max_input_sequence_length,output_dimensions, pretrained_embedding_sequences,activation_func='softmax', lstm_units=320, dense_units=256, dropout_rate = 0.45):
# 	'''
# 	prepare model graph,
# 	:param max_input_sequence_length: max length of input sequences tensor
# 	:param output_dimensions: number of classification classes
# 	:param pretrained_embedding_sequences: for using separate embedding model, pass the pretrained embedding model's weight
# 	:param activation_func: activation function for final output layer
# 	:param lstm_units: lstm hidden units
# 	:param dense_units: dense layer hidden unit
# 	:param dropout_rate: dropout rate use for both lstm and dense layer
# 	:return: model, early_stopping, model_checkpoint (for using when fitting model latter)
# 	'''
# 	### todo: remake
# 	### extract pretrained embedding model weight
# 	pretrained_embedding_matrix = np.zeros((pretrained_embedding_sequences.shape[0], 512))
# 	for i, vec in enumerate(pretrained_embedding_sequences):
# 		pretrained_embedding_matrix[i] = vec
# 	### create layers
# 	lstm_layer = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))
# 	### pass pretrained_embedding_weight matrix to Embedding layer. trainable=True allows weight to be updated, else the weight will me keep remain.
# 	### trainable = True: allow embedding_matrix weight to be updated while training
# 	embedding_layer = Embedding(input_dim=pretrained_embedding_sequences.shape[0], output_dim=512, weights=[pretrained_embedding_matrix], input_length=max_input_sequence_length, trainable=True)
# 	input_ = Input(shape=(max_input_sequence_length,))
# 	embedding_sequences = embedding_layer(input_)
# 	lstm_layer_1_output = lstm_layer(embedding_sequences)
# 	dropout_layer_output = Dropout(dropout_rate)(lstm_layer_1_output)
# 	attention = Attention(step_dim=max_input_sequence_length)
# 	with_attention_output = attention(dropout_layer_output)
# 	dense_layer_output = Dense(dense_units, activation='relu')(with_attention_output)
# 	dropout_layer_2_output = Dropout(dropout_rate)(dense_layer_output)
# 	final_output = Dense(output_dimensions, activation=activation_func)(dropout_layer_2_output)
# 	model = Model(inputs=[input_], outputs=final_output)
# 	return model
#
#
#
# def _process_text_data(path = 'data/processed/ratings_train.txt'):
# 	'''
# 	embed text to sequence using kor2vec model (pretrained)
# 	text sequences will be saved in data/processed/text_squence.npy
# 	text label will be saved in data/processed/text_label.npy
# 	these files will be used to extract embedding model weight to pass to embedding layer of keras model
# 	:param path: path to text (and label) file
# 	:return: sequence and corressponding label
# 	### this might take time
# 	'''
# 	texts = []
# 	labels = []
# 	k2v = Kor2Vec.load('source/models/k2v.model')
# 	with open(path,'r', encoding='utf-8') as f:
# 		for line in tqdm(f):
# 			line_ = line.replace('\n','')
# 			text_id, text, text_label = line_.split('\t')
# 			labels.append(text_label)
# 			texts.append(text)
# 	text_seqs = k2v.embedding(texts,seq_len=1, numpy=True)
# 	label_seqs = np.array(labels, dtype=np.int)
# 	np.save('data/processed/text_label_.npy', label_seqs.astype(np.int), allow_pickle=False)
# 	np.save('data/processed/text_squence_.npy', text_seqs.astype(np.float32), allow_pickle=False)
# 	return text_seqs, label_seqs
#
#
# def prepare_data(train_path='data/processed/ratings_train.txt', test_path='data/processed/ratings_test.txt',max_text_length=50):
# 	'''
# 	preparing data into train input, output and validation input, output.
# 	:param train_path: path to training text file
# 	:param test_path: path to testing text file
# 	:param max_text_length: max length of text, text which is shorter than this, will be padded to unique length
# 	:param output_classes_dim: number of output classes (2 with binary classification)
# 	:return:
# 	'''
#
# 	from sklearn.preprocessing import MultiLabelBinarizer
# 	mlb = MultiLabelBinarizer()
# 	list_sentences_train = []
# 	list_label_train = []
# 	list_sentences_test = []
# 	list_label_test = []
# 	### put train texts and labels to lists
# 	with open(train_path, encoding='utf8') as train_file:
# 		for line in train_file:
# 			list_sentences_train.append(line.replace('\n','').split('\t')[1])
# 			list_label_train.append(line.replace('\n','').split('\t')[2])
# 	del line
#
# 	### put test texts and labels to lists
# 	with open(test_path, encoding='utf8') as test_file:
# 		for line in test_file:
# 			list_sentences_test.append(line.replace('\n','').split('\t')[1])
# 			list_label_test.append(line.replace('\n','').split('\t')[2])
# 	del line
# 	## calculate max sentences
# 	max_sentences = len(list_sentences_train) + len(list_sentences_test)
#
# 	## use tokenizer class of keras to transform text to sequences
# 	tokenizer = text.Tokenizer(num_words=max_sentences)
# 	tokenizer.fit_on_texts(list(list_sentences_train))
# 	list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# 	list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#
# 	### pad text sequences to unique length
# 	input_train = sequence.pad_sequences(list_tokenized_train, maxlen=max_text_length)
# 	input_test = sequence.pad_sequences(list_tokenized_test, maxlen=max_text_length)
# 	# ### reshape output due to number of output classes
# 	# output_train = np.asarray(list_label_train, dtype=int).reshape(-1,output_classes_dim)
# 	# output_test = np.asarray(list_label_test, dtype=int).reshape(-1,output_classes_dim)
# 	output_train =  mlb.fit_transform(list_label_train)
# 	output_test =  mlb.fit_transform(list_label_test)
# 	return input_train, output_train, input_test, output_test
#
#
# if __name__=='__main__':
# 	args = get_arguments()
# 	if args.preprocess:
# 		clean_training_data(input_path='data/ratings_train.txt', output_path='data/processed/ratings_train.txt')
# 		clean_training_data(input_path='data/ratings_test.txt', output_path='data/processed/ratings_test.txt')
# 	### preparing train data
# 	## load korean to vector model
# 	### open train data file
#
# 	text_label = np.load('data/processed/text_label_.npy')
# 	embedded_sequence = np.load('data/processed/text_squence_.npy')
# 	input_train, output_train, input_test, output_test = load_data_from_database()
# 	# # del embedded_sequence, text_sequences
# 	# # del text_sequences
# 	model = lstm_attention_model(max_input_sequence_length=50, output_dimensions=output_train.shape[1], pretrained_embedding_sequences=embedded_sequence, activation_func='softmax')
# 	### prepare data here
# 	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 	print(model.summary())
# 	best_model_path = 'args.checkpoint_path'
# 	early_stopping = EarlyStopping(patience=2)
# 	model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
# 	###
# 	#### check model in-out shape and real in-out data shape
# 	print('model input shape: {}'.format(model.input_shape))
# 	print('model output shape: {}'.format(model.output_shape))
# 	print('input data shape: {}'.format(input_train.shape))
# 	print('input data shape: {}'.format(output_train.shape))
#
# 	### fit model
# 	hist = model.fit(x=input_train,
# 					 y=output_train,
# 					 epochs=20,
# 					 batch_size=256,
# 					 shuffle=True,
# 					 callbacks=[early_stopping, model_checkpoint],
# 					 verbose=1)
# 	## save model
# 	model.save('source/models/keyword_classification.h5')
#
# 	### inference
# 	pred = model.predict(x=input_test, batch_size=256, verbose=1)
#
# 	pred_ = pd.DataFrame(pred)
# 	pred_['result'] = pred_.idxmax(axis=1)
# 	real_ = pd.DataFrame(output_test)
# 	real_['result'] = real_.idxmax(axis=1)
# 	result = pd.DataFrame({'Pred_0': pred_['result'] , 'real_0': real_['result']})
# 	result.to_csv('data/result__.csv')
