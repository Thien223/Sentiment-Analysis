#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, Input, Bidirectional
from keras.layers import LSTM, Dropout
from keras.models import Model
from keras.preprocessing import text, sequence
from kor2vec import Kor2Vec
from tqdm import tqdm
from source.TextProcessing.TextPreprocessing import clean_training_data
from source.modules.Attention import Attention


def get_arguments():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_path', type=str, default='source/models/lstm_attention_best_model.h5', help='Whether embedding model will be trained')
	parser.add_argument('--preprocess', type=bool, default=True, help='Whether clean data before training')
	args = parser.parse_args()
	return args

def lstm_attention_model(max_input_sequence_length,output_dimensions, pretrained_embedding_sequences,activation_func='sigmoid', lstm_units=320, dense_units=256, dropout_rate = 0.45):
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
	### input sequence (not embedded) of text
	input_ = Input(shape=(max_input_sequence_length,))
	### embedding them
	embedding_sequences = embedding_layer(input_)
	lstm_layer_1_output = lstm_layer(embedding_sequences)
	dropout_layer_output = Dropout(dropout_rate)(lstm_layer_1_output)
	attention = Attention(step_dim=max_input_sequence_length)
	with_attention_output = attention(dropout_layer_output)
	dense_layer_output = Dense(dense_units, activation='relu')(with_attention_output)
	dropout_layer_2_output = Dropout(dropout_rate)(dense_layer_output)
	final_output = Dense(output_dimensions, activation=activation_func)(dropout_layer_2_output)
	model = Model(inputs=[input_], outputs=final_output)
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
	# output_classes_dim=1
	# output_train = np.asarray(list_label_train, dtype=int).reshape(-1,output_classes_dim)
	# output_test = np.asarray(list_label_test, dtype=int).reshape(-1,output_classes_dim)
	output_train =  mlb.fit_transform(list_label_train)
	output_test =  mlb.fit_transform(list_label_test)
	return input_train, output_train, input_test, output_test


if __name__=='__main__':
	args = get_arguments()
	if args.preprocess:
		clean_training_data(input_path='data/ratings_train.txt', output_path='data/processed/ratings_train.txt')
		clean_training_data(input_path='data/ratings_test.txt', output_path='data/processed/ratings_test.txt')
	### preparing train data
	## load korean to vector model
	### open train data file

	text_label = np.load('data/processed/text_label_.npy')
	text_sequences = np.load('data/processed/text_squence_.npy')
	input_train, output_train, input_test, output_test = prepare_data(train_path='data/processed/ratings_train.txt', test_path='data/processed/ratings_test.txt', max_text_length=50)

	embedded_sequence = text_sequences
	# # del embedded_sequence, text_sequences
	# # del text_sequences
	model = lstm_attention_model(max_input_sequence_length=50, output_dimensions=2, pretrained_embedding_sequences=embedded_sequence, activation_func='softmax')
	### prepare data here
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print(model.summary())
	best_model_path = 'args.checkpoint_path'
	early_stopping = EarlyStopping(patience=2)
	model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
	###
	#### check model in-out shape and real in-out data shape
	print('model input shape: {}'.format(model.input_shape))
	print('model output shape: {}'.format(model.output_shape))
	print('input data shape: {}'.format(input_train.shape))
	print('input data shape: {}'.format(output_train.shape))

	### fit model
	hist = model.fit(x=input_train,
					 y=output_train,
					 epochs=20,
					 batch_size=256,
					 shuffle=True,
					 callbacks=[early_stopping, model_checkpoint],
					 verbose=1)
	## save model
	model.save('source/models/sentiment_analysis.h5')
