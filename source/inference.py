
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from source.classification_model import load_data_from_database
from source.modules.Attention import Attention
from source.sentiment_models import prepare_data

if __name__=='__main__':

	import keras

	model_type='classification_model'
	if model_type=='classification_model':
		########## inference
		model = keras.models.load_model('source/models/keyword_classification.h5', custom_objects={'Attention': Attention})
		input_train, output_train, input_test, output_test, mapper, class_w = load_data_from_database(train=False)
		pred = model.predict(x=input_test, batch_size=256, verbose=1)

		predicted_index = np.argmax(pred, axis=1)
		real_index = np.argmax(output_test, axis=1)

		aaa = predicted_index - real_index
		print((aaa==0).sum())



	else: #### model_type='sentiment_analysis'

		model = keras.models.load_model('source/models/keyword_classification.h5', custom_objects={'Attention': Attention})
		input_train, output_train, input_test, output_test, mapper = prepare_data(train_path='data/processed/ratings_train.txt', test_path='data/processed/ratings_test.txt', max_text_length=50)


		### inference
		pred = model.predict(x=input_test, batch_size=256, verbose=1)
		result = pd.DataFrame({'Pred_0': pred[:, 0], 'real_0': output_test[:, 0], 'Pred_1': pred[:, 1], 'real_1': output_test[:, 1]})

		result.to_csv('data/result_.csv')