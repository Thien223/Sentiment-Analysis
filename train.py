import keras.backend as K
from keras import constraints
from keras import initializers, regularizers
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.models import Sequential



def dot_product(x, kernel):
    """
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
		x (): input
		kernel (): weights
	Returns:
	"""
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
	Attention operation, with a context/query vector, for temporal data.
	Supports Masking.
	follows these equations:

	(1) u_t = tanh(W h_t + b)
	(2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
	(3) v_t = \alpha_t * h_t, v in time t
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		3D tensor with shape: `(samples, steps, features)`.
	"""

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's.
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class Addition(Layer):
    """
	This layer is supposed to add of all activation weight.
	We split this from AttentionWithContext to help us getting the activation weights
	follows this equation:
	(1) v = \sum_t(\alpha_t * h_t)

	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		2D tensor with shape: `(samples, features)`.
	"""

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def initializing(inputs_size=120, hidden_units=128, num_layers=3, max_sequence_length=120):
    num_classes = 2  ### final  output dim
    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model = Sequential()
    model.add(Embedding(inputs_size, output_dim=128, input_length=max_sequence_length))
    for i in range(num_layers):
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=0.5,kernel_initializer=initializers.glorot_normal(seed=777),bias_initializer='zeros')))
        model.add(AttentionWithContext())
        model.add(Addition())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=777),bias_initializer='zeros'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    model.summary()
    return model


def train_lstm(train_data, train_labels, test_data, test_labels):
    batch_size = 12
    print(train_data)
    train_data = train_data.reshape(-1,1)
    # print(train_data)
    print(train_data.shape)
    model = initializing()
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=100, shuffle=True, verbose=1)
    pred = model.predict(test_data)
    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.plot(test_labels)


#
# def test(test_model, test_data, test_labels, show_mistake=False):
#     test_predictions = test_model.predict(test_data, verbose=0)
#     # TEST PERFORMANCE
#     res_accu = eval.accuracy(test_predictions, test_labels)
#     res_f1 = eval.fscore(test_predictions, test_labels)
#     res_recall = eval.recall(test_predictions, test_labels)
#     res_precision = eval.precision(test_predictions, test_labels)
#     print('Test Accuracy: %.3f' % res_accu)
#     print('Test F1-score: %.3f' % res_f1)
#     print('Test Recall: %.3f' % res_recall)
#     print('Test Precision: %.3f' % res_precision)
#
#     return res_accu, res_f1, res_recall, res_precision
#

#
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_type", dest="model_type", default="LSTM", help="default=LSTM, LSTM/SVM", metavar="FILE")
#     parser.add_argument("--is_attention", dest="is_attention", default=True, help="default=True, use attention mechanism",
#                       metavar="FILE")
#     parser.add_argument("--is_finetune", dest="is_finetune", default=True, help="default=True, use fine tuning",
#                       metavar="FILE")
#     parser.add_argument("--hidden_units", dest="hidden_units", default=64, help="default=64, number of hidden units",
#                       metavar="FILE")
#     parser.add_argument("--num_layers", dest="num_layers", default=1,
#                       help="default=1, number of layers (only for LSTM/BLSTM models)", metavar="FILE")
#     parser.add_argument("--is_bidirectional", dest="is_bidirectional", default=True, help="default=True, use BLSTM",
#                       metavar="FILE")
#     parser.add_argument("--input_data_path", dest="input_data_path", default="../../dataset", help="input datapath",
#                       metavar="FILE")
#     parser.add_argument("--max_sequence_length", dest="max_sequence_length", default=35, help="max sequence length",
#                       metavar="FILE")
#     parser.add_argument("--word_embedding_path", dest="word_embedding_path", default=None,
#                       help="word embedding in numpy array format path", metavar="FILE")
#     parser.add_argument("--vocab_path", dest="vocab_path", default=None, help="vocab in numpy array path", metavar="FILE")
#     parser.add_argument("--skip_preprocess", dest="skip_preprocess", default=True,
#                       help="load preprocess files, skip preprocessing part", metavar="FILE")
#     parser.add_argument("--validation_split", dest="validation_split", default=0.2, help="validation split", metavar="FILE")
#     return parser.parse_args()


def load_data(file_path):
    with open(file_path, mode='r') as file:
        meta = file.readline()
        sentences = meta.split('|')[:-1]
        inputs = [x.split('-')[0] for x in sentences]
        outputs = [x.split('-')[1] for x in sentences]
        # inputs = [np.array(x.split('-')[0]) for x in sentences]
        # outputs = [np.array(x.split('-')[1]) for x in sentences]
    # return (np.asarray(inputs), np.asarray(outputs))
    return (inputs, outputs)


if __name__ == '__main__':
    ##### load data
    train_data, train_labels = load_data('data/processed/train.txt')
    test_data, test_labels = load_data('data/processed/test.txt')



    train_lstm(train_data, train_labels, test_data, test_labels)
