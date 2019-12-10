import torch
from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from torch import nn

#### Luong attention using pytorch
class Multiplicative_Attention(nn.Module):
	''''
	implementation of Luong's attention
	'''
	def __init__(self, hidden_size, method="dot"):
		super(Multiplicative_Attention, self).__init__()
		self.method = method
		self.hidden_size = hidden_size

		# Defining the layers/weights required depending on alignment scoring method
		if method == "general":
			self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

		elif method == "concat":
			self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
			self.weight = nn.Parameter(torch.FloatTensor((1,hidden_size)), requires_grad=False)

	def forward(self, decoder_hidden, encoder_outputs):
		'''
		calculate attention score from decoder hidden state and encoder output
		:param decoder_hidden:
		:param encoder_outputs:
		:return: attention score (alignment scores)
		### to use the attention score, calculate attention weight from attention score with: attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
		### then calculate context vector from attention weight and encoder output with: context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs)
		### then concatenate attention context vector with decoder's output with: output = torch.cat((lstm_out, context_vector),-1)
		### create linear layer: linear = nn.Linear(self.hidden_size*2, self.output_size)
		#### run output through a linear layer,
		#### final output is log_sofmax of above output: final output = torch.nn.functional.log_softmax(linear(output[0]), dim=1)
		### more info here: https://blog.floydhub.com/attention-mechanism/
		'''
		if self.method == "dot":
			# For the dot scoring method, no weights or linear layers are involved
			attention_score= encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)
			return attention_score

		elif self.method == "general":
			# For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
			out = self.fc(decoder_hidden)
			attention_score = encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)
			return attention_score

		elif self.method == "concat":
			# For concat scoring, decoder hidden state and encoder outputs are concatenated first
			out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
			attention_score = out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)
			return attention_score

### attention using keras layer


class Attention(Layer):
	# Input shape 3D tensor with shape: `(samples, steps, features)`.
	# Output shape 2D tensor with shape: `(samples, features)`.

	def __init__(self, step_dim, W_regulizer=None, b_regulizer=None,
				 W_constraint=None, b_constraint=None, bias=True, **kwargs):

		self.W_regulizer = W_regulizer
		self.b_regulizer = b_regulizer

		self.W_constraint = W_constraint
		self.b_constraint = b_constraint

		self.bias = bias
		self.step_dim = step_dim
		self.features_dim = 0
		self.init = initializers.get('glorot_uniform')
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		### accept 3D shape  (samples, steps, features)
		assert len(input_shape) == 3
		### Create a trainable weight variable for this layer.
		self.W = self.add_weight(name='{}_W'.format(self.name),
								 shape=(input_shape[-1],),
								 initializer=self.init,
								 constraint=self.W_constraint,
								 regularizer=self.W_regulizer,
								 trainable=True)
		### get features length
		self.features_dim = input_shape[-1]

		if self.bias:
			self.b = self.add_weight(name='{}_b'.format(self.name),
									 shape=(input_shape[1],),
									 initializer='zero',
									 regularizer=self.b_regulizer,
									 constraint=self.b_constraint)
		else:
			self.b = None
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		'''
		calculate
		:param x: input tensor
		:param mask:
		:return:
		'''
		features_dim = self.features_dim
		step_dim = self.step_dim

		eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a

		return K.sum(weighted_input, axis=1)


	def compute_mask(self, input, input_mask=None):
		return None

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.features_dim

	def get_config(self):
		config = super(Attention, self).get_config()
		config.update({'step_dim':self.step_dim,
				  'W_regulizer':self.W_regulizer,
				  'b_regulizer':self.b_regulizer,
				  'W_constraint':self.W_constraint,
				  'b_constraint':self.b_constraint ,
				  'bias':self.bias})
		return config
