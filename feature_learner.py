from utils import generate_embedding_matrix

from keras.applications.vgg16 import VGG16
from keras.layers import Embedding, LSTM, GRU, Lambda, Activation, \
	Flatten
from keras import backend as K

class FeatureLearner:
	def __init__(self, args):
		self.wt = args.wt
		self.mode = args.feature_learning_mode
		self.glove_embed_size = int(args.glove_embed_size)
		self.rep_dims = int(args.rep_dims)
		self.batch_size = int(args.batch_size)

		self.embedding_matrix = generate_embedding_matrix(self.wt, args.embedding_dir, self.glove_embed_size)

		self.embedding = Embedding(
			name='embedded_question',
			input_dim=len(self.embedding_matrix),
			output_dim=self.glove_embed_size,
			weights=[self.embedding_matrix],
			batch_input_shape=(self.batch_size, None),
			input_length=None,
			trainable=False,
			mask_zero=False
		)

		if self.mode == 'lstm':
			self.summarizer = LSTM(self.rep_dims, return_sequences=False)
		elif self.mode == 'gru':
			self.summarizer = GRU(self.rep_dims, return_sequences=False)
		elif 'pooling' in self.mode:
			split = self.mode.split('_')
			self.pooling_method = split[0]
			self.mode = split[1]
			self.summarizer = self.pool

		self.base_img_model = VGG16(weights='imagenet', include_top=False, input_shape=args.img_dims)
		for layer in self.base_img_model.layers:
		    layer.trainable = False

	def pool(self, ctx):
		if self.pooling_method == 'avg':
			condensed = Lambda(lambda x: K.mean(x, axis=-1))(ctx)
		elif self.pooling_method == 'mean':
			condensed = Lambda(lambda x: K.max(x, axis=-1))(ctx)
		else:
			raise Exception('Only avg and mean pooling are supported at this juncture.')

		return Dense(self.rep_dims)(condensed)


	def vgg_features(self, data):
		return Flatten()(self.base_img_model(data))

	def embed(self, data):
		return self.embedding(data)

	def summarize(self, encoded_rep):
		return self.summarizer(encoded_rep)

