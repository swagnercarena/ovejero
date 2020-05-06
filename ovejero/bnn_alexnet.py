# -*- coding: utf-8 -*-
"""
Build the TensorFlow model and loss functions

This module contains the functions needed to build the BNN model used in
ovejero as well as the loss functions for the different posteriors.

See the script model_trainer.py for examples of how to use these functions.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers, activations
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Input, Dense
from tensorflow.keras.layers import Layer, InputSpec


class AlwaysDropout(Layer):
	"""
	This class applies dropout to an input both during training and inference.
	This is consistent with the BNN methodology.
	"""
	def __init__(self, dropout_rate, **kwargs):
		"""
		Initialize the AlwaysDropout layer.

		Parameters
		----------
			dropout_rate (float): A number in the range [0,1) that will serve
				as the dropout rate for the layer. A larger rate means more
				dropout.
		"""
		super(AlwaysDropout, self).__init__(**kwargs)
		# Check for a bad dropout input
		if dropout_rate >= 1.0 or dropout_rate < 0.0:
			raise ValueError('dropout rate of %f not between 0 and 1' % (
				dropout_rate))
		# Save the dropout rate for later.
		self.dropout_rate = dropout_rate

	def call(self, inputs, training=None):
		"""
		The function that takes the inputs (likely outputs of a previous layer)
		and conducts dropout.

		Parameters
		----------
		inputs (tf.Keras.Layer): The inputs to the Dense layer.
		training (bool): A required input for call. Setting training to
			true or false does nothing because always dropout behaves the
			same way in both cases.

		Returns
		-------
		(tf.Keras.Layer): The output of the Dense layer.
		"""
		return tf.nn.dropout(inputs, self.dropout_rate)

	def get_config(self):
		"""
		Return the configuration dictionary required by Keras.
		"""
		config = {'dropout_rate': self.dropout_rate}
		base_config = super(AlwaysDropout, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
		"""
		Compute the shape of the output given the input. Needed for Keras
		layer.

		Parameters
		----------
		input_shape ((int,...)): The shape of the input to our Dense layer.

		Returns
		-------
		((int,...)): The output shape of the layer.
		"""
		return input_shape


def cd_regularizer(p, kernel, kernel_regularizer, dropout_regularizer,
	input_dim):
	"""
	Calculate the regularization term for concrete dropout.

	Parameters
	----------
		p (tf.Tensor): A 1D Tensor containing the p value for dropout (between
			0 and 1).
		kernel (tf.Tensor): A 2D Tensor defining the weights of the Dense
			layer
		kernel_initializer (float): The relative strength of kernel
			regularization term.
		dropout_regularizer (float): The relative strength of the dropout
			regularization term.
		input_dim (int): The dimension of the input to the layer.

	Returns
	-------
		(tf.Tensor): The tensorflow graph to calculate the regularization
			term.

	Notes
	-----
	This is currently not being used because of issues with the Keras
		framework. Once it updates this will be employed instead of dividing
		the loss into two parts.
	"""
	regularizer = p * K.log(p)
	regularizer += (1.0 - p) + K.log(1.0 - p)
	regularizer *= dropout_regularizer * input_dim
	regularizer += kernel_regularizer * K.sum(K.square(kernel)) / (1.0 - p)
	return regularizer


class ConcreteDropout(Layer):
	"""
	This class defines a concrete dropout layer that is built around a
	Keras Dense layer. The dropout is parametrized by a weight that is
	optimized along with the model's weights themselves. Heavy inspiration
	from code for arxiv.1705.07832.
	"""
	def __init__(self, output_dim, activation=None,
		kernel_initializer='glorot_uniform', bias_initializer='zeros',
		kernel_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1,
		init_max=0.1, temp=0.1, random_seed=None, **kwargs):
		"""
		Initialize the Concrete dropout Dense layer. This will initialize the
		dense layer along with the overhead needed for concrete dropout.

		Parameters
		----------
			output_dim (int): The number of output parameters
			activation (str): The type of activation function to be used. Will
				be passed into tensorflow's activation function library.
			kernel_initializer (str): The type of initializer to use for the
				kernel. Will be passed to tensorflow's initializer library
			bias_initializer (str): The type of initializer to use for the
				bias. Will be passed to tensorflow's initializer library
			kernel_regularizer (float): The strength of the concrete dropout
				regularization term
			dropout_regularizer (float): The strength of the concrete dropout
				p regularization term
			init_min (float): The minimum initial value of the dropout rate
			init_max (float): The maximum initial value of the dropout rate
			temp (float): The temperature that defines how close the concrete
				distribution will be to true dropout.
			random_seed (int): A seed to use in the random function calls. If
				None no explicit seed will be used.

		Returns
		-------
			(keras.Layer): The initialized ConcreteDropout layer. Must still be
				built.

		Notes
		-----
			Technically the regularization terms must be divided by the number
				of training examples. This is degenerate with the value of the
				regularizers, so we do not specify it here.
			The initial dropout rate will be drawn from a uniform distribution
				with the bounds passed into init.
		"""
		# We do this because Keras does this
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)
		# First initialize the properties required by the Dense class
		super(ConcreteDropout, self).__init__(**kwargs)
		# Save everything important to self
		self.output_dim = output_dim
		self.activation = activations.get(activation)
		self.kernel_initializer = initializers.get(
			kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = kernel_regularizer
		self.dropout_regularizer = dropout_regularizer
		# Convert to logit space (since we want to parameterize our weights
		# such that any value outputted by the network is valid).
		self.init_min = np.log(init_min) - np.log(1.0 - init_min)
		self.init_max = np.log(init_max) - np.log(1.0 - init_max)
		self.temp = temp
		self.random_seed = random_seed

	def build(self, input_shape=None):
		"""
		Build the weights and operations that the network will use.

		Parameters
		----------
			input_shape ((int,...)): The shape of the input to our Dense layer.
		"""
		assert len(input_shape) >= 2
		input_dim = input_shape[-1]

		self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
			initializer=self.kernel_initializer, name='kernel')
		self.bias = self.add_weight(shape=(self.output_dim,),
			initializer=self.bias_initializer, name='bias')
		# Although we define p in logit space, we then apply the sigmoid
		# operation to get the desired value between 0 and 1.
		self.p_logit = self.add_weight(name='p_logit', shape=(1,),
			initializer=initializers.RandomUniform(self.init_min,
				self.init_max), trainable=True)

		# Because of issues with Keras, these functions need to be defined
		# here.
		def p_logit_regularizer(p_logit):
			"""
			Calculate the regularization term for p_logit.

			Parameters
			----------
				p_logit (tf.Tensor): A 1D Tensor containing the p_logit value
					for dropout.

			Returns
			-------
				(tf.Tensor): The tensorflow graph to calculate the
					p_logit regularization term.
			"""
			# Although we define p in logit space, we then apply the sigmoid
			# operation to get the desired value between 0 and 1.
			p = K.sum(K.sigmoid(p_logit))
			regularizer = p * K.log(p)
			regularizer += (1.0 - p) * K.log(1.0 - p)
			regularizer *= self.dropout_regularizer * input_dim
			return regularizer

		def kernel_regularizer(kernel):
			"""
			Calculate the regularization term for concrete dropout.

			Parameters
			----------
				kernel (tf.Tensor): A 2D Tensor containing the kernel for our
					Dense layer computation.

			Returns
			-------
				(tf.Tensor): The tensorflow graph to calculate the
					kernel regularization term.
			"""
			regularizer = self.kernel_regularizer * K.sum(
				K.square(kernel)) / (1.0 - K.sum(K.sigmoid(self.p_logit)))
			return regularizer

		# This is supposed to change in later versions.
		self._handle_weight_regularization('p_logit_regularizer',self.p_logit,
			p_logit_regularizer)
		self._handle_weight_regularization('kernel_regularizer',self.kernel,
			kernel_regularizer)

		# Requirement for Keras
		self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
		self.built = True

	def call(self, inputs, training=None):
		"""
		The function that takes the inputs of the layer and conducts the
		Dense layer multiplication with concrete dropout.

		Parameters
		----------
		inputs (tf.Keras.Layer): The inputs to the Dense layer.
		training (bool): A required input for call. Setting training to
			true or false does nothing because concrete dropout behaves the
			same way in both cases.

		Returns
		-------
		(tf.Keras.Layer): The output of the Dense layer.
		"""
		# Small epsilon parameter needed for stable optimization
		eps = K.cast_to_floatx(K.epsilon())

		# Build the random tensor for dropout from uniform noise. This
		# formulation allows for a derivative with respect to p.
		unif_noise = K.random_uniform(shape=K.shape(inputs),
			seed=self.random_seed)
		drop_prob = (K.log(K.sigmoid(self.p_logit)+eps) - K.log(1.0-
			K.sigmoid(self.p_logit) + eps) + K.log(unif_noise + eps) -
			K.log(1.0 - unif_noise + eps))
		drop_prob = K.sigmoid(drop_prob / self.temp)
		inputs *= (1.0 - drop_prob)
		inputs /= (1.0 - K.sigmoid(self.p_logit))

		# Now just carry out the basic operations of a Dense layer.
		output = K.dot(inputs, self.kernel)
		output = K.bias_add(output, self.bias, data_format='channels_last')
		if self.activation is not None:
			output = self.activation(output)
		return output

	def compute_output_shape(self, input_shape):
		"""
		Compute the shape of the output given the input. Needed for Keras
		layer.

		Parameters
		----------
		input_shape ((int,...)): The shape of the input to our Dense layer.

		Returns
		-------
		((int,...)): The output shape of the layer.
		"""
		output_shape = list(input_shape)
		output_shape[-1] = self.output_dim
		return tuple(output_shape)

	def get_config(self):
		"""
		Return the configuration dictionary required by Keras.
		"""
		config = {
			'output_shape': self.output_shape,
			'activation': activations.serialize(self.activation),
			'kernel_initializer': initializers.serialize(
				self.kernel_initializer),
			'bias_initializer': initializers.serialize(
				self.bias_initializer),
			'kernel_regularizer': self.kernel_regularizer,
			'dropout_regularizer': self.dropout_regularizer
		}
		base_config = super(ConcreteDropout, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class SpatialConcreteDropout(Conv2D):
	"""
	This class defines a spatial concrete dropout layer that is built around a
	Keras Conv2D layer. The dropout is parametrized by a weight that is
	optimized along with the model's weights themselves. Heavy inspiration
	from code for arxiv.1705.07832.
	"""
	def __init__(self, filters, kernel_size, strides=(1,1), padding='valid',
		activation=None, kernel_regularizer=1e-6, dropout_regularizer=1e-5,
		init_min=0.1, init_max=0.1, temp=0.1, random_seed=None, **kwargs):
		"""
		Initialize the Spatial Concrete dropout Dense layer. This will initialize
		the Conv2d layer along with the overhead needed for spatial concrete
		dropout.

		Parameters
		----------
			filters (int): The number of filters to use for the Conv2D layer
			kernel_size ((int,int)): The dimensions of the kernel for the
				Conv2D layer
			strides ((int,int)): The stride to take in each direction for the
				Conv2D layer.
			padding (str): What type of padding to use to get the desired
				output dimensions from the Conv2D layer. Either valid or same
			activation (str): The type of activation function to be used. Will
				be passed into tensorflow's activation function library.
			kernel_regularizer (float): The strength of the concrete dropout
				regularization term
			dropout_regularizer (float): The strength of the concrete dropout
				p regularization term
			init_min (float): The minimum initial value of the dropout rate
			init_max (float): The maximum initial value of the dropout rate
			temp (float): The temperature that defines how close the concrete
				distribution will be to true dropout.
			random_seed (int): A seed to use in the random function calls. If
				None no explicit seed will be used.

		Returns
		-------
			(keras.Layer): The initialized SpatialConcreteDropout layer. Must
				still be built.

		Notes
		-----
			Technically the regularization terms must be divided by the number
				of training examples. This is degenerate with the value of the
				regularizers, so we do not specify it here.
			The initial dropout rate will be drawn from a uniform distribution
				with the bounds passed into init.
		"""
		super(SpatialConcreteDropout, self).__init__(filters, kernel_size,
			strides=strides, padding=padding, activation=activation, **kwargs)
		# Need to change name to avoid issues with Conv2D
		self.cd_kernel_regularizer = kernel_regularizer
		self.dropout_regularizer =dropout_regularizer
		self.init_min = np.log(init_min) - np.log(1.0 - init_min)
		self.init_max = np.log(init_max) - np.log(1.0 - init_max)
		self.temp = temp
		self.random_seed = random_seed

	def build(self, input_shape=None):
		"""
		Build the weights and operations that the network will use.

		Parameters
		----------
			input_shape ((int,...)): The shape of the input to our Conv2D layer.
		"""
		super(SpatialConcreteDropout, self).build(input_shape)
		input_dim = input_shape[3]

		# kernel already set by inherited build function.
		# Although we define p in logit space, we then apply the sigmoid
		# operation to get the desired value between 0 and 1.
		self.p_logit = self.add_weight(name='p_logit',shape=(1,),
			initializer=initializers.RandomUniform(self.init_min,
				self.init_max), trainable=True)

		# Because of issues with Keras, these functions need to be defined
		# here.
		def p_logit_regularizer(p_logit):
			"""
			Calculate the regularization term for p_logit.

			Parameters
			----------
				p_logit (tf.Tensor): A 1D Tensor containing the p_logit value
					for dropout.

			Returns
			-------
				(tf.Tensor): The tensorflow graph to calculate the
					p_logit regularization term.
			"""
			# Although we define p in logit space, we then apply the sigmoid
			# operation to get the desired value between 0 and 1.
			p = K.sum(K.sigmoid(p_logit))
			regularizer = p * K.log(p)
			regularizer += (1.0 - p) * K.log(1.0 - p)
			regularizer *= self.dropout_regularizer * input_dim
			return regularizer

		def kernel_regularizer(kernel):
			"""
			Calculate the regularization term for concrete dropout.

			Parameters
			----------
				kernel (tf.Tensor): A 2D Tensor containing the kernel for our
					Dense layer computation.

			Returns
			-------
				(tf.Tensor): The tensorflow graph to calculate the
					kernel regularization term.
			"""
			regularizer = self.cd_kernel_regularizer * K.sum(
				K.square(kernel)) / (1.0 - K.sum(K.sigmoid(self.p_logit)))
			return regularizer

		# This is supposed to change in later versions.
		self._handle_weight_regularization('p_logit_regularizer',self.p_logit,
			p_logit_regularizer)
		self._handle_weight_regularization('kernel_regularizer',self.kernel,
			kernel_regularizer)

		self.built = True

	def call(self, inputs, training=None):
		"""
		The function that takes the inputs of the layer and conducts the
		Dense layer multiplication with concrete dropout.

		Parameters
		----------
			inputs (tf.Keras.Layer): The inputs to the Dense layer.
			training (bool): A required input for call. Setting training to
				true or false does nothing because concrete dropout behaves the
				same way in both cases.

		Returns
		-------
			(tf.Keras.Layer): The output of the Dense layer.
		"""
		# Small epsilon parameter needed for stable optimization
		eps = K.cast_to_floatx(K.epsilon())

		# Build the random tensor for dropout from uniform noise. This
		# formulation allows for a derivative with respect to p.
		input_shape = K.shape(inputs)
		noise_shape = (input_shape[0], 1, 1, input_shape[3])
		unif_noise = K.random_uniform(shape=noise_shape,
			seed=self.random_seed)
		drop_prob = (K.log(K.sigmoid(self.p_logit)+eps) -
			K.log(1.0-K.sigmoid(self.p_logit)+eps) + K.log(unif_noise + eps)
			- K.log(1.0 - unif_noise + eps))
		drop_prob = K.sigmoid(drop_prob/self.temp)
		inputs *= (1.0 - drop_prob)
		inputs /= (1.0 - K.sigmoid(self.p_logit))

		# Now just carry out the basic operations of a Dense layer.
		return super(SpatialConcreteDropout, self).call(inputs)

	def compute_output_shape(self, input_shape):
		"""
		Compute the shape of the output given the input. Needed for Keras
		layer.

		Parameters
		----------
			input_shape ((int,...)): The shape of the input to our Dense layer.

		Returns
		-------
			((int,...)): The output shape of the layer.
		"""
		return super(SpatialConcreteDropout, self).compute_output_shape(
			input_shape)


def dropout_alexnet(img_size, num_params, kernel_regularizer=1e-6,
	dropout_rate=0.1,random_seed=None):
	"""
	Build the tensorflow graph for the alexnet BNN.

	Parameters
	----------
		img_size ((int,int,int)): A tupe with shape (pix,pix,freq) that describes
			the size of the input images
		num_params (int): The number of lensing parameters to predict
		kernel_regularizer (float): The strength of the l2 norm (associated to
			the strength of the prior on the weights)
		dropout_rate (float): The dropout rate to use for the layers.
		random_seed (int): A seed to use in the random function calls. If None
			no explicit seed will be used.

	Returns
	-------
		(tf.Tensor): The model (i.e. the tensorflow graph for the model)
	"""

	# Initialize model
	inputs = Input(shape=img_size)
	regularizer = tf.keras.regularizers.l2(kernel_regularizer*dropout_rate)

	# Layer 1
	# model.add(AlwaysDropout(dropout_rate))
	x = AlwaysDropout(dropout_rate)(inputs)
	x = Conv2D(filters=64, kernel_size=(5,5), strides=(2,2),
		padding='valid', activation='relu', input_shape=img_size,
		kernel_regularizer=regularizer)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 2
	x = AlwaysDropout(dropout_rate)(x)
	x = Conv2D(filters=192, kernel_size=(5,5), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=regularizer)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 3
	x = AlwaysDropout(dropout_rate)(x)
	x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=regularizer)(x)

	# Layer 4
	x = AlwaysDropout(dropout_rate)(x)
	x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=regularizer)(x)

	# Layer 5
	x = AlwaysDropout(dropout_rate)(x)
	x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=regularizer)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Pass to fully connected layers
	x = Flatten()(x)

	# Layer 6
	x = AlwaysDropout(dropout_rate)(x)
	x = Dense(4096, activation='relu',
		kernel_regularizer=regularizer)(x)

	# Layer 7
	x = AlwaysDropout(dropout_rate)(x)
	x = Dense(4096, activation='relu',
		kernel_regularizer=regularizer)(x)

	# Output
	x = AlwaysDropout(dropout_rate)(x)
	outputs = Dense(num_params,
		kernel_regularizer=regularizer)(x)

	# Construct model
	model = Model(inputs=inputs, outputs=outputs)

	return model


def concrete_alexnet(img_size, num_params, kernel_regularizer=1e-6,
	dropout_regularizer=1e-5, init_min=0.1, init_max=0.1,
	temp=0.1, random_seed=None):
	"""
	Build the tensorflow graph for the concrete dropout alexnet BNN.

	Parameters
	----------
		img_size ((int,int,int)): A tupe with shape (pix,pix,freq) that describes
			the size of the input images
		num_params (int): The number of lensing parameters to predict
		kernel_regularizer (float): The strength of the l2 norm (associated to
			the strength of the prior on the weights)
		dropout_regularizer (float): The stronger it is, the more concrete
			dropout will tend towards larger dropout rates.
		init_min (float): The minimum value that the dropout weight p will
			be initialized to.
		init_max (float): The maximum value that the dropout weight p will
			be initialized to.
		temp (float): The temperature that defines how close the concrete
			distribution will be to true dropout.
		random_seed (int): A seed to use in the random function calls. If None
			no explicit seed will be used.

	Returns
	-------
		(tf.Tensor): The model (i.e. the tensorflow graph for the model)

	Notes
	-----
		While the concrete dropout implementation works, the training of the
		dropout terms is very slow. It's possible that modifying the learning
		rate schedule may help.
	"""

	# Initialize model
	inputs = Input(shape=img_size)

	# Layer 1
	# model.add(AlwaysDropout(dropout_rate))
	x = SpatialConcreteDropout(filters=64, kernel_size=(5,5), strides=(2,2),
		padding='valid', activation='relu', input_shape=img_size,
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer,
		init_min=init_min, init_max=init_max, temp=temp,
		random_seed=random_seed)(inputs)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 2
	x = SpatialConcreteDropout(filters=192, kernel_size=(5,5), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer,
		init_min=init_min, init_max=init_max, temp=temp,
		random_seed=random_seed)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 3
	x = SpatialConcreteDropout(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer,
		init_min=init_min, init_max=init_max, temp=temp,
		random_seed=random_seed)(x)

	# Layer 4
	x = SpatialConcreteDropout(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer,
		init_min=init_min, init_max=init_max, temp=temp,
		random_seed=random_seed)(x)

	# Layer 5
	x = SpatialConcreteDropout(filters=256, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu',
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer,
		init_min=init_min, init_max=init_max, temp=temp,
		random_seed=random_seed)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Pass to fully connected layers
	x = Flatten()(x)

	# Layer 6
	x = ConcreteDropout(4096, activation='relu',
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer, init_min=init_min,
		init_max=init_max, temp=temp, random_seed=random_seed)(x)

	# Layer 7
	x = ConcreteDropout(4096, activation='relu',
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer, init_min=init_min,
		init_max=init_max, temp=temp, random_seed=random_seed)(x)

	# Output
	outputs = ConcreteDropout(num_params,
		kernel_regularizer=kernel_regularizer,
		dropout_regularizer=dropout_regularizer, init_min=init_min,
		init_max=init_max, temp=temp, random_seed=random_seed)(x)

	# Construct model
	model = Model(inputs=inputs, outputs=outputs)

	return model


class LensingLossFunctions:
	"""
	A class used to generate the loss functions for the three types of bayesian
	nn models we have implemented: diagonal covariance, full covariance,
	and mixture of full covariances. Currently only two gaussians are allowed
	in the mixture.
	"""
	def __init__(self,flip_pairs,num_params):
		"""
		Initialize the class with the pairs of parameters that must be flipped.
		These are parameters like shear and ellipticity that have been defined
		such that negating both parameters gives the same
		physical definition of the system.

		Parameters
		----------
			flip_pairs ([[int,int,...],...]): A list of pairs of numbers to
				conduct the flip operation on. If empty no flip pairs will be
				used. Note if you also want to consider two sets of parameters
				being flipped at the same time, that must be added to this list.
			num_params (int): The number of parameters to predict.
		"""

		self.flip_pairs = flip_pairs
		self.num_params = num_params

		# Calculate the split list for lower traingular matrix
		self.split_list = []
		for i in range(1,num_params+1):
			self.split_list += [i]

		# Now for each flip pair (including no flip) we will add a flip
		# matrix to our list.
		self.flip_mat_list = [tf.linalg.diag(tf.constant(np.ones(
			self.num_params),dtype=tf.float32))]
		for flip_pair in self.flip_pairs:
			# Initialize a numpy array since this is the easiest way
			# to flexibly set the tensor.
			const_initializer = np.ones(self.num_params)
			const_initializer[flip_pair] = -1
			self.flip_mat_list.append(tf.linalg.diag(tf.constant(
				const_initializer,dtype=tf.float32)))

	def mse_loss(self, y_true, output):
		"""
		Returns the MSE loss of the predicted parameters. Will ignore parameters
		associated with the covariance matrix.
		Parameters
		----------
			y_true (tf.Tensor): The true values of the parameters
			output (tf.Tensor): The predicted values of the lensing parameters.
				This assumes the first num_params are

		Returns
		-------
			(tf.Tensor): The mse loss function.

		Notes
		-----
			This function should never be used as a loss function. It is useful
			as a metric to understand what portion of the reduciton in the loss
			function can be attributed to improved parameter accuracy. Also
			note that for the gmm models the output will default to the first
			Gaussian for this metric.
		"""
		y_pred, _ = tf.split(output,num_or_size_splits=(self.num_params,-1),
			axis=-1)
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(tf.reduce_mean(tf.square(
				tf.matmul(y_pred,flip_mat)-y_true),axis=-1))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)

	def log_gauss_diag(self,y_true,y_pred,std_pred):
		"""
		Return the negative log posterior of a Gaussian with diagonal
		covariance matrix

		Parameters
		----------
			y_true (tf.Tensor): The true values of the parameters
			y_pred (tf.Tensor): The predicted value of the parameters
			std_pred (tf.Tensor): The predicted diagonal entries of the
				covariance. Note that std_pred is assumed to be the log of the
				covariance matrix values.

		Returns
		-------
			(tf.Tensor): The TF graph for calculating the nlp

		Notes
		-----
			This loss does not include the constant factor of 1/(2*pi)^(d/2).
		"""
		return 0.5*tf.reduce_sum(tf.multiply(tf.square(y_pred-y_true),
			tf.exp(-std_pred)),axis=-1) + 0.5*tf.reduce_sum(
			std_pred,axis=-1)

	def diagonal_covariance_loss(self,y_true,output):
		"""
		Return the loss function assuming a diagonal covariance matrix

		Parameters
		----------
			y_true (tf.Tensor): The true values of the lensing parameters
			output (tf.Tensor): The predicted values of the lensing parameters.
				This should include 2*self.num_params parameters to account for
				the diagonal entries of our covariance matrix. Covariance matrix
				values are assumed to be in log space.

		Returns
		-------
			(tf.Tensor): The loss function (i.e. the tensorflow graph for it).
		"""
		# First split the data into predicted parameters and covariance matrix
		# element
		y_pred, std_pred = tf.split(output,num_or_size_splits=2,axis=-1)

		# Add each possible flip to the loss list. We will then take the
		# minimum.
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(self.log_gauss_diag(y_true,
				tf.matmul(y_pred,flip_mat),std_pred))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)

	def construct_precision_matrix(self,L_mat_elements):
		"""
		Take the matrix elements for the log cholesky decomposition and
		convert them to the precision matrix. Also return the value of
		the diagonal elements before exponentiation, since we get that for
		free.

		Parameters
		----------
			L_mat_elements (tf.Tensor): A tensor of length
				num_params*(num_params+1)/2 that define the lower traingular
				matrix elements of the log cholesky decomposition

		Returns
		-------
			((tf.Tensor,tf.Tensor)): Both the precision matrix and the diagonal
				elements (before exponentiation) of the log cholesky L matrix.
				Note that this second value is important for the posterior
				calculation.
		"""
		# First split the tensor into the elements that will populate each row
		cov_elements_split = tf.split(L_mat_elements,
			num_or_size_splits=self.split_list,axis=-1)
		# Before we stack these elements, we have to pad them with zeros
		# (corresponding to the 0s of the lower traingular matrix).
		cov_elements_stack = []
		pad_offset = 1
		for cov_element in cov_elements_split:
			# Use tf pad function since it's likely the fastest option.
			pad = tf.constant([[0,0],[0,self.num_params-pad_offset]])
			cov_elements_stack.append(tf.pad(cov_element,pad))
			pad_offset+=1
		# Stack the tensors to form our matrix. Use axis=-2 to avoid issues
		# with batches of matrices being passed in.
		L_mat = tf.stack(cov_elements_stack,axis=-2)
		# Pull out the diagonal part, and then (since we're using log
		# cholesky) exponentiate the diagonal.
		L_mat_diag = tf.linalg.diag_part(L_mat)
		L_mat = tf.linalg.set_diag(L_mat,tf.exp(L_mat_diag))
		# Calculate the actual precision matrix
		prec_mat = tf.matmul(L_mat,tf.transpose(L_mat,perm=[0,2,1]))

		return prec_mat, L_mat_diag, L_mat

	def log_gauss_full(self,y_true,y_pred,prec_mat,L_diag):
		"""
		Return the negative log posterior of a Gaussian with full
		covariance matrix

		Parameters
		----------
			y_true (tf.Tensor): The true values of the parameters
			y_pred (tf.Tensor): The predicted value of the parameters
			prec_mat: The precision matrix
			L_diag (tf.Tensor): The diagonal (non exponentiated) values of the
				log cholesky decomposition of the precision matrix

		Returns
		-------
			(tf.Tensor): The TF graph for calculating the nlp

		Notes
		-----
			This loss does not include the constant factor of 1/(2*pi)^(d/2).
		"""
		y_dif = y_true - y_pred
		return -tf.reduce_sum(L_diag,-1) + 0.5 * tf.reduce_sum(
			tf.multiply(y_dif,tf.reduce_sum(tf.multiply(tf.expand_dims(
				y_dif,-1),prec_mat),axis=-2)),-1)

	def full_covariance_loss(self,y_true,output):
		"""
		Return the loss function assuming a full covariance matrix

		Parameters
		----------
			y_true (tf.Tensor): The true values of the lensing parameters
			output (tf.Tensor): The predicted values of the lensing parameters.
				This should include self.num_params parameters for the prediction
				and self.num_params*(self.num_params+1)/2 parameters for the
				lower triangular log cholesky decomposition

		Returns
		-------
			(tf.Tensor): The loss function (i.e. the tensorflow graph for it).
		"""
		# Start by dividing the output into the L_elements and the prediction
		# values.
		L_elements_len = int(self.num_params*(self.num_params+1)/2)
		y_pred, L_mat_elements = tf.split(output,
			num_or_size_splits=[self.num_params,L_elements_len],axis=-1)

		# Build the precision matrix and extract the diagonal part
		prec_mat, L_diag, _ = self.construct_precision_matrix(L_mat_elements)

		# Add each possible flip to the loss list. We will then take the
		# minimum.
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(self.log_gauss_full(y_true,
				tf.matmul(y_pred,flip_mat),prec_mat,L_diag))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)

	def log_gauss_gm_full(self,y_true,y_preds,prec_mats,L_diags,pis):
		"""
		Return the negative log posterior of a GMM with full
		covariance matrix for each GM. Note this code allows for any number
		of GMMs.

		Parameters
		----------
			y_true (tf.Tensor): The true values of the parameters
			y_preds ([tf.Tensor,...]): A list of the predicted value of the
				parameters
			prec_mats ([tf.Tensor,...]): A list of the precision matrices
			L_diags ([tf.Tensor,...]): A list of the diagonal (non exponentiated)
				values of the log cholesky decomposition of the precision
				matrices

		Returns
		-------
			(tf.Tensor): The TF graph for calculating the nlp

		Notes
		-----
			This loss does not include the constant factors of 1/(2*pi)^(d/2).
		"""
		# Stack together the loss to be able to do the logsumexp trick
		loss_list = []
		for p_i in range(len(y_preds)):
			# Since we're multiplying the probabilities, we don't want the
			# negative here.
			loss_list.append(-self.log_gauss_full(y_true,y_preds[p_i],
				prec_mats[p_i],L_diags[p_i])+tf.squeeze(tf.math.log(pis[p_i]),
				axis=-1))

		# Use tf implementation of logsumexp
		return -tf.reduce_logsumexp(tf.stack(loss_list,axis=-1),axis=-1)

	def gm_full_covariance_loss(self,y_true,output):
		"""
		Return the loss function assuming a mixture of two gaussians each with
		a full covariance matrix

		Parameters
		----------
			y_true (tf.Tensor): The true values of the lensing parameters
			output (tf.Tensor): The predicted values of the lensing parameters.
				This should include 2 gm which consists of self.num_params
				parameters for the prediction and
				self.num_params*(self.num_params+1)/2 parameters for the
				lower triangular log cholesky decomposition of each gm.
				It should also include one final parameter for the ratio
				between the two gms.

		Returns
		-------
			(tf.Tensor): The loss function (i.e. the tensorflow graph for it).
		"""
		# Start by seperating out the predictions for each gaussian model.
		L_elements_len = int(self.num_params*(self.num_params+1)/2)
		y_pred1, L_mat_elements1, y_pred2, L_mat_elements2, pi_logit = tf.split(
			output,num_or_size_splits=[self.num_params,L_elements_len,
			self.num_params,L_elements_len,1],axis=-1)

		# Set the probability between 0.5 and 1.0. In this parameterization the
		# first Gaussian is always favored.
		pi = 0.5+tf.sigmoid(pi_logit)/2.0

		# Now build the precision matrix for our two models and extract the
		# diagonal components used for the loss calculation
		prec_mat1, L_diag1, _ = self.construct_precision_matrix(L_mat_elements1)
		prec_mat2, L_diag2, _ = self.construct_precision_matrix(L_mat_elements2)

		# Add each possible flip to the loss list. We will then take the
		# minimum.
		loss_list = []
		prec_mats = [prec_mat1,prec_mat2]
		L_diags = [L_diag1,L_diag2]
		pis = [pi,1-pi]
		for flip_mat1 in self.flip_mat_list:
			for flip_mat2 in self.flip_mat_list:
				# The y_preds depends on the selected flips
				y_preds = [tf.matmul(y_pred1,flip_mat1),
					tf.matmul(y_pred2,flip_mat2)]
				loss_list.append(self.log_gauss_gm_full(y_true,y_preds,
					prec_mats,L_diags,pis))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)


def p_value(model):
	"""
	Returns the average value of the dropout in each concrete layer.

	Parameters
	----------
		model (keras.Model): A Keras model from with the dropout values will be
			extracted.

	Notes
	-----
		This is a hack that allows us to easily keep track of the dropout value
		during training.
	"""
	def p_fake_loss(y_true,y_pred):
		# We won't be using either y_true or y_pred
		loss = []
		for layer in model.layers:
			if 'dropout' in layer.name:
				loss.append(tf.sigmoid(layer.weights[2]))
		return tf.reduce_mean(loss)

	return p_fake_loss
