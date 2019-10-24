import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input

# ovejero code uses the concrete_dropout code described in arxiv.1705.07832
from concrete_dropout import ConcreteDropout, SpatialConcreteDropout

def concrete_alexnet(img_size, num_params, weight_regularizer=1e-6,
	dropout_regularizer=1e-5):
	"""
	Builds the tensorflow graph for the concrete dropout BNN.

	Parameters:
		img_size: A tupe with shape (pix,pix,freq) that describes the size of the
			input images
		num_params: The number of lensing parameters to predict
		weight_regularizer: The strength of the l2 norm (associated to the 
			strength of the prior on the weights)
		dropout_regularizer: The stronger it is, the more concrete dropout
			will tend towards larger dropout rates.

	Returns:
		The model (i.e. the tensorflow graph for the model)
	"""

	# Initialize model
	inputs = Input(shape=img_size)

	# Layer 1
	# model.add(Always_Dropout(dropout_rate))
	x = SpatialConcreteDropout(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), 
		padding='valid', activation='relu', input_shape=img_size),
		weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(inputs)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 2
	x = SpatialConcreteDropout(Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), 
		padding='same', activation='relu'),weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 3
	x = SpatialConcreteDropout(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
		padding='same', activation='relu'),weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)

	# Layer 4
	x = SpatialConcreteDropout(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
		padding='same', activation='relu'),weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)

	# Layer 5
	x = SpatialConcreteDropout(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
		padding='same', activation='relu'),weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Pass to fully connected layers
	x = Flatten()(x)

	# Layer 6
	x = ConcreteDropout(Dense(4096, activation='relu'),
		weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)

	# Layer 7
	x = ConcreteDropout(Dense(4096, activation='relu'),
		weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)

	# Output
	outputs = ConcreteDropout(Dense(num_params),
		weight_regularizer=weight_regularizer,
		dropout_regularizer=dropout_regularizer)(x)

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
			These are parameters like shear and ellipticity that have been
			defined such that negating both parameters gives the same 
			physical definition of the system.

		Parameters:
			flip_pairs: A list of pairs of numbers to conduct the flip operation
				on. If empty no flip pairs will be used. Note if you also want
				to consider two sets of parameters being flipped at the same 
				time, that must be added to this list.
			num_params: The number of parameters to predict.

		Returns:
			The initialized class.
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

	def log_gauss_diag(self,y_true,y_pred,std_pred):
		"""
		Returns the negative log likelihood of a Gaussian with diagonal
		covariance matrix

		Parameters:
			y_true: The true values of the parameters
			y_pred: The predicted value of the parameters
			std_pred: The predicted diagonal entries of the covariance. Note
				that std_pred is assumed to be the log of the covariance matrix
				values.

		Returns:
			The TF graph for calculating the nll
		"""
		return 0.5*tf.reduce_sum(tf.multiply(tf.square(y_pred-y_true),
			tf.exp(-std_pred)),axis=-1) + 0.5*tf.reduce_sum(
			std_pred,axis=-1)

	def diagonal_covariance_loss(self,y_true,output):
		"""
		Loss function assuming a diagonal covariance matrix

		Parameters:
			y_true: The true values of the lensing parameters
			output: The predicted values of the lensing parameters. This should
				include 2*self.num_params parameters to account for the
				diagonal entries of our covariance matrix. Covariance matrix
				values are assumed to be in log space.

		Returns:
			The loss function (i.e. the tensorflow graph for it).
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
		return tf.reduce_min(loss_stack,axis=1)

	def construct_precision_matrix(self,L_mat_elements):
		"""
		Take the matrix elements for the log cholesky decomposition and
		convert them to the precision matrix. Will also return the value of
		the diagonal elements before exponentiation, since we get that for 
		free.
		
		Parameters:
			L_mat_elements: A tensor of length num_params*(num_params+1)/2 that
				define the lower traingular matrix elements of the log cholesky
				decomposition

		Returns:
			Both the precision matrix and the diagonal elements (before 
				exponentiation) of the log cholesky L matrix. Note that
				this second value is important for the posterior calculation.
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

		return prec_mat, L_mat_diag

	def log_gauss_full(self,y_true,y_pred,prec_mat,L_diag):
		"""
		Returns the negative log likelihood of a Gaussian with full
		covariance matrix

		Parameters:
			y_true: The true values of the parameters
			y_pred: The predicted value of the parameters
			prec_mat: The precision matrix
			L_diag: The diagonal (non exponentiated) values of the log
				cholesky decomposition of the precision matrix

		Returns:
			The TF graph for calculating the nll
		"""
		y_dif = y_true - y_pred
		return -tf.reduce_sum(L_diag,-1) + 0.5 * tf.reduce_sum(
			tf.multiply(y_dif,tf.reduce_sum(tf.multiply(tf.expand_dims(
				y_dif,-1),prec_mat),axis=-2)),-1)

	def full_covariance_loss(self,y_true,output):
		"""
		Loss function assuming a full covariance matrix

		Parameters:
			y_true: The true values of the lensing parameters
			y_pred: The predicted values of the lensing parameters. This should
				include self.num_params parameters for the prediction and
				self.num_params*(self.num_params+1)/2 parameters for the
				lower triangular log cholesky decomposition

		Returns:
			The loss function (i.e. the tensorflow graph for it).
		"""
		# Start by dividing the y_pred into the L_elements and the prediction
		# values.
		L_elements_len = int(self.num_params*(self.num_params+1)/2)
		y_pred, L_mat_elements = tf.split(output,
			num_or_size_splits=[self.num_params,L_elements_len],axis=-1)

		# Build the precision matrix and extract the diagonal part
		prec_mat, L_diag = self.construct_precision_matrix(L_mat_elements)

		# Add each possible flip to the loss list. We will then take the
		# minimum.
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(self.log_gauss_full(y_true,
				tf.matmul(y_pred,flip_mat),prec_mat,L_diag))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=1)



