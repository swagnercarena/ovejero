import unittest, os
import numpy as np
import tensorflow as tf
from ovejero import bnn_alexnet
from scipy.stats import multivariate_normal
# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BNNTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(BNNTests, self).__init__(*args, **kwargs)
		self.random_seed = 1234
		tf.random.set_seed(self.random_seed)
		np.random.seed(self.random_seed)

	def test_AlwaysDropout(self):
		# Test that the implementation of Always dropout behaves as expected.
		# Start with no dropout and make sure that behaves how you want it to.
		input_layer = tf.ones((200,200,200))
		dropout_rate = 0
		d_layer = bnn_alexnet.AlwaysDropout(dropout_rate)
		output_layer = d_layer(input_layer)
		np.testing.assert_equal(input_layer.numpy(),output_layer.numpy())

		dropout_rate = 0.1
		d_layer = bnn_alexnet.AlwaysDropout(dropout_rate)
		output_layer = d_layer(input_layer)
		# Test that the two arrays aren't equal.
		self.assertGreater(np.mean(np.abs(input_layer.numpy()-output_layer.numpy()
			)),0)
		# Test that the mean value hasn't changed (remember we divide the output
		# by the dropout rate so the mean is unchanged)
		self.assertAlmostEqual(np.mean(input_layer.numpy()),
			np.mean(output_layer.numpy()),places=3)
		# Test that the median value is as expected.
		self.assertAlmostEqual(np.median(output_layer.numpy()),1/0.9,places=5)

		# Repeat the above tests for other dropout rates.
		dropout_rate = 0.5
		d_layer = bnn_alexnet.AlwaysDropout(dropout_rate)
		output_layer = d_layer(input_layer)
		self.assertGreater(np.mean(np.abs(input_layer.numpy()-output_layer.numpy()
			)),0)
		self.assertAlmostEqual(np.mean(input_layer.numpy()),
			np.mean(output_layer.numpy()),places=2)

		dropout_rate = 0.9
		d_layer = bnn_alexnet.AlwaysDropout(dropout_rate)
		output_layer = d_layer(input_layer)
		self.assertGreater(np.mean(np.abs(input_layer.numpy()-output_layer.numpy()
			)),0)
		self.assertAlmostEqual(np.mean(input_layer.numpy()),
			np.mean(output_layer.numpy()),places=2)
		self.assertEqual(np.median(output_layer.numpy()),0.0)

	def test_ConcreteDropout(self):
		# Test that our implementation of ConcreteDropout works as expected.
		output_dim = 100
		activation = 'relu'
		kernel_regularizer = 1e-6
		dropout_regularizer = 1e-5
		init_min = 0.1
		init_max = 0.1
		input_shape = (None,200)

		cd_layer = bnn_alexnet.ConcreteDropout(output_dim,activation=activation,
			kernel_regularizer=kernel_regularizer,
			dropout_regularizer=dropout_regularizer, init_min=init_min,
			init_max=init_max)
		cd_layer.build(input_shape=input_shape)

		# Check that all of the weights have the right shapes
		kernel = cd_layer.weights[0]
		bias = cd_layer.weights[1]
		p_logit = cd_layer.weights[2]
		self.assertListEqual(list(kernel.shape),[200,100])
		self.assertListEqual(list(bias.shape),[100])
		self.assertListEqual(list(p_logit.shape),[1])

		# Check that the initializations worked as we wanted them to
		self.assertEqual(np.sum(bias.numpy()),0)
		self.assertEqual(p_logit.numpy(),np.log(0.1)-np.log(1-0.1))

		# Check that the losses for the layer is what we would expect for
		# concrete dropout.
		p_logit_reg = cd_layer.losses[0].numpy()
		kernel_reg = cd_layer.losses[1].numpy()
		# We know what we set p to
		p = 0.1
		p_logit_correct = p * np.log(p) + (1-p)*np.log(1-p)
		p_logit_correct *= dropout_regularizer * 200
		self.assertAlmostEqual(p_logit_reg, p_logit_correct)
		kernel_correct = kernel_regularizer * np.sum(np.square(
			kernel.numpy())) / (1-p)
		self.assertAlmostEqual(kernel_reg, kernel_correct)

		# Now check that the call function doesn't return the same value each
		# time
		false_input = tf.constant((np.random.rand(1,200)),dtype=tf.float32)
		output1 = cd_layer(false_input).numpy()
		output2 = cd_layer(false_input).numpy()
		self.assertGreater(np.sum(np.abs(output1-output2)),1)

	def test_SpatialConcreteDropout(self):
		# Test that our implementation of ConcreteDropout works as expected.
		filters = 64
		kernel_size = (5,5)
		activation = 'relu'
		kernel_regularizer = 1e-6
		dropout_regularizer = 1e-5
		init_min = 0.1
		init_max = 0.1
		input_shape = (None,20,20,64)

		cd_layer = bnn_alexnet.SpatialConcreteDropout(filters, kernel_size,
			activation=activation,
			kernel_regularizer=kernel_regularizer,
			dropout_regularizer=dropout_regularizer, init_min=init_min,
			init_max=init_max)
		cd_layer.build(input_shape=input_shape)

		# Check that all of the weights have the right shapes
		kernel = cd_layer.weights[0]
		bias = cd_layer.weights[1]
		p_logit = cd_layer.weights[2]
		self.assertListEqual(list(kernel.shape),[5,5,64,64])
		self.assertListEqual(list(bias.shape),[64])
		self.assertListEqual(list(p_logit.shape),[1])

		# Check that the initializations worked as we wanted them to
		self.assertEqual(np.sum(bias.numpy()),0)
		self.assertEqual(p_logit.numpy(),np.log(0.1)-np.log(1-0.1))

		# Check that the losses for the layer is what we would expect for
		# concrete dropout.
		p_logit_reg = cd_layer.losses[0].numpy()
		kernel_reg = cd_layer.losses[1].numpy()
		# We know what we set p to
		p = 0.1
		p_logit_correct = p * np.log(p) + (1-p)*np.log(1-p)
		p_logit_correct *= dropout_regularizer * 64
		self.assertAlmostEqual(p_logit_reg, p_logit_correct)
		kernel_correct = kernel_regularizer * np.sum(np.square(
			kernel.numpy())) / (1-p)
		self.assertAlmostEqual(kernel_reg, kernel_correct)

		# Now check that the call function doesn't return the same value each
		# time
		false_input = tf.constant((np.random.rand(1,20,20,64)),dtype=tf.float32)
		output1 = cd_layer(false_input).numpy()
		output2 = cd_layer(false_input).numpy()
		self.assertGreater(np.sum(np.abs(output1-output2)),1)

	def test_concrete_alexnet(self):
		# Test that the models initialized agree with what we intended
		layer_names = ['input','spatial_concrete_dropout','max_pooling2d',
			'spatial_concrete_dropout','max_pooling2d',
			'spatial_concrete_dropout','spatial_concrete_dropout',
			'spatial_concrete_dropout','max_pooling2d','flatten',
			'concrete_dropout','concrete_dropout','concrete_dropout']

		image_size = (100,100,1)
		num_params = 8
		model = bnn_alexnet.concrete_alexnet(image_size, num_params,
			kernel_regularizer=1e-6,dropout_regularizer=1e-5)
		input_shapes = [[],(100,100,1),(48,48,64),
			(24,24,64),(24,24,192),(12,12,192),(12,12,384),(12,12,384),
			(12,12,256),(6,6,256),(9216,),(4096,),(4096,)]
		output_shapes = [[]]+input_shapes[2:] + [(num_params,)]

		l_i = 0
		# All I can really check is that the layers are of the right type and
		# have the right shapes
		for layer in model.layers:
			self.assertTrue(layer_names[l_i] in layer.name)
			self.assertEqual(layer.dtype,tf.float32)
			self.assertEqual(layer.input_shape[1:],input_shapes[l_i])
			self.assertEqual(layer.output_shape[1:],output_shapes[l_i])
			# Check that all the concrete dropout layer except the last have
			# a ReLU activation function.
			if 'concrete' in layer.name and l_i < len(model.layers)-1:
				self.assertEqual(layer.activation,tf.keras.activations.relu)
			l_i += 1

	def test_dropout_alexnet(self):
		# Test that the models initialized agree with what we intended
		layer_names = ['input','always_dropout','conv2d','max_pooling2d',
			'always_dropout','conv2d','max_pooling2d','always_dropout',
			'conv2d','always_dropout','conv2d','always_dropout',
			'conv2d','max_pooling2d','flatten','always_dropout','dense',
			'always_dropout','dense','always_dropout','dense']

		image_size = (100,100,1)
		num_params = 8
		# Kernel regularizer and dropout rate
		kr = 1e-6
		dr = 0.1
		model = bnn_alexnet.dropout_alexnet(image_size, num_params,
			kernel_regularizer=kr,dropout_rate=dr)
		input_shapes = [[],(100,100,1),(100,100,1),(48,48,64),
			(24,24,64),(24,24,64),(24,24,192),(12,12,192),(12,12,192),
			(12,12,384),(12,12,384),(12,12,384),(12,12,384),(12,12,256),
			(6,6,256),(9216,),(9216,),(4096,),(4096,),(4096,),(4096,)]
		output_shapes = [[]]+input_shapes[2:] + [(num_params,)]

		# All I can really check is that the layers are of the right type and
		# have the right shapes
		for l_i, layer in enumerate(model.layers):
			self.assertTrue(layer_names[l_i] in layer.name)
			self.assertEqual(layer.dtype,tf.float32)
			self.assertEqual(layer.input_shape[1:],input_shapes[l_i])
			self.assertEqual(layer.output_shape[1:],output_shapes[l_i])
			# Check that all the concrete dropout layer except the last have
			# a ReLU activation function.
			if 'conv2d' in layer.name:
				self.assertEqual(layer.activation,tf.keras.activations.relu)
				self.assertEqual(layer.kernel_regularizer.l2,np.array(kr*(1-dr),
					dtype=np.float32))
			if 'dense' in layer.name and l_i < len(model.layers)-1:
				self.assertEqual(layer.activation,tf.keras.activations.relu)
				self.assertEqual(layer.kernel_regularizer.l2,np.array(kr*(1-dr),
					dtype=np.float32))

		# Repeat the test for dropout of 0
		layer_names = ['input','conv2d','max_pooling2d','conv2d',
			'max_pooling2d','conv2d','conv2d','conv2d','max_pooling2d','flatten',
			'dense','dense','dense']

		image_size = (100,100,1)
		num_params = 8
		dr = 0.0
		model = bnn_alexnet.dropout_alexnet(image_size, num_params,
			kernel_regularizer=kr,dropout_rate=dr)
		input_shapes = [[],(100,100,1),(48,48,64),
			(24,24,64),(24,24,192),(12,12,192),
			(12,12,384),(12,12,384),(12,12,256),
			(6,6,256),(9216,),(4096,),(4096,)]
		output_shapes = [[]]+input_shapes[2:] + [(num_params,)]

		# All I can really check is that the layers are of the right type and
		# have the right shapes
		for l_i, layer in enumerate(model.layers):
			self.assertTrue(layer_names[l_i] in layer.name)
			self.assertEqual(layer.dtype,tf.float32)
			self.assertEqual(layer.input_shape[1:],input_shapes[l_i])
			self.assertEqual(layer.output_shape[1:],output_shapes[l_i])
			# Check that all the concrete dropout layer except the last have
			# a ReLU activation function.
			if 'conv2d' in layer.name:
				self.assertEqual(layer.activation,tf.keras.activations.relu)
				self.assertEqual(layer.kernel_regularizer.l2,np.array(kr*(1-dr),
					dtype=np.float32))
			if 'dense' in layer.name and l_i < len(model.layers)-1:
				self.assertEqual(layer.activation,tf.keras.activations.relu)
				self.assertEqual(layer.kernel_regularizer.l2,np.array(kr*(1-dr),
					dtype=np.float32))


class LensingLossFunctionsTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(LensingLossFunctionsTests, self).__init__(*args, **kwargs)
		# Set a seed to make sure that the behaviour of all the test functions
		# is consistent.
		np.random.seed(2)

	def test_mse_loss(self):
		# Test that for a variety of number of parameters and bnn types, the
		# algorithm always returns the MSE loss.
		flip_pairs = []
		for num_params in range(1,20):
			# Diagonal covariance
			loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)
			y_true = np.random.randn(num_params).reshape(1,-1)
			y_pred = np.random.randn(num_params*2).reshape(1,-1)
			mse_tensor = loss_class.mse_loss(tf.constant(y_true,dtype=tf.float32),
				tf.constant(y_pred,dtype=tf.float32))
			self.assertAlmostEqual(mse_tensor.numpy()[0],np.mean(np.square(
				y_true-y_pred[:,:num_params])),places=5)

			# Full covariance
			y_true = np.random.randn(num_params).reshape(1,-1)
			y_pred = np.random.randn(int(num_params*(num_params+1)/2)).reshape(
				1,-1)
			mse_tensor = loss_class.mse_loss(tf.constant(y_true,dtype=tf.float32),
				tf.constant(y_pred,dtype=tf.float32))
			self.assertAlmostEqual(mse_tensor.numpy()[0],np.mean(np.square(
				y_true-y_pred[:,:num_params])),places=5)

			# GMM two matrices full covariance
			y_true = np.random.randn(num_params).reshape(1,-1)
			y_pred = np.random.randn(2*(num_params + int(
				num_params*(num_params+1)/2))+1).reshape(1,-1)
			mse_tensor = loss_class.mse_loss(tf.constant(y_true,dtype=tf.float32),
				tf.constant(y_pred,dtype=tf.float32))
			self.assertAlmostEqual(mse_tensor.numpy()[0],np.mean(np.square(
				y_true-y_pred[:,:num_params])),places=5)

		# Now an explicit test that flip_pairs is working
		flip_pairs = [[1,2]]
		num_params = 5
		loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)
		y_true = np.ones((4,num_params))
		y_pred = np.ones((4,num_params))
		y_pred[:,1:3] *= -1
		mse_tensor = loss_class.mse_loss(tf.constant(y_true,dtype=tf.float32),
			tf.constant(y_pred,dtype=tf.float32))
		self.assertEqual(np.sum(mse_tensor.numpy()),0)

		# Make sure flipping other pairs does not return 0
		y_pred[:,4] *= -1
		mse_tensor = loss_class.mse_loss(tf.constant(y_true,dtype=tf.float32),
			tf.constant(y_pred,dtype=tf.float32))
		self.assertGreater(np.sum(mse_tensor.numpy()),0.1)


	def test_log_gauss_diag(self):
		# Will not be used for this test, but must be passed in.
		flip_pairs = []
		for num_params in range(1,20):
			# Pick a random true, pred, and std and make sure it agrees with the
			# scipy calculation
			loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)
			y_true = np.random.randn(num_params)
			y_pred = np.random.randn(num_params)
			std_pred = np.random.randn(num_params)
			nlp_tensor = loss_class.log_gauss_diag(tf.constant(y_true),
				tf.constant(y_pred),tf.constant(std_pred))

			# Compare to scipy function to be exact. Add 2 pi offset.
			scipy_nlp = -multivariate_normal.logpdf(y_true,y_pred,
				np.diag(np.exp(std_pred))) - np.log(2 * np.pi) * num_params/2
			self.assertAlmostEqual(nlp_tensor.numpy(),scipy_nlp)

	def test_diagonal_covariance_loss(self):
		# Test that the diagonal covariance loss gives the correct values
		flip_pairs = [[1,2],[3,4],[1,2,3,4]]
		num_params = 6
		loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)

		# Set up a couple of test function to make sure that the minimum loss
		# is taken
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,num_params))
		y_pred1 = np.ones((1,num_params)); y_pred1[:,[1,2]] = -1
		y_pred2 = np.ones((1,num_params)); y_pred2[:,[3,4]] = -1
		y_pred3 = np.ones((1,num_params)); y_pred3[:,[1,2,3,4]] = -1
		y_preds = [y_pred,y_pred1,y_pred2,y_pred3]
		std_pred = np.ones((1,num_params))

		# The correct value of the nlp
		scipy_nlp = -multivariate_normal.logpdf(y_true.flatten(),y_pred.flatten(),
			np.diag(np.exp(std_pred.flatten()))) -np.log(2 * np.pi)*num_params/2

		for yp in y_preds:
			yptf = tf.constant(np.concatenate([yp,std_pred],axis=-1),
				dtype=tf.float32)
			yttf = tf.constant(y_true,dtype=tf.float32)
			diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf)

			self.assertAlmostEqual(diag_loss.numpy(),scipy_nlp)

		# Repeat this excercise, but introducing error in prediction
		for yp in y_preds:
			yp[:,0] = 10
		scipy_nlp = -multivariate_normal.logpdf(y_true.flatten(),y_pred.flatten(),
			np.diag(np.exp(std_pred.flatten()))) -np.log(2 * np.pi)*num_params/2

		for yp in y_preds:
			yptf = tf.constant(np.concatenate([yp,std_pred],axis=-1),
				dtype=tf.float32)
			yttf = tf.constant(y_true,dtype=tf.float32)
			diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf)

			self.assertAlmostEqual(diag_loss.numpy(),scipy_nlp)

		# Confirm that when the wrong pair is flipped, it does not
		# return the same answer.
		y_pred4 = np.ones((1,num_params))
		y_pred4[:,[5,2]] = -1
		y_pred4[:,0] = 10
		yptf = tf.constant(np.concatenate([y_pred4,std_pred],axis=-1),
				dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf)

		self.assertGreater(np.abs(diag_loss.numpy()-scipy_nlp),1)

		# Make sure it is still consistent with the true nlp
		scipy_nlp = -multivariate_normal.logpdf(y_true.flatten(),
			y_pred4.flatten(),
			np.diag(np.exp(std_pred.flatten()))) -np.log(2 * np.pi)*num_params/2
		self.assertAlmostEqual(diag_loss.numpy(),scipy_nlp)

		# Finally, confirm that batching works
		yptf = tf.constant(np.concatenate(
			[np.concatenate([y_pred,std_pred],axis=-1),
			np.concatenate([y_pred1,std_pred],axis=-1)],axis=0),dtype=tf.float32)
		self.assertEqual(yptf.shape,[2,12])
		diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf).numpy()
		self.assertEqual(diag_loss.shape,(2,))
		self.assertEqual(diag_loss[0],diag_loss[1])

	def test_construct_precision_matrix(self):
		# A couple of test cases to make sure that the generalized precision
		# matrix code works as expected.

		num_params = 4
		flip_pairs = []
		loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)

		# Set up a fake l matrix with elements
		l_mat_elements = np.array([[1,2,3,4,5,6,7,8,9,10]],dtype=float)
		l_mat = np.array([[np.exp(1),0,0,0],[2,np.exp(3),0,0],[4,5,np.exp(6),0],
			[7,8,9,np.exp(10)]])
		prec_mat = np.matmul(l_mat,l_mat.T)

		# Get the tf representation of the prec matrix
		l_mat_elements_tf = tf.constant(l_mat_elements)
		p_mat_tf, diag_tf, L_mat = loss_class.construct_precision_matrix(
			l_mat_elements_tf)

		# Make sure everything matches
		np.testing.assert_almost_equal(p_mat_tf.numpy()[0],prec_mat,decimal=5)
		diag_elements = np.array([1,3,6,10])
		np.testing.assert_almost_equal(diag_tf.numpy()[0],diag_elements)
		for pi, p_mat_np in enumerate(p_mat_tf.numpy()):
			np.testing.assert_almost_equal(p_mat_np,np.dot(
				L_mat.numpy()[pi],L_mat.numpy()[pi].T))

		# Rinse and repeat for a different number of elements with batching
		num_params = 3
		flip_pairs = []
		loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)

		# Set up a fake l matrix with elements
		l_mat_elements = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6]],dtype=float)
		l_mat = np.array([[np.exp(1),0,0],[2,np.exp(3),0],[4,5,np.exp(6)]])
		prec_mat = np.matmul(l_mat,l_mat.T)

		# Get the tf representation of the prec matrix
		l_mat_elements_tf = tf.constant(l_mat_elements)
		p_mat_tf, diag_tf, _ = loss_class.construct_precision_matrix(
			l_mat_elements_tf)

		# Make sure everything matches
		for p_mat in p_mat_tf.numpy():
			np.testing.assert_almost_equal(p_mat,prec_mat)
		diag_elements = np.array([1,3,6])
		for diag in diag_tf.numpy():
			np.testing.assert_almost_equal(diag,diag_elements)

	def test_log_gauss_full(self):
		# Will not be used for this test, but must be passed in.
		flip_pairs = []
		for num_params in range(1,10):
			# Pick a random true, pred, and std and make sure it agrees with the
			# scipy calculation
			loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)
			y_true = np.random.randn(num_params)
			y_pred = np.random.randn(num_params)

			l_mat_elements_tf = tf.constant(
				np.expand_dims(np.random.randn(int(num_params*(num_params+1)/2)),
					axis=0),dtype=tf.float32)

			p_mat_tf, L_diag, _ = loss_class.construct_precision_matrix(
				l_mat_elements_tf)

			p_mat = p_mat_tf.numpy()[0]

			nlp_tensor = loss_class.log_gauss_full(tf.constant(np.expand_dims(
				y_true,axis=0),dtype=float),tf.constant(np.expand_dims(
				y_pred,axis=0),dtype=float),p_mat_tf,L_diag)

			# Compare to scipy function to be exact. Add 2 pi offset.
			scipy_nlp = (-multivariate_normal.logpdf(y_true,y_pred,np.linalg.inv(
				p_mat)) - np.log(2 * np.pi) * num_params/2)
			# The decimal error can be significant due to inverting the precision
			# matrix
			self.assertAlmostEqual(np.sum(nlp_tensor.numpy()),scipy_nlp,places=1)

	def test_full_covariance_loss(self):
		# Test that the diagonal covariance loss gives the correct values
		flip_pairs = [[1,2],[3,4],[1,2,3,4]]
		num_params = 6
		loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)

		# Set up a couple of test function to make sure that the minimum loss
		# is taken
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,num_params))
		y_pred1 = np.ones((1,num_params)); y_pred1[:,[1,2]] = -1
		y_pred2 = np.ones((1,num_params)); y_pred2[:,[3,4]] = -1
		y_pred3 = np.ones((1,num_params)); y_pred3[:,[1,2,3,4]] = -1
		y_preds = [y_pred,y_pred1,y_pred2,y_pred3]
		L_elements_len = int(num_params*(num_params+1)/2)
		# Have to keep this matrix simple so that we still get a reasonable
		# answer when we invert it for scipy check
		L_elements = np.zeros((1,L_elements_len))+1e-2

		# Get out the covariance matrix in numpy
		l_mat_elements_tf = tf.constant(L_elements,dtype=tf.float32)
		p_mat_tf, L_diag, _ = loss_class.construct_precision_matrix(
			l_mat_elements_tf)
		cov_mat = np.linalg.inv(p_mat_tf.numpy()[0])

		# The correct value of the nlp
		scipy_nlp = -multivariate_normal.logpdf(y_true.flatten(),y_pred.flatten(),
			cov_mat) -np.log(2 * np.pi)*num_params/2

		for yp in y_preds:
			yptf = tf.constant(np.concatenate([yp,L_elements],axis=-1),
				dtype=tf.float32)
			yttf = tf.constant(y_true,dtype=tf.float32)
			diag_loss = loss_class.full_covariance_loss(yttf,yptf)

			self.assertAlmostEqual(np.sum(diag_loss.numpy()),scipy_nlp,places=4)

		# Repeat this excercise, but introducing error in prediction
		for yp in y_preds:
			yp[:,0] = 10
		scipy_nlp = -multivariate_normal.logpdf(y_true.flatten(),y_pred.flatten(),
			cov_mat) -np.log(2 * np.pi)*num_params/2

		for yp in y_preds:
			yptf = tf.constant(np.concatenate([yp,L_elements],axis=-1),
				dtype=tf.float32)
			yttf = tf.constant(y_true,dtype=tf.float32)
			diag_loss = loss_class.full_covariance_loss(yttf,yptf)

			self.assertAlmostEqual(np.sum(diag_loss.numpy()),scipy_nlp,places=4)


		# Confirm that when the wrong pair is flipped, it does not
		# return the same answer.
		y_pred4 = np.ones((1,num_params)); y_pred4[:,[5,2]] = -1
		y_pred4[:,0] = 10
		yptf = tf.constant(np.concatenate([y_pred4,L_elements],axis=-1),
				dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.full_covariance_loss(yttf,yptf)

		self.assertGreater(np.abs(diag_loss.numpy()-scipy_nlp),1)

		# Make sure it is still consistent with the true nlp
		scipy_nlp = -multivariate_normal.logpdf(y_true.flatten(),
			y_pred4.flatten(),cov_mat) -np.log(2 * np.pi)*num_params/2
		self.assertAlmostEqual(np.sum(diag_loss.numpy()),scipy_nlp,places=2)

		# Finally, confirm that batching works
		yptf = tf.constant(np.concatenate(
			[np.concatenate([y_pred,L_elements],axis=-1),
			np.concatenate([y_pred1,L_elements],axis=-1)],axis=0),
			dtype=tf.float32)
		self.assertEqual(yptf.shape,[2,27])
		diag_loss = loss_class.full_covariance_loss(yttf,yptf).numpy()
		self.assertEqual(diag_loss.shape,(2,))
		self.assertEqual(diag_loss[0],diag_loss[1])

	def test_log_gauss_gm_full(self):
		# Will not be used for this test, but must be passed in.
		flip_pairs = []
		for num_params in range(1,10):
			# Pick a random true, pred, and std and make sure it agrees with the
			# scipy calculation
			loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)
			y_true = np.random.randn(num_params)
			yttf=tf.constant(np.expand_dims(y_true,axis=0),dtype=float)
			y_pred1 = np.random.randn(num_params)
			yp1tf=tf.constant(np.expand_dims(y_pred1,axis=0),dtype=float)
			y_pred2 = np.random.randn(num_params)
			yp2tf=tf.constant(np.expand_dims(y_pred2,axis=0),dtype=float)
			pi = np.random.rand()
			pitf = tf.constant(np.array([[pi]]),dtype=float)

			l_mat_elements_tf1 = tf.constant(
				np.expand_dims(np.random.randn(int(num_params*(num_params+1)/2)),
					axis=0),dtype=tf.float32)
			l_mat_elements_tf2 = tf.constant(
				np.expand_dims(np.random.randn(int(num_params*(num_params+1)/2)),
					axis=0),dtype=tf.float32)

			p_mat_tf1, L_diag1, _ = loss_class.construct_precision_matrix(
				l_mat_elements_tf1)
			p_mat_tf2, L_diag2, _ = loss_class.construct_precision_matrix(
				l_mat_elements_tf2)

			cov_mat1 = np.linalg.inv(p_mat_tf1.numpy()[0])
			cov_mat2 = np.linalg.inv(p_mat_tf2.numpy()[0])

			nlp_tensor = loss_class.log_gauss_gm_full(yttf,[yp1tf,yp2tf],
				[p_mat_tf1,p_mat_tf2],[L_diag1,L_diag2],[pitf,1-pitf])

			# Compare to scipy function to be exact. Add 2 pi offset.
			scipy_nlp1 = (multivariate_normal.logpdf(y_true,y_pred1,cov_mat1)
				+ np.log(2 * np.pi) * num_params/2 + np.log(pi))
			scipy_nlp2 = (multivariate_normal.logpdf(y_true,y_pred2,cov_mat2)
				+ np.log(2 * np.pi) * num_params/2 + np.log(1-pi))
			scipy_nlp = -np.logaddexp(scipy_nlp1,scipy_nlp2)
			# The decimal error can be significant due to inverting the precision
			# matrix
			self.assertAlmostEqual(np.sum(nlp_tensor.numpy()),scipy_nlp,places=2)

	def test_gm_full_covariance_loss(self):
		# Test that the diagonal covariance loss gives the correct values
		flip_pairs = [[1,2],[3,4],[1,2,3,4]]
		num_params = 6
		loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)

		# Set up a couple of test function to make sure that the minimum loss
		# is taken
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,num_params))
		y_pred1 = np.ones((1,num_params))
		y_pred1[:,[1,2]] = -1
		y_pred2 = np.ones((1,num_params))
		y_pred2[:,[3,4]] = -1
		y_pred3 = np.ones((1,num_params))
		y_pred3[:,[1,2,3,4]] = -1
		y_preds = [y_pred,y_pred1,y_pred2,y_pred3]
		L_elements_len = int(num_params*(num_params+1)/2)
		# Have to keep this matrix simple so that we still get a reasonable
		# answer when we invert it for scipy check
		L_elements = np.zeros((1,L_elements_len))+1e-2
		pi_logit = 2
		pi = np.exp(pi_logit)/(np.exp(pi_logit)+1)
		pi_arr = np.array([[pi_logit]])

		# Get out the covariance matrix in numpy
		l_mat_elements_tf = tf.constant(L_elements,dtype=tf.float32)
		p_mat_tf, L_diag, _ = loss_class.construct_precision_matrix(
			l_mat_elements_tf)
		cov_mat = np.linalg.inv(p_mat_tf.numpy()[0])

		scipy_nlp1 = (multivariate_normal.logpdf(y_true[0],y_pred[0],cov_mat)
			+ np.log(2 * np.pi) * num_params/2 + np.log(pi))
		scipy_nlp2 = (multivariate_normal.logpdf(y_true[0],y_pred[0],cov_mat)
			+ np.log(2 * np.pi) * num_params/2 + np.log(1-pi))
		scipy_nlp = -np.logaddexp(scipy_nlp1,scipy_nlp2)

		for yp1 in y_preds:
			for yp2 in y_preds:
				yptf = tf.constant(np.concatenate([yp1,L_elements,yp2,L_elements,
					pi_arr],axis=-1),dtype=tf.float32)
				yttf = tf.constant(y_true,dtype=tf.float32)
				diag_loss = loss_class.gm_full_covariance_loss(yttf,yptf)

				self.assertAlmostEqual(np.sum(diag_loss.numpy()),
					scipy_nlp,places=4)

		# Repeat this excercise, but introducing error in prediction
		for yp in y_preds:
			yp[:,0] = 10
		scipy_nlp1 = (multivariate_normal.logpdf(y_true[0],y_pred[0],cov_mat)
			+ np.log(2 * np.pi) * num_params/2 + np.log(pi))
		scipy_nlp2 = (multivariate_normal.logpdf(y_true[0],y_pred[0],cov_mat)
			+ np.log(2 * np.pi) * num_params/2 + np.log(1-pi))
		scipy_nlp = -np.logaddexp(scipy_nlp1,scipy_nlp2)

		for yp1 in y_preds:
			for yp2 in y_preds:
				yptf = tf.constant(np.concatenate([yp1,L_elements,yp2,L_elements,
					pi_arr],axis=-1),dtype=tf.float32)
				yttf = tf.constant(y_true,dtype=tf.float32)
				diag_loss = loss_class.gm_full_covariance_loss(yttf,yptf)

				self.assertAlmostEqual(np.sum(diag_loss.numpy()),
					scipy_nlp,places=4)


		# Confirm that when the wrong pair is flipped, it does not
		# return the same answer.
		y_pred4 = np.ones((1,num_params)); y_pred4[:,[5,2]] = -1
		y_pred4[:,0] = 10
		yptf = tf.constant(np.concatenate([y_pred4,L_elements,y_pred,L_elements,
			pi_arr],axis=-1),dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.gm_full_covariance_loss(yttf,yptf)

		self.assertGreater(np.abs(diag_loss.numpy()-scipy_nlp),0.1)


		# Finally, confirm that batching works
		single_batch1 = np.concatenate([y_pred2,L_elements,y_pred,L_elements,
			pi_arr],axis=-1)
		single_batch2 = np.concatenate([y_pred3,L_elements,y_pred,L_elements,
			pi_arr],axis=-1)
		yptf = tf.constant(np.concatenate([single_batch1,single_batch2],axis=0),
			dtype=tf.float32)
		self.assertEqual(yptf.shape,[2,55])
		diag_loss = loss_class.gm_full_covariance_loss(yttf,yptf).numpy()
		self.assertEqual(diag_loss.shape,(2,))
		self.assertEqual(diag_loss[0],diag_loss[1])
		self.assertAlmostEqual(diag_loss[0],scipy_nlp,places=4)

	def test_p_value(self):
		# Test that the p_value function correctly return the mean p_value of the
		# function.
		# Initialize a model an test the the function returns the desired value.
		image_size = (100,100,1)
		num_params = 8
		model = bnn_alexnet.concrete_alexnet(image_size, num_params,
			kernel_regularizer=1e-6,dropout_regularizer=1e-5)
		p_fake_loss = bnn_alexnet.p_value(model)
		self.assertAlmostEqual(p_fake_loss(None,None).numpy(),0.1)

		model = bnn_alexnet.concrete_alexnet(image_size, num_params,
			kernel_regularizer=1e-6,dropout_regularizer=1e-5,
			init_min=0.3,init_max=0.3)
		p_fake_loss = bnn_alexnet.p_value(model)
		self.assertAlmostEqual(p_fake_loss(None,None).numpy(),0.3)

