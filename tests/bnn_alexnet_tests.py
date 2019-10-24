# TODO: Change the imports once this is a package!!
import unittest
import sys
import numpy as np
import tensorflow as tf
sys.path.append("../")
import bnn_alexnet
from scipy.stats import multivariate_normal

class BNNTests(unittest.TestCase):

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
			weight_regularizer=1e-6,dropout_regularizer=1e-5)
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
				self.assertEqual(layer.layer.activation,tf.keras.activations.relu)
			l_i += 1

class LensingLossFunctionsTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(LensingLossFunctionsTests, self).__init__(*args, **kwargs)
		# Set a seed to make sure that the behaviour of all the test functions
		# is consistent.
		np.random.seed(2)

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
			nll_tensor = loss_class.log_gauss_diag(tf.constant(y_true),
				tf.constant(y_pred),tf.constant(std_pred))

			# Compare to scipy function to be exact. Add 2 pi offset.
			scipy_nll = -multivariate_normal.logpdf(y_true,y_pred,
				np.diag(np.exp(std_pred))) - np.log(2 * np.pi) * num_params/2
			self.assertAlmostEqual(nll_tensor.numpy(),scipy_nll)

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

		# The correct value of the nll
		scipy_nll = -multivariate_normal.logpdf(y_true.flatten(),y_pred.flatten(),
			np.diag(np.exp(std_pred.flatten()))) -np.log(2 * np.pi)*num_params/2

		for yp in y_preds:
			yptf = tf.constant(np.concatenate([yp,std_pred],axis=-1),
				dtype=tf.float32)
			yttf = tf.constant(y_true,dtype=tf.float32)
			diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf)

			self.assertAlmostEqual(diag_loss.numpy(),scipy_nll)

		# Repeat this excercise, but introducing error in prediction
		for yp in y_preds:
			yp[:,0] = 10
		scipy_nll = -multivariate_normal.logpdf(y_true.flatten(),y_pred.flatten(),
			np.diag(np.exp(std_pred.flatten()))) -np.log(2 * np.pi)*num_params/2

		for yp in y_preds:
			yptf = tf.constant(np.concatenate([yp,std_pred],axis=-1),
				dtype=tf.float32)
			yttf = tf.constant(y_true,dtype=tf.float32)
			diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf)

			self.assertAlmostEqual(diag_loss.numpy(),scipy_nll)


		# Finally, confirm that when the wrong pair is flipped, it does not
		# return the same answer.
		y_pred4 = np.ones((1,num_params)); y_pred4[:,[5,2]] = -1
		y_pred4[:,0] = 10
		yptf = tf.constant(np.concatenate([y_pred4,std_pred],axis=-1),
				dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.diagonal_covariance_loss(yttf,yptf)

		self.assertGreater(np.abs(diag_loss.numpy()-scipy_nll),1)
		
		# Make sure it is still consistent with the true nll
		scipy_nll = -multivariate_normal.logpdf(y_true.flatten(),
			y_pred4.flatten(),
			np.diag(np.exp(std_pred.flatten()))) -np.log(2 * np.pi)*num_params/2
		self.assertAlmostEqual(diag_loss.numpy(),scipy_nll)

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
		p_mat_tf, diag_tf = loss_class.construct_precision_matrix(
			l_mat_elements_tf)

		# Make sure everything matches
		self.assertAlmostEqual(np.sum(np.abs(p_mat_tf.numpy()-prec_mat)),0)
		diag_elements = np.array([1,3,6,10])
		self.assertAlmostEqual(np.sum(np.abs(diag_tf.numpy()-diag_elements)),0)

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
		p_mat_tf, diag_tf = loss_class.construct_precision_matrix(
			l_mat_elements_tf)

		# Make sure everything matches
		for p_mat in p_mat_tf.numpy():
			self.assertAlmostEqual(np.sum(np.abs(p_mat-prec_mat)),0)
		diag_elements = np.array([1,3,6])
		for diag in diag_tf.numpy():
			self.assertAlmostEqual(np.sum(np.abs(diag-diag_elements)),0)


