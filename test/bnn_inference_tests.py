import unittest, os, json
# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
from ovejero import bnn_inference, data_tools
import numpy as np
import pandas as pd

class BNNInferenceTest(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(BNNInferenceTest, self).__init__(*args, **kwargs)
		# Open up the config file.
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		with open(self.root_path+'test.json','r') as json_f:
			self.cfg = json.load(json_f)
		self.final_params = self.cfg['training_params']['final_params']
		self.num_params = len(self.final_params)
		self.batch_size = self.cfg['training_params']['batch_size']
		self.normalized_param_path = self.root_path + 'normed_metadata.csv'
		self.normalization_constants_path = self.root_path + 'norm.csv'
		self.lens_params_path = self.root_path + 'metadata.csv'
		self.lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.cfg['dataset_params']['normalization_constants_path'] = 'norm.csv'
		self.cfg['training_params']['final_params'] = self.lens_params
		self.cfg['training_params']['bnn_type'] = 'diag'
		self.tf_record_path = self.root_path+self.cfg['validation_params'][
			'tf_record_path']
		self.infer_class = bnn_inference.InferenceClass(self.cfg)

	def test_fix_flip_pairs(self):
		# Check that fix_flip_pairs always selects the best possible configuration
		# to return.
		

		# Get the set of all flip pairs we want to check
		flip_pairs = self.cfg['training_params']['flip_pairs']
		flip_set = set()
		for flip_pair in flip_pairs:
			flip_set.update(flip_pair)
		
		y_test = np.ones((self.batch_size,self.num_params))
		predict_samps = np.ones((10,self.batch_size,self.num_params))

		pi = 0
		for flip_index in flip_set:
			predict_samps[pi,:,flip_index] = -1

		# Flip pairs of points.
		self.infer_class.fix_flip_pairs(predict_samps,y_test)

		self.assertEqual(np.sum(np.abs(predict_samps-y_test)),0)

		dont_flip_set = set(range(self.num_params))
		dont_flip_set=dont_flip_set.difference(flip_set)

		pi = 0
		for flip_index in dont_flip_set:
			predict_samps[pi,:,flip_index] = -1

		# Flip pairs of points.
		self.infer_class.fix_flip_pairs(predict_samps,y_test)

		self.assertEqual(np.sum(np.abs(predict_samps-y_test)),
			2*self.batch_size*len(dont_flip_set))


	def test_undo_param_norm(self):
				# Test if normalizing the lens parameters works correctly.
		train_or_test='train'
		data_tools.normalize_lens_parameters(self.lens_params,
			self.lens_params_path,self.normalized_param_path,
			self.normalization_constants_path,train_or_test=train_or_test)

		lens_params_csv = pd.read_csv(self.lens_params_path, index_col=None)
		norm_params_csv = pd.read_csv(self.normalized_param_path, index_col=None)

		# Pull lens parameters out of the csv files.
		lens_params_numpy = []
		norms_params_numpy = []
		for lens_param in self.lens_params:
			lens_params_numpy.append(lens_params_csv[lens_param])
			norms_params_numpy.append(norm_params_csv[lens_param])
		lens_params_numpy = np.array(lens_params_numpy).T
		norms_params_numpy = np.array(norms_params_numpy).T
		predict_samps = np.tile(norms_params_numpy,(3,1,1))

		# Try to denormalize everything
		self.infer_class.undo_param_norm(predict_samps,norms_params_numpy)

		self.assertAlmostEqual(np.mean(np.abs(norms_params_numpy-
			lens_params_numpy)),0)
		self.assertAlmostEqual(np.mean(np.abs(predict_samps-
			lens_params_numpy)),0)

		# Clean up the file now that we're done
		os.remove(self.normalized_param_path)	
		os.remove(self.normalization_constants_path)

	def test_gen_samples(self):

		# First we have to make a fake model whose statistics are very well
		# defined.

		class ToyModel():
			def __init__(self,mean,covariance,batch_size):
				# We want to make sure our performance is consistent for a
				# test
				np.random.seed(4)
				self.mean=mean
				self.covariance = covariance
				self.batch_size = batch_size
			def predict(self,image):
				# We won't actually be using the image. We just want it for
				# testing.
				return np.concatenate([np.random.multivariate_normal(self.mean,
					self.covariance,self.batch_size),np.ones((self.batch_size,
						len(self.mean)))-1000],axis=-1)

		# Start with a simple covariance matrix example.
		mean = np.ones(self.num_params)*2
		covariance = np.diag(np.ones(self.num_params))
		diag_model = ToyModel(mean,covariance,self.batch_size)

		# We don't want any flipping going on
		self.infer_class.flip_mat_list = [np.diag(np.ones(self.num_params))]

		# Create tf record. This won't be used, but it has to be there for
		# the function to be able to pull some images.
		# Make fake norms data
		fake_norms = {}
		for lens_param in self.lens_params:
			fake_norms[lens_param] = np.array([0.0,1.0])
		fake_norms = pd.DataFrame(data=fake_norms)
		fake_norms.to_csv(self.normalization_constants_path,index=False)
		data_tools.generate_tf_record(self.root_path,self.lens_params,
			self.lens_params_path,self.tf_record_path)

		# Replace the real model with our fake model and generate samples
		self.infer_class.model = diag_model
		self.infer_class.gen_samples(10000)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-np.diag(
			covariance))),0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-covariance)),
			0,places=1)

		# Repeat this process again with a new covariance matrix and means
		mean = np.random.rand(self.num_params)
		covariance = np.random.rand(self.num_params,self.num_params)
		# Make sure covariance is positive semidefinite
		covariance = np.dot(covariance,covariance.T)
		diag_model = ToyModel(mean,covariance,self.batch_size)
		self.infer_class.model = diag_model
		self.infer_class.gen_samples(10000)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-np.sqrt(
			np.diag(covariance)))),0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-covariance)),
			0,places=1)

		# Make sure our test probes things well. 
		wrong_mean = np.random.randn(self.num_params)
		wrong_covariance = np.random.rand(self.num_params,self.num_params)
		# Make sure covariance is positive semidefinite
		wrong_covariance = np.dot(wrong_covariance,wrong_covariance.T)
		diag_model = ToyModel(wrong_mean,wrong_covariance,self.batch_size)
		self.infer_class.model = diag_model
		self.infer_class.gen_samples(10000)
		self.assertGreater(np.mean(np.abs(self.infer_class.y_pred-mean)),0.05)
		self.assertGreater(np.mean(np.abs(self.infer_class.y_std-np.sqrt(
			np.diag(covariance)))),0.05)
		self.assertGreater(np.mean(np.abs(self.infer_class.y_cov-covariance)),
			0.05)

		# For now inference for more complex model is not implemented. Make
		# sure an error is raised.
		err_cfg = self.cfg.copy()
		err_cfg['training_params']['bnn_type'] = 'full'
		with self.assertRaises(NotImplementedError):
			err_class = bnn_inference.InferenceClass(err_cfg)
			err_class.gen_samples(10)

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.tf_record_path)








