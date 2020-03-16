import unittest, os, json
# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
from ovejero import bnn_inference, data_tools, bnn_alexnet
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class BNNInferenceTest(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(BNNInferenceTest, self).__init__(*args, **kwargs)
		# Open up the config file.
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		with open(self.root_path+'test.json','r') as json_f:
			self.cfg = json.load(json_f)
		self.batch_size = self.cfg['training_params']['batch_size']
		self.normalized_param_path = self.root_path + 'normed_metadata.csv'
		self.normalization_constants_path = self.root_path + 'norm.csv'
		self.lens_params_path = self.root_path + 'metadata.csv'
		self.lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.num_params = len(self.lens_params)
		self.cfg['dataset_params']['normalization_constants_path'] = 'norm.csv'
		self.cfg['training_params']['final_params'] = self.lens_params
		self.cfg['training_params']['bnn_type'] = 'diag'
		self.tf_record_path = self.root_path+self.cfg['validation_params'][
			'tf_record_path']
		self.infer_class = bnn_inference.InferenceClass(self.cfg)
		np.random.seed(2)
		tf.random.set_seed(2)

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
		# TODO: write a good test for al_samps!
		al_samps = np.ones((3,3,self.num_params,self.num_params))

		# Try to denormalize everything
		self.infer_class.undo_param_norm(predict_samps,norms_params_numpy,
			al_samps)

		self.assertAlmostEqual(np.mean(np.abs(norms_params_numpy-
			lens_params_numpy)),0)
		self.assertAlmostEqual(np.mean(np.abs(predict_samps-
			lens_params_numpy)),0)

		# Clean up the file now that we're done
		os.remove(self.normalized_param_path)	
		os.remove(self.normalization_constants_path)

	def test_gen_samples_diag(self):

		# First we have to make a fake model whose statistics are very well
		# defined.

		class ToyModel():
			def __init__(self,mean,covariance,batch_size,al_std):
				# We want to make sure our performance is consistent for a
				# test
				np.random.seed(4)
				self.mean=mean
				self.covariance = covariance
				self.batch_size = batch_size
				self.al_std = al_std
			def predict(self,image):
				# We won't actually be using the image. We just want it for
				# testing.
				return tf.constant(np.concatenate([np.random.multivariate_normal(
					self.mean,self.covariance,self.batch_size),np.zeros((
					self.batch_size,len(self.mean)))+self.al_std],axis=-1),
					tf.float32)

		# Start with a simple covariance matrix example.
		mean = np.ones(self.num_params)*2
		covariance = np.diag(np.ones(self.num_params))
		al_std = -1000
		diag_model = ToyModel(mean,covariance,self.batch_size,al_std)

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
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov)),0)

		# Repeat this process again with a new covariance matrix and means
		mean = np.random.rand(self.num_params)
		covariance = np.random.rand(self.num_params,self.num_params)
		al_std = 0
		# Make sure covariance is positive semidefinite
		covariance = np.dot(covariance,covariance.T)
		diag_model = ToyModel(mean,covariance,self.batch_size,al_std)
		self.infer_class.model = diag_model
		self.infer_class.gen_samples(10000)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean)),0,
			places=1)
		# Covariance is the sum of two random variables
		covariance = covariance+np.eye(self.num_params)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-np.sqrt(
			np.diag(covariance)))),0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-covariance)),
			0,places=1)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov-
			np.eye(self.num_params))),0)

		# Make sure our test probes things well. 
		wrong_mean = np.random.randn(self.num_params)
		wrong_covariance = np.random.rand(self.num_params,self.num_params)
		al_std = -1000
		# Make sure covariance is positive semidefinite
		wrong_covariance = np.dot(wrong_covariance,wrong_covariance.T)
		diag_model = ToyModel(wrong_mean,wrong_covariance,self.batch_size,
			al_std)
		self.infer_class.model = diag_model
		self.infer_class.gen_samples(10000)
		self.assertGreater(np.mean(np.abs(self.infer_class.y_pred-mean)),0.05)
		self.assertGreater(np.mean(np.abs(self.infer_class.y_std-np.sqrt(
			np.diag(covariance)))),0.05)
		self.assertGreater(np.mean(np.abs(self.infer_class.y_cov-covariance)),
			0.05)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov)),0)

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.tf_record_path)

	def test_gen_samples_full(self):

		# First we have to make a fake model whose statistics are very well
		# defined.

		class ToyModel():
			def __init__(self,mean,covariance,batch_size,L_elements):
				# We want to make sure our performance is consistent for a
				# test
				np.random.seed(6)
				self.mean=mean
				self.num_params = len(mean)
				self.covariance = covariance
				self.batch_size = batch_size
				self.L_elements = L_elements
				self.L_elements_len = int(self.num_params*(self.num_params+1)/2)
			def predict(self,image):
				# We won't actually be using the image. We just want it for
				# testing.
				return tf.constant(np.concatenate([np.random.multivariate_normal(
					self.mean,self.covariance,self.batch_size),np.zeros((
					self.batch_size,self.L_elements_len))+self.L_elements],
					axis=-1),tf.float32)

		# Start with a simple covariance matrix example.
		mean = np.ones(self.num_params)*2
		covariance = np.diag(np.ones(self.num_params)*0.000001)
		L_elements = np.array([np.log(1)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		full_model = ToyModel(mean,covariance,self.batch_size,L_elements)

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
		self.infer_class.model = full_model
		self.infer_class.bnn_type = 'full'
		self.infer_class.gen_samples(1000)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean)),
			0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-1)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-np.eye(
			self.num_params))),0,places=1)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov-np.eye(
			self.num_params))),0)

		mean = np.zeros(self.num_params)
		loss_class = bnn_alexnet.LensingLossFunctions([],self.num_params)
		L_elements = np.ones((1,len(L_elements)))*0.2
		full_model = ToyModel(mean,covariance,self.batch_size,L_elements)
		self.infer_class.model = full_model
		self.infer_class.gen_samples(1000)

		# Calculate the corresponding covariance matrix
		prec_mat, _ = loss_class.construct_precision_matrix(
					tf.constant(L_elements))
		prec_mat = prec_mat.numpy()[0]
		cov_mat = np.linalg.inv(prec_mat)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-np.sqrt(
			np.diag(cov_mat)))),0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-cov_mat)),
			0,places=1)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov-cov_mat)),
			0)

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.tf_record_path)

	def test_gen_samples_gmm(self):

		# First we have to make a fake model whose statistics are very well
		# defined.

		class ToyModel():
			def __init__(self,mean1,covariance1,mean2,covariance2,batch_size,
				L_elements1,L_elements2,pi_logit):
				# We want to make sure our performance is consistent for a
				# test
				np.random.seed(6)
				self.mean1=mean1
				self.mean2=mean2
				self.covariance1=covariance1
				self.covariance2=covariance2
				self.num_params = len(mean1)
				self.batch_size = batch_size
				self.L_elements1 = L_elements1
				self.L_elements2 = L_elements2
				self.pi_logit = pi_logit
				self.L_elements_len = int(self.num_params*(self.num_params+1)/2)
			def predict(self,image):
				# We won't actually be using the image. We just want it for
				# testing.
				return tf.constant(np.concatenate([
					np.random.multivariate_normal(self.mean1,self.covariance1,
						self.batch_size),
					np.zeros((
						self.batch_size,self.L_elements_len))+self.L_elements1,
					np.random.multivariate_normal(self.mean2,self.covariance2,
					self.batch_size),
					np.zeros((
						self.batch_size,self.L_elements_len))+self.L_elements2,
					np.zeros(
						(self.batch_size,1))+self.pi_logit],axis=-1),tf.float32)

		# Start with a simple covariance matrix example where both gmms
		# are the same. This is just checking the base case.
		mean1 = np.ones(self.num_params)*2
		mean2 = np.ones(self.num_params)*2
		covariance1 = np.diag(np.ones(self.num_params)*0.000001)
		covariance2 = np.diag(np.ones(self.num_params)*0.000001)
		L_elements1 = np.array([np.log(1)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		L_elements2 = np.array([np.log(1)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		pi_logit = 0
		gmm_model = ToyModel(mean1,covariance1,mean2,covariance2,
			self.batch_size,L_elements1,L_elements2,pi_logit)

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
		self.infer_class.model = gmm_model
		self.infer_class.bnn_type = 'gmm'
		self.infer_class.gen_samples(1000)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean1)),
			0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-1)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-np.eye(
			self.num_params))),0,places=1)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov-np.eye(
			self.num_params))),0)

		# Now we try and example where all the samples should be drawn from one
		# of the two gmms because of the logit.
		mean1 = np.ones(self.num_params)*2
		mean2 = np.ones(self.num_params)*200
		covariance1 = np.diag(np.ones(self.num_params)*0.000001)
		covariance2 = np.diag(np.ones(self.num_params)*0.000001)
		L_elements1 = np.array([np.log(1)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		L_elements2 = np.array([np.log(10)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		pi_logit = np.log(0.99999)-np.log(0.00001)
		gmm_model = ToyModel(mean1,covariance1,mean2,covariance2,
			self.batch_size,L_elements1,L_elements2,pi_logit)
		self.infer_class.model = gmm_model
		self.infer_class.gen_samples(1000)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean1)),
			0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-1)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-np.eye(
			self.num_params))),0,places=1)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov-np.eye(
			self.num_params))),0)

		# Same as before but now fousing on the second gmm
		mean1 = np.ones(self.num_params)*2
		mean2 = np.ones(self.num_params)*4
		covariance1 = np.diag(np.ones(self.num_params)*0.000001)
		covariance2 = np.diag(np.ones(self.num_params)*0.000001)
		L_elements1 = np.array([np.log(10)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		L_elements2 = np.array([np.log(1)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		pi_logit = np.log(0.0001)-np.log(0.9999)
		gmm_model = ToyModel(mean1,covariance1,mean2,covariance2,
			self.batch_size,L_elements1,L_elements2,pi_logit)
		self.infer_class.model = gmm_model
		self.infer_class.gen_samples(1000)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-mean2)),
			0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-1)),0,
			places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_cov-np.eye(
			self.num_params))),0,places=1)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.al_cov-np.eye(
			self.num_params))),0,places=4)

		# Now test that it takes a combination of them correctly
		mean1 = np.ones(self.num_params)*2
		mean2 = np.ones(self.num_params)*6
		covariance1 = np.diag(np.ones(self.num_params)*0.000001)
		covariance2 = np.diag(np.ones(self.num_params)*0.000001)
		L_elements1 = np.array([np.log(10)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		L_elements2 = np.array([np.log(1)]*self.num_params+[0]*int(
			self.num_params*(self.num_params-1)/2))
		pi_logit = 0
		gmm_model = ToyModel(mean1,covariance1,mean2,covariance2,
			self.batch_size,L_elements1,L_elements2,pi_logit)
		self.infer_class.model = gmm_model
		self.infer_class.gen_samples(2000)

		# Make sure these samples follow the required statistics.
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_pred-4)),
			0,places=1)
		self.assertAlmostEqual(np.mean(np.abs(self.infer_class.y_std-np.sqrt(5))),
			0,places=0)
		self.assertTupleEqual(self.infer_class.al_cov.shape,(self.batch_size,
			self.num_params,self.num_params))

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.tf_record_path)

	def test_gen_samples_save(self):

		# First we have to make a fake model whose statistics are very well
		# defined.
		class ToyModel():
			def __init__(self,mean,covariance,batch_size,al_std):
				# We want to make sure our performance is consistent for a
				# test
				np.random.seed(4)
				self.mean=mean
				self.covariance = covariance
				self.batch_size = batch_size
				self.al_std = al_std
			def predict(self,image):
				# We won't actually be using the image. We just want it for
				# testing.
				return tf.constant(np.concatenate([np.random.multivariate_normal(
					self.mean,self.covariance,self.batch_size),np.zeros((
					self.batch_size,len(self.mean)))+self.al_std],axis=-1),
					tf.float32)

		# Start with a simple covariance matrix example.
		mean = np.ones(self.num_params)*2
		covariance = np.diag(np.ones(self.num_params))
		al_std = -1000
		diag_model = ToyModel(mean,covariance,self.batch_size,al_std)

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
		# Provide a save path to then check that we get the same data
		save_path = self.root_path + 'test_gen_samps/'
		self.infer_class.gen_samples(10000,save_path)

		pred_1 = np.copy(self.infer_class.predict_samps)
		# Generate again and make sure they are equivalent
		self.infer_class.gen_samples(10000,save_path)

		np.testing.assert_almost_equal(pred_1,self.infer_class.predict_samps)

		# Test that none of the plotting routines break
		self.infer_class.gen_coverage_plots(block=False)
		plt.close()
		self.infer_class.report_stats()
		self.infer_class.plot_posterior_contours(1,block=False)
		plt.close()
		plt.close()
		self.infer_class.comp_al_ep_unc(block=False)
		plt.close()
		self.infer_class.comp_al_ep_unc(block=False,norm_diagonal=False)
		plt.close()
		self.infer_class.plot_calibration(block=False)
		plt.close()

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.tf_record_path)
		os.remove(save_path+'pred.npy')
		os.remove(save_path+'al_samp.npy')
		os.remove(save_path+'images.npy')
		os.remove(save_path+'y_test.npy')
		os.rmdir(save_path)

	def test_calc_p_dlt(self):
		# Test that the calc_p_dlt returns the correct percentages for some
		# toy examples

		# Check a simple case
		size = int(1e6)
		self.infer_class.predict_samps = np.random.normal(size=size*2).reshape(
			(size//10,10,2))
		self.infer_class.predict_samps[:,:,1]=0
		self.infer_class.y_pred = np.mean(self.infer_class.predict_samps,axis=0)
		self.infer_class.y_test = np.array([[1,2,3,4,5,6,7,8,9,10],
			[0,0,0,0,0,0,0,0,0,0]],dtype=np.float32).T

		self.infer_class.calc_p_dlt(cov_emp=np.diag(np.ones(2)))
		percentages = [0.682689,0.954499,0.997300,0.999936,0.999999]+[1.0]*5
		for p_i in range(len(percentages)):
			self.assertAlmostEqual(percentages[p_i],self.infer_class.p_dlt[p_i],
				places=2)

		# Shift the mean
		size = int(1e6)
		self.infer_class.predict_samps = np.random.normal(loc=2,
			size=size*2).reshape((size//10,10,2))
		self.infer_class.predict_samps[:,:,1]=0
		self.infer_class.y_pred = np.mean(self.infer_class.predict_samps,axis=0)
		self.infer_class.y_test = np.array([[1,2,3,4,5,6,7,8,9,10],
			[0,0,0,0,0,0,0,0,0,0]],dtype=np.float32).T
		self.infer_class.calc_p_dlt(cov_emp=np.diag(np.ones(2)))
		percentages = [0.682689,0,0.682689,0.954499,0.997300,0.999936]+[1.0]*4
		for p_i in range(len(percentages)):
			self.assertAlmostEqual(percentages[p_i],self.infer_class.p_dlt[p_i],
				places=2)

		# Expand to higher dimensions
		size = int(1e6)
		self.infer_class.predict_samps = np.random.normal(loc=0,
			size=size*2).reshape((size//10,10,2))
		self.infer_class.predict_samps /= np.sqrt(np.sum(np.square(
			self.infer_class.predict_samps),axis=-1,keepdims=True))
		self.infer_class.predict_samps *= np.random.random(size=size).reshape((
			size//10,10,1))*5
		self.infer_class.y_pred = np.mean(self.infer_class.predict_samps,axis=0)
		self.infer_class.y_test = np.array([[1,2,3,4,5,6,7,8,9,10],[0]*10]).T
		self.infer_class.calc_p_dlt(cov_emp=np.diag(np.ones(2)))
		percentages = [1/5,2/5,3/5,4/5,1,1]+[1.0]*4
		for p_i in range(len(percentages)):
			self.assertAlmostEqual(percentages[p_i],self.infer_class.p_dlt[p_i],
				places=2)

		# Expand to higher dimensions
		size = int(1e6)
		self.infer_class.predict_samps = np.random.normal(loc=0,
			size=size*2).reshape((size//2,2,2))*5
		self.infer_class.predict_samps[:,:,1]=0
		self.infer_class.y_pred = np.mean(self.infer_class.predict_samps,axis=0)
		self.infer_class.y_test = np.array([[0,np.sqrt(2)],[0]*2]).T
		self.infer_class.calc_p_dlt()
		percentages = [0,0.223356]
		for p_i in range(len(percentages)):
			self.assertAlmostEqual(percentages[p_i],self.infer_class.p_dlt[p_i],
				places=2)





