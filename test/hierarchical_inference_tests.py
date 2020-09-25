import unittest, os, json, gc
from ovejero import hierarchical_inference, model_trainer
from baobab import configs
from lenstronomy.Util.param_util import ellipticity2phi_q
from baobab import distributions
import numpy as np
from scipy import stats, special
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


class HierarchicalnferenceTest(unittest.TestCase):

	def setUp(self):
		# Open up the config file.
		np.random.seed(2)
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		self.cfg = configs.BaobabConfig.from_file(self.root_path +
			'test_baobab_cfg.py')
		self.cfg_pr = hierarchical_inference.load_prior_config(self.root_path +
			'test_ovejero_cfg_prior.py')
		self.cfg_cov = hierarchical_inference.load_prior_config(self.root_path +
			'test_emp_cfg_prior.py')
		self.lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.lens_params_cov = ['external_shear_gamma_ext',
			'external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_q','lens_mass_phi',
			'lens_mass_gamma','lens_mass_theta_E']
		self.eval_dict = hierarchical_inference.build_eval_dict(self.cfg,
			self.lens_params)
		self.eval_dict_prior = hierarchical_inference.build_eval_dict(
			self.cfg_pr,self.lens_params,baobab_config=False)
		self.eval_dict_cov = hierarchical_inference.build_eval_dict(
			self.cfg_cov,self.lens_params_cov,baobab_config=False)

	def tearDown(self):
		# Clean up for memory
		self.cfg = None
		self.cfg_pr = None
		self.cfg_cov = None
		self.eval_dict = None
		self.eval_dict_prior = None
		self.eval_dict_cov = None

	def test_build_eval_dict(self):
		# Check that the eval dictionary is built correctly for a test config.
		n_lens_param_p_params = [2,5,2,2,4,4,2,2]

		# First we test the case without priors.
		self.assertEqual(self.eval_dict['hyp_len'],23)
		self.assertListEqual(list(self.eval_dict['hyp_values']),[-2.73,1.05,0.0,
			0.5*np.pi,10.0,-0.5*np.pi,0.5*np.pi,0.0,0.102,0.0,0.102,4.0,4.0,
			-0.55,0.55,4.0,4.0,-0.55,0.55,0.7,0.1,0.0,0.1])
		self.assertListEqual(self.eval_dict['hyp_names'],[
			'external_shear_gamma_ext:mu','external_shear_gamma_ext:sigma',
			'external_shear_psi_ext:mu','external_shear_psi_ext:alpha',
			'external_shear_psi_ext:p','external_shear_psi_ext:lower',
			'external_shear_psi_ext:upper','lens_mass_center_x:mu',
			'lens_mass_center_x:sigma','lens_mass_center_y:mu',
			'lens_mass_center_y:sigma','lens_mass_e1:a',
			'lens_mass_e1:b','lens_mass_e1:lower','lens_mass_e1:upper',
			'lens_mass_e2:a','lens_mass_e2:b','lens_mass_e2:lower',
			'lens_mass_e2:upper','lens_mass_gamma:mu',
			'lens_mass_gamma:sigma','lens_mass_theta_E:mu',
			'lens_mass_theta_E:sigma'])

		total = 0
		for li,lens_param in enumerate(self.lens_params):
			n_p = n_lens_param_p_params[li]
			self.assertListEqual(list(self.eval_dict[lens_param]['hyp_ind']),
				list(range(total,total+n_p)))
			self.assertFalse(self.eval_dict[lens_param]['eval_fn_kwargs'])
			if n_p == 2:
				self.assertTrue((self.eval_dict[lens_param]['eval_fn'] is
					distributions.eval_normal_logpdf_approx) or (
					self.eval_dict[lens_param]['eval_fn'] is
					distributions.eval_lognormal_logpdf_approx))
			if n_p == 4:
				self.assertTrue(self.eval_dict[lens_param]['eval_fn'] is
					distributions.eval_beta_logpdf_approx)
			if n_p == 5:
				self.assertTrue(self.eval_dict[lens_param]['eval_fn'] is
					distributions.eval_generalized_normal_logpdf_approx)
			total += n_p

		# Now we test the case with priors.
		self.assertEqual(self.eval_dict_prior['hyp_len'],14)
		self.assertListEqual(list(self.eval_dict_prior['hyp_init']),[-2.73,1.05,
			0.0,0.102,0.0,0.102,0.0,0.1,0.0,0.1,0.7,0.1,0.0,0.1])
		self.assertListEqual(list(self.eval_dict_prior['hyp_sigma']),[0.5,0.05,
			0.2,0.03,0.2,0.03,0.3,0.01,0.3,0.01,0.3,0.01,0.3,0.01])
		self.assertListEqual(self.eval_dict_prior['hyp_names'],[
			'external_shear_gamma_ext:mu','external_shear_gamma_ext:sigma',
			'lens_mass_center_x:mu','lens_mass_center_x:sigma',
			'lens_mass_center_y:mu','lens_mass_center_y:sigma','lens_mass_e1:mu',
			'lens_mass_e1:sigma','lens_mass_e2:mu','lens_mass_e2:sigma',
			'lens_mass_gamma:mu','lens_mass_gamma:sigma','lens_mass_theta_E:mu',
			'lens_mass_theta_E:sigma'])

		n_lens_param_p_params = [2,0,2,2,2,2,2,2]

		total = 0
		for li, lens_param in enumerate(self.lens_params):
			n_p = n_lens_param_p_params[li]
			if n_p==0:
				self.assertFalse(list(self.eval_dict_prior[lens_param]['hyp_ind']
					))
				self.assertTrue((self.eval_dict_prior[lens_param]['eval_fn'] is
					distributions.eval_uniform_logpdf_approx))
			else:
				self.assertListEqual(list(self.eval_dict_prior[lens_param]['hyp_ind']),
					list(range(total,total+n_p)))
				self.assertTrue((self.eval_dict_prior[lens_param]['eval_fn'] is
					distributions.eval_normal_logpdf_approx) or (
					self.eval_dict_prior[lens_param]['eval_fn'] is
					distributions.eval_lognormal_logpdf_approx))
			total += n_p

		hyp_eval_values = np.log([1/10,1/10,1/10,1/10,1/10,1/10,1/2,1/10,1/2,
			1/10,1/10,1/10,1/10,1/10])
		self.assertEqual(len(hyp_eval_values),len(
			self.eval_dict_prior['hyp_prior']))
		for hpi, hyp_prior in enumerate(self.eval_dict_prior['hyp_prior']):
			self.assertAlmostEqual(hyp_eval_values[hpi],hyp_prior(0.5))
		hyp_eval_values = np.log([1/10,0,1/10,0,1/10,0,1/2,
			0,1/2,0,1/10,0,1/10,0,1/10,0])
		for hpi, hyp_prior in enumerate(self.eval_dict_prior['hyp_prior']):
			self.assertAlmostEqual(hyp_eval_values[hpi],hyp_prior(-0.5))

		# Now test a distribution with a covariance
		self.assertEqual(self.eval_dict_cov['hyp_len'],15)
		self.assertListEqual(list(self.eval_dict_cov['hyp_init']),[-2.73,1.05,
			0.0,0.102,0.0,0.102,0.242, -0.408, 0.696,0.5,0.5,0.5,0.4,0.4,0.4])
		self.assertListEqual(list(self.eval_dict_cov['hyp_sigma']),[0.5,0.05,
			0.2,0.03,0.2,0.03,0.1,0.1,0.1,0.5,0.5,0.5,0.4,0.4,0.4])
		self.assertListEqual(self.eval_dict_cov['hyp_names'],[
			'external_shear_gamma_ext:mu','external_shear_gamma_ext:sigma',
			'lens_mass_center_x:mu','lens_mass_center_x:sigma',
			'lens_mass_center_y:mu','lens_mass_center_y:sigma','cov_mu_0',
			'cov_mu_1','cov_mu_2','cov_tril_0','cov_tril_1','cov_tril_2',
			'cov_tril_3','cov_tril_4','cov_tril_5'])

	def test_log_p_omega(self):
		# Check that the log_p_omega function returns the desired value for both
		# dicts.
		hyp=np.ones(14)*0.5
		self.assertAlmostEqual(hierarchical_inference.log_p_omega(hyp,
			self.eval_dict_prior),np.sum(np.log([1/10,1/10,1/10,1/10,1/10,1/10,
				1/2,1/10,1/2,1/10,1/10,1/10,1/10,1/10])))
		hyp=-np.ones(14)*0.5
		self.assertAlmostEqual(hierarchical_inference.log_p_omega(hyp,
			self.eval_dict_prior),-np.inf)

		hyp=np.ones(15)*0.5
		self.assertAlmostEqual(hierarchical_inference.log_p_omega(hyp,
			self.eval_dict_cov),np.log(1/10)*15)

		hyp[-1] = -1
		self.assertAlmostEqual(hierarchical_inference.log_p_omega(hyp,
			self.eval_dict_cov),-np.inf)

	def test_log_p_xi_omega(self):
		# Test that the log_p_xi_omega function returns the correct value
		# for some sample data points.
		hyp = np.array([-2.73,1.05,0.0,0.102,0.0,0.102,0.0,0.1,0.0,0.1,0.7,0.1,
			0.0,0.1])
		samples = np.ones((8,2,2))*0.3

		def hand_calc_log_pdf(samples,hyp):
			# Add each one of the probabilities
			scipy_pdf = stats.lognorm.logpdf(samples[0],scale=np.exp(hyp[0]),
				s=hyp[1])

			scipy_pdf += stats.uniform.logpdf(samples[1],loc=-0.5*np.pi,
				scale=np.pi)

			scipy_pdf += stats.norm.logpdf(samples[2],loc=hyp[2],scale=hyp[3])
			scipy_pdf += stats.norm.logpdf(samples[3],loc=hyp[4],scale=hyp[5])

			scipy_pdf += stats.norm.logpdf(samples[4],loc=hyp[6],scale=hyp[7])
			scipy_pdf += stats.norm.logpdf(samples[5],loc=hyp[8],scale=hyp[9])

			scipy_pdf += stats.lognorm.logpdf(samples[6],scale=np.exp(
				hyp[10]),s=hyp[11])
			scipy_pdf += stats.lognorm.logpdf(samples[7],scale=np.exp(
				hyp[12]),s=hyp[13])

			return scipy_pdf

		def hand_calc_log_pdf_cov(samples,hyp):
			# Add each one of the probabilities
			scipy_pdf = stats.lognorm.logpdf(samples[0],scale=np.exp(hyp[0]),
				s=hyp[1])
			scipy_pdf += stats.uniform.logpdf(samples[1],loc=-0.5*np.pi,
				scale=np.pi)

			scipy_pdf += stats.norm.logpdf(samples[2],loc=hyp[2],scale=hyp[3])
			scipy_pdf += stats.norm.logpdf(samples[3],loc=hyp[4],scale=hyp[5])

			scipy_pdf += stats.uniform.logpdf(samples[5],loc=-0.5*np.pi,
				scale=np.pi)

			# Now calculate the covariance matrix values.
			cov_samples = samples[[7,4,6]]
			mu = [0.242,-0.408,0.696]
			cov = np.array([[0.25, 0.25, 0.2],
				[0.25, 0.5, 0.4],[0.2, 0.4, 0.48]])
			for i in range(len(scipy_pdf)):
				for j in range(len(scipy_pdf[0])):
					scipy_pdf[i,j] += stats.multivariate_normal.logpdf(
						np.log(cov_samples[:,i,j]),mean=mu,cov=cov)
					scipy_pdf[i,j] -= np.log(stats.norm(mu[1],
						np.sqrt(cov[1,1])).cdf(1))
			return scipy_pdf

		np.testing.assert_array_almost_equal(
			hierarchical_inference.log_p_xi_omega(samples,hyp,
				self.eval_dict_prior,self.lens_params),
			hand_calc_log_pdf(samples,hyp))

		samples = np.random.uniform(size=(8,2,3))*0.3
		np.testing.assert_array_almost_equal(
			hierarchical_inference.log_p_xi_omega(samples,hyp,
				self.eval_dict_prior,self.lens_params),
			hand_calc_log_pdf(samples,hyp))

		hyp = np.array([-2.73,1.10,0.0,0.2,0.1,0.2,0.0,0.1,0.0,0.1,0.8,0.1,
			0.0,0.1])
		np.testing.assert_array_almost_equal(
			hierarchical_inference.log_p_xi_omega(samples,hyp,
				self.eval_dict_prior,self.lens_params),
			hand_calc_log_pdf(samples,hyp))

		hyp = np.array([-2.73,1.05,0.0,0.102,0.0,0.102,0.242,-0.408,0.696,0.5,
			0.5,0.5,0.4,0.4,0.4])
		np.testing.assert_array_almost_equal(
			hierarchical_inference.log_p_xi_omega(samples,hyp,
				self.eval_dict_cov,self.lens_params_cov),
			hand_calc_log_pdf_cov(samples,hyp))


class HierarchicalClassTest(unittest.TestCase):

	def setUp(self):
		# Open up the config file.
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		with open(self.root_path+'test.json','r') as json_f:
			self.cfg = json.load(json_f)
		self.interim_baobab_omega_path = self.root_path+'test_baobab_cfg.py'
		self.target_ovejero_omega_path = self.root_path+'test_ovejero_cfg_prior.py'
		self.target_baobab_omega_path = self.root_path+'test_baobab_cfg_target.py'
		self.lens_params = self.cfg['dataset_params']['lens_params']
		self.num_params = len(self.lens_params)
		self.batch_size = 20

		self.normalized_param_path = self.root_path + 'new_metadata.csv'
		self.normalization_constants_path = self.root_path + 'norm.csv'
		self.final_params = self.cfg['training_params']['final_params']
		self.cfg['dataset_params']['normalization_constants_path'] = 'norm.csv'
		self.cfg['training_params']['bnn_type'] = 'diag'
		self.tf_record_path = self.root_path+self.cfg['validation_params'][
			'tf_record_path']

		# We'll have to make the tf record and clean it up at the end
		model_trainer.prepare_tf_record(self.cfg,self.root_path,
				self.tf_record_path,self.final_params,
				train_or_test='train')

		self.hclass = hierarchical_inference.HierarchicalClass(self.cfg,
			self.interim_baobab_omega_path,self.target_ovejero_omega_path,
			self.root_path,self.tf_record_path,self.target_baobab_omega_path,
			lite_class=True)

		os.remove(self.tf_record_path)

	def tearDown(self):
		# Do some cleanup for memory management
		self.hclass.infer_class = None
		self.hclass = None
		self.cfg = None
		# Force collection
		gc.collect()

	def test_init(self):
		# Check that the true hyperparameter values were correctly initialized.
		true_hyp_values = [-2.73,1.05,0.0,0.102,0.0,0.102,0.0,0.1,0.0,0.1,0.7,
			0.1,0.0,0.1]
		self.assertListEqual(self.hclass.true_hyp_values,true_hyp_values)

	def test_gen_samples(self):

		# Test that generating samples gives reasonable outputs.
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
		self.hclass.infer_class.flip_mat_list = [
			np.diag(np.ones(self.num_params))]

		# Create tf record. This won't be used, but it has to be there for
		# the function to be able to pull some images.
		# Make fake norms data
		fake_norms = {}
		for lens_param in self.final_params + self.lens_params:
			fake_norms[lens_param] = np.array([0.0,1.0])
		fake_norms = pd.DataFrame(data=fake_norms)
		fake_norms.to_csv(self.normalization_constants_path,index=False)
		train_or_test = 'test'
		model_trainer.prepare_tf_record(self.cfg, self.root_path,
				self.tf_record_path,self.final_params,train_or_test)

		# Replace the real model with our fake model and generate samples
		self.hclass.infer_class.model = diag_model
		# Set this to false so the system doesn't complain when we try
		# to generate samples.
		self.hclass.infer_class.lite_class = False

		self.hclass.gen_samples(100)

		# First, make sure all of the values of lens_samps were filled out.
		for pi in range(self.num_params):
			self.assertGreater(np.sum(hierarchical_inference.lens_samps[pi]),0)

		# Check that the parameters that got changed did so in the right way.
		# First check theta_e
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,-1]-
			np.log(hierarchical_inference.lens_samps[-1]))),0)
		# Now check the catersian to polar for shears.
		gamma = hierarchical_inference.lens_samps[0]
		ang = hierarchical_inference.lens_samps[1]
		g1 = gamma*np.cos(2*ang)
		g2 = gamma*np.sin(2*ang)
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,0]-
			g1)),0)
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,1]-
			g2)),0)

		# Just make sure this is set and set using the interim dict.
		np.testing.assert_almost_equal(self.hclass.prob_class.pt_omegai,
			hierarchical_inference.log_p_xi_omega(
				hierarchical_inference.lens_samps,
				self.hclass.interim_eval_dict['hyp_values'],
				self.hclass.interim_eval_dict,self.hclass.lens_params_train))

		# Now check that if we offer a save path it gets used.
		save_path = self.root_path + 'save_samps/'
		self.hclass.gen_samples(100,save_path)
		orig_samps = np.copy(hierarchical_inference.lens_samps)
		orig_bnn_samps = np.copy(self.hclass.infer_class.predict_samps)
		self.hclass.gen_samples(100,save_path)

		np.testing.assert_almost_equal(orig_samps,
			hierarchical_inference.lens_samps)
		np.testing.assert_almost_equal(orig_bnn_samps,
			self.hclass.infer_class.predict_samps)

		# Finall check that subsampling doesn't cause any issues
		subsample= 10
		self.hclass.gen_samples(100,save_path,subsample=subsample)
		np.testing.assert_almost_equal(hierarchical_inference.lens_samps,
			orig_samps[:,:,:10])

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.normalized_param_path)
		os.remove(self.tf_record_path)
		os.remove(save_path+'lens_samps.npy')
		os.remove(save_path+'pred.npy')
		os.remove(save_path+'al_cov.npy')
		os.remove(save_path+'images.npy')
		os.remove(save_path+'y_test.npy')
		os.rmdir(save_path)

	def test_log_post_omega(self):
		# Test that the log_p_xi_omega function returns the correct value
		# for some sample data points.
		hyp = np.array([-2.73,1.10,0.0,0.2,0.1,0.2,0.0,0.1,0.0,0.1,0.8,0.1,
			0.0,0.1])
		samples = np.ones((8,2,2))*0.3

		# Initialize the fake sampling in our hclass
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		lpxo = hierarchical_inference.log_p_xi_omega(samples,hyp,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpo = hierarchical_inference.log_p_omega(hyp,
			self.hclass.target_eval_dict)

		places = 7
		post_hand = np.sum(special.logsumexp(lpxo-
			self.hclass.prob_class.pt_omegai,axis=0))+lpo
		self.assertAlmostEqual(self.hclass.log_post_omega(hyp),post_hand,
			places=places)

		samples = np.random.uniform(size=(8,2,2))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		lpxo = hierarchical_inference.log_p_xi_omega(samples,hyp,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpo = hierarchical_inference.log_p_omega(hyp,
			self.hclass.target_eval_dict)

		places = 7
		post_hand = np.sum(special.logsumexp(lpxo-
			self.hclass.prob_class.pt_omegai,axis=0))+lpo
		self.assertAlmostEqual(self.hclass.log_post_omega(hyp),post_hand,
			places=places)

		hyp = np.array([-2.73,1.10,0.1,0.3,0.01,0.3,0.0,0.1,0.0,0.1,0.8,0.1,
			0.0,0.1])
		lpxo = hierarchical_inference.log_p_xi_omega(samples,hyp,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpo = hierarchical_inference.log_p_omega(hyp,
			self.hclass.target_eval_dict)

		places = 7
		post_hand = np.sum(special.logsumexp(lpxo-
			self.hclass.prob_class.pt_omegai,axis=0))+lpo
		self.assertAlmostEqual(self.hclass.log_post_omega(hyp),post_hand,
			places=places)

		# self.hclass.samples_init=False

	def test_initialize_sampler(self):
		# Test that the walker initialization is correct.
		test_chains_path = self.root_path + 'test_chains_is.h5'
		n_walkers = 60

		# Make some fake samples.
		samples = np.ones((8,2,2))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		self.hclass.initialize_sampler(n_walkers,test_chains_path)

		self.assertTrue(os.path.isfile(test_chains_path))
		self.assertTrue(self.hclass.cur_state is not None)
		self.assertEqual(len(self.hclass.cur_state),n_walkers)

		# Check that no walker was initialized to a point with
		# -np.inf
		for walker in self.hclass.cur_state:
			self.assertFalse(self.hclass.log_post_omega(walker)==-np.inf)

		os.remove(test_chains_path)

	def test_run_sampler(self):
		# Here, we're mostly just testing things don't crash since the
		# work is being done by emcee.
		test_chains_path = self.root_path + 'test_chains_rs.h5'
		n_walkers = 60
		n_samps = 10

		# Make some fake samples.
		samples = np.ones((8,2,2))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		self.hclass.initialize_sampler(n_walkers,test_chains_path)
		self.hclass.run_sampler(n_samps)

		chains = self.hclass.sampler.get_chain()
		self.assertEqual(chains.shape[0],n_samps)
		self.assertEqual(chains.shape[1],n_walkers)

		self.assertGreater(np.max(np.abs(chains[:,-1]-chains[:,-2])),0)

		os.remove(test_chains_path)

	def test_plots(self):
		# Here, we're mostly just testing things don't crash again.
		test_chains_path = self.root_path + 'test_chains_tp.h5'
		n_walkers = 60
		n_samps = 10
		burnin = 0

		# Make some fake samples.
		samples = np.random.uniform(size=(8,2,40))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		self.hclass.initialize_sampler(n_walkers,test_chains_path)
		self.hclass.run_sampler(n_samps)

		chains = self.hclass.sampler.get_chain()
		hyperparam_plot_names = ['test']*chains.shape[-1]

		block = False
		self.hclass.plot_chains(block=block)
		plt.close('all')
		self.hclass.plot_corner(burnin,block=block)
		plt.close('all')
		self.hclass.plot_distributions(burnin,block=block)
		plt.close('all')
		self.hclass.plot_corner(burnin,hyperparam_plot_names,block=block)
		plt.close('all')
		self.hclass.plot_single_corner(burnin,'external_shear_gamma_ext',
			hyperparam_plot_names,block=block)
		plt.close('all')
		self.hclass.plot_distributions(burnin,hyperparam_plot_names,block=block,
			dpi=50)
		plt.close('all')
		self.hclass.plot_auto_corr(block=block)
		plt.close('all')
		dist_lens_params = ['external_shear_gamma_ext','lens_mass_center_x',
			'lens_mass_center_y','lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.hclass.plot_parameter_distribtuion(burnin,dist_lens_params)

		os.remove(test_chains_path)

	def test_calculate_samples_weights(self):
		# Test that the weights calculated are as expected.
		# First we set some samples
		samples = np.ones((8,2,2))*0.3
		hierarchical_inference.lens_samps = samples

		# We want some hyperparameters to test. W'll only put two groups of
		# samples in our fake chain.
		hyp1 = np.array([-2.73,1.10,0.1,0.3,0.01,0.3,0.0,0.1,0.0,0.1,0.8,0.1,
			0.0,0.1])
		hyp2 = np.array([-2.72,1.10,0.1,0.3,0.01,0.3,0.0,0.1,0.0,0.1,0.7,0.1,
			0.0,0.1])

		# Make a fake samples
		class FakeSampler():
			def __init__(self,hyp1,hyp2):
				self.chain = np.zeros((10,len(hyp1)))
				self.chain[:5] = hyp1
				self.chain[5:] = hyp2

			def get_chain(self):
				return self.chain

		# Make our fake sampler and port that in.
		self.hclass.sampler = FakeSampler(hyp1,hyp2)

		lpxi1 = hierarchical_inference.log_p_xi_omega(samples,hyp1,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpxi2 = hierarchical_inference.log_p_xi_omega(samples,hyp2,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpi = hierarchical_inference.log_p_xi_omega(samples,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)
		self.hclass.prob_class.pt_omegai = lpi

		# Calculate weights with function
		n_p_omega_samps = 10
		burnin = 0
		weights = self.hclass.calculate_sample_log_weights(n_p_omega_samps,
			burnin)

		# Calculate the weights we expect by hand.
		hand_weights = 0.5*(np.exp(lpxi1-lpi)+np.exp(lpxi2-lpi))
		np.testing.assert_almost_equal(np.exp(weights),hand_weights)

		self.hclass.sampler = None

class HierarchicalEmpiricalTest(unittest.TestCase):

	def setUp(self):
		# Open up the config file.
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		with open(self.root_path+'test.json','r') as json_f:
			self.cfg = json.load(json_f)
		self.interim_baobab_omega_path = self.root_path+'test_baobab_cfg.py'
		self.target_ovejero_omega_path = self.root_path+'test_emp_cfg_prior.py'
		self.target_baobab_omega_path = self.root_path+'test_emp_cfg.py'
		self.lens_params = self.cfg['dataset_params']['lens_params']
		self.num_params = len(self.lens_params)
		self.batch_size = 20

		self.normalized_param_path = self.root_path + 'new_metadata.csv'
		self.normalization_constants_path = self.root_path + 'norm.csv'
		self.final_params = self.cfg['training_params']['final_params']
		self.cfg['dataset_params']['normalization_constants_path'] = 'norm.csv'
		self.cfg['training_params']['bnn_type'] = 'diag'
		self.tf_record_path = self.root_path+self.cfg['validation_params'][
			'tf_record_path']

		# We'll have to make the tf record and clean it up at the end
		model_trainer.prepare_tf_record(self.cfg,self.root_path,
				self.tf_record_path,self.final_params,
				train_or_test='train')

		# Make the train_to_test_param_map
		self.train_to_test_param_map = dict(
			orig_params=['lens_mass_e1','lens_mass_e2'],
			transform_func=ellipticity2phi_q,
			new_params=['lens_mass_phi','lens_mass_q'])

		self.hclass = hierarchical_inference.HierarchicalClass(self.cfg,
			self.interim_baobab_omega_path,self.target_ovejero_omega_path,
			self.root_path,self.tf_record_path,self.target_baobab_omega_path,
			self.train_to_test_param_map,lite_class=True)

		os.remove(self.tf_record_path)

	def tearDown(self):
		# Clean up to save memory
		self.hclass.infer_class = None
		self.hclass = None
		self.cfg = None
		self.train_to_test_param_map = None
		# This is so bad I'm going to force garbage collection
		gc.collect()

	def test_init(self):
		# Check that the true hyperparameter values were correctly initialized.
		true_hyp_values = np.array([-2.73,0.1,0.0,0.05,0.0,0.05,0.242,-0.408,
			0.696, 0.30592393, -0.1046923, 0.34144243, -0.04409956, 0.01650766,
			0.11489763])
		np.testing.assert_almost_equal(self.hclass.true_hyp_values,
			true_hyp_values)

	def test_gen_samples(self):

		# Test that generating samples gives reasonable outputs.
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
		self.hclass.infer_class.flip_mat_list = [
			np.diag(np.ones(self.num_params))]

		# Create tf record. This won't be used, but it has to be there for
		# the function to be able to pull some images.
		# Make fake norms data
		fake_norms = {}
		for lens_param in self.final_params + self.lens_params:
			fake_norms[lens_param] = np.array([0.0,1.0])
		fake_norms = pd.DataFrame(data=fake_norms)
		fake_norms.to_csv(self.normalization_constants_path,index=False)
		train_or_test = 'test'
		model_trainer.prepare_tf_record(self.cfg, self.root_path,
				self.tf_record_path,self.final_params,train_or_test)

		# Replace the real model with our fake model and generate samples
		self.hclass.infer_class.model = diag_model
		# Set this to false so the system doesn't complain when we try
		# to generate samples.
		self.hclass.infer_class.lite_class = False

		self.hclass.gen_samples(100)

		# All we need to check here is that the mapping from e1, e2 to
		# phi and q was succesful.
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,-1]-
			np.log(hierarchical_inference.lens_samps[-1]))),0)
		# Now check the catersian to polar for shears.
		e1 = self.hclass.predict_samps[:,:,4]
		e2 = self.hclass.predict_samps[:,:,5]
		phi, q = ellipticity2phi_q(e1,e2)
		np.testing.assert_almost_equal(phi,hierarchical_inference.lens_samps[4])
		np.testing.assert_almost_equal(q,hierarchical_inference.lens_samps[5])

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.normalized_param_path)
		os.remove(self.tf_record_path)

	def test_log_post_omega(self):
		# Test that the log_p_xi_omega function returns the correct value
		# for some sample data points.
		hyp = np.array([-2.73,1.05,0.0,0.102,0.0,0.102,0.242,-0.408,0.696,0.5,
			0.5,0.5,0.4,0.4,0.4])
		samples = np.ones((8,2,2))*0.3

		# Initialize the fake sampling in our hclass
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		lpxo = hierarchical_inference.log_p_xi_omega(samples,hyp,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpo = hierarchical_inference.log_p_omega(hyp,
			self.hclass.target_eval_dict)

		places = 7
		post_hand = np.sum(special.logsumexp(lpxo-
			self.hclass.prob_class.pt_omegai,axis=0))+lpo
		self.assertAlmostEqual(self.hclass.log_post_omega(hyp),post_hand,
			places=places)

		samples = np.random.uniform(size=(8,2,2))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		lpxo = hierarchical_inference.log_p_xi_omega(samples,hyp,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpo = hierarchical_inference.log_p_omega(hyp,
			self.hclass.target_eval_dict)

		places = 7
		post_hand = np.sum(special.logsumexp(lpxo-
			self.hclass.prob_class.pt_omegai,axis=0))+lpo
		self.assertAlmostEqual(self.hclass.log_post_omega(hyp),post_hand,
			places=places)

	def test_initialize_sampler(self):
		# Test that the walker initialization is correct.
		test_chains_path = self.root_path + 'test_chains_is.h5'
		n_walkers = 60

		# Make some fake samples.
		samples = np.ones((8,2,2))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		self.hclass.initialize_sampler(n_walkers,test_chains_path)

		self.assertTrue(os.path.isfile(test_chains_path))
		self.assertTrue(self.hclass.cur_state is not None)
		self.assertEqual(len(self.hclass.cur_state),n_walkers)

		# Check that no walker was initialized to a point with
		# -np.inf
		for walker in self.hclass.cur_state:
			self.assertFalse(self.hclass.log_post_omega(walker)==-np.inf)

		os.remove(test_chains_path)

	def test_run_sampler(self):
		# Here, we're mostly just testing things don't crash since the
		# work is being done by emcee.
		test_chains_path = self.root_path + 'test_chains_rs.h5'
		n_walkers = 60
		n_samps = 2

		# Make some fake samples.
		samples = np.ones((8,2,2))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		self.hclass.initialize_sampler(n_walkers,test_chains_path)
		self.hclass.run_sampler(n_samps)

		chains = self.hclass.sampler.get_chain()
		self.assertEqual(chains.shape[0],n_samps)
		self.assertEqual(chains.shape[1],n_walkers)

		self.assertGreater(np.max(np.abs(chains[:,-1]-chains[:,-2])),0)

		os.remove(test_chains_path)

	def test_plots(self):
		# Here, we're mostly just testing things don't crash again.
		test_chains_path = self.root_path + 'test_chains_tp.h5'
		n_walkers = 60
		n_samps = 2
		burnin = 0

		# Make some fake samples.
		samples = np.random.uniform(size=(8,2,40))*0.3
		hierarchical_inference.lens_samps=samples
		self.hclass.prob_class.set_samples()
		self.hclass.prob_class.pt_omegai=hierarchical_inference.log_p_xi_omega(
			hierarchical_inference.lens_samps,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)

		self.hclass.initialize_sampler(n_walkers,test_chains_path)
		self.hclass.run_sampler(n_samps)

		chains = self.hclass.sampler.get_chain()
		hyperparam_plot_names = ['test']*chains.shape[-1]

		block = False
		self.hclass.plot_chains(block=block)
		plt.close('all')
		self.hclass.plot_corner(burnin,block=block)
		plt.close('all')
		self.hclass.plot_distributions(burnin,block=block)
		plt.close('all')
		self.hclass.plot_corner(burnin,hyperparam_plot_names,block=block)
		plt.close('all')
		self.hclass.plot_distributions(burnin,hyperparam_plot_names,block=block,
			dpi=50)
		_ = self.hclass.plot_cov_corner(burnin,hyperparam_plot_names)
		dist_lens_params = ['lens_mass_center_x',
			'lens_mass_center_y',
			'lens_mass_gamma','lens_mass_theta_E']
		self.hclass.plot_parameter_distribtuion(burnin,dist_lens_params)
		plt.close('all')

		os.remove(test_chains_path)

	def test_calculate_samples_weights(self):
		# Test that the weights calculated are as expected.
		# First we set some samples
		samples = np.ones((8,2,2))*0.3
		hierarchical_inference.lens_samps = samples

		# We want some hyperparameters to test. W'll only put two groups of
		# samples in our fake chain.
		hyp1 = np.array([-2.73,1.05,0.0,0.102,0.0,0.102,0.242,-0.408,0.696,0.5,
			0.5,0.5,0.4,0.4,0.4])
		hyp2 = np.array([-2.72,1.05,0.0,0.102,0.0,0.102,0.242,-0.408,0.696,0.5,
			0.4,0.4,0.4,0.4,0.4])

		# Make a fake samples
		class FakeSampler():
			def __init__(self,hyp1,hyp2):
				self.chain = np.zeros((10,len(hyp1)))
				self.chain[:5] = hyp1
				self.chain[5:] = hyp2

			def get_chain(self):
				return self.chain

		# Make our fake sampler and port that in.
		self.hclass.sampler = FakeSampler(hyp1,hyp2)

		lpxi1 = hierarchical_inference.log_p_xi_omega(samples,hyp1,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpxi2 = hierarchical_inference.log_p_xi_omega(samples,hyp2,
			self.hclass.target_eval_dict,self.hclass.lens_params_test)
		lpi = hierarchical_inference.log_p_xi_omega(samples,
			self.hclass.interim_eval_dict['hyp_values'],
			self.hclass.interim_eval_dict,self.hclass.lens_params_train)
		self.hclass.prob_class.pt_omegai = lpi

		# Calculate weights with function
		n_p_omega_samps = 10
		burnin = 0
		weights = self.hclass.calculate_sample_log_weights(n_p_omega_samps,burnin)

		# Calculate the weights we expect by hand.
		hand_weights = 0.5*(np.exp(lpxi1-lpi)+np.exp(lpxi2-lpi))
		np.testing.assert_almost_equal(weights,np.log(hand_weights))

		self.hclass.sampler = None
