import unittest, os, json
# import tensorflow as tf
from ovejero import hierarchical_inference, model_trainer
from baobab.configs import BaobabConfig
import numpy as np
from scipy import stats
import tensorflow as tf
import pandas as pd

class HierarchicalnferenceTest(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(HierarchicalnferenceTest, self).__init__(*args, **kwargs)
		# Open up the config file.
		np.random.seed(2)

	def test_eval_norm_pdf(self):
		# Test that the norm pdf gives what you expect
		n_test = 10
		mu = np.random.randn(n_test)
		sigma = np.random.rand(n_test)
		n_points = 1000
		samples = np.random.rand(n_points)

		for i in range(n_test):
			pdf = hierarchical_inference.eval_norm_log_pdf(samples,[mu[i],
				sigma[i]])
			scipy_p = stats.norm.logpdf(samples,loc=mu[i],scale=sigma[i])
			self.assertAlmostEqual(np.max(np.abs(pdf-scipy_p)),0)

	def test_eval_log_norm_pdf(self):
		# Test that the norm pdf gives what you expect
		n_test = 10
		mu = np.random.randn(n_test)
		sigma = np.random.rand(n_test)
		n_points = 1000
		samples = np.random.rand(n_points)

		for i in range(n_test):
			pdf = hierarchical_inference.eval_log_norm_log_pdf(samples,[mu[i],
				sigma[i]])
			scipy_p = stats.lognorm.logpdf(samples,scale=np.exp(mu[i]),s=sigma[i])
			self.assertAlmostEqual(np.max(np.abs(pdf-scipy_p)),0)

	def test_eval_beta_log_pdf(self):
		# Test that the norm pdf gives what you expect
		n_test = 10
		a = np.random.rand(n_test)
		b = np.random.rand(n_test)
		lower = np.random.rand(n_test)-1
		upper = np.random.rand(n_test)+1
		n_points = 1000
		samples = np.random.rand(n_points)

		for i in range(n_test):
			pdf = hierarchical_inference.eval_beta_log_pdf(samples,[a[i],b[i],
				lower[i],upper[i]])
			scipy_p = stats.beta.logpdf(samples,a=a[i],b=b[i],loc=lower[i],
				scale=upper[i]-lower[i])
			self.assertAlmostEqual(np.max(np.abs(pdf-scipy_p)),0)

		samples = np.random.rand(n_points) - 5
		self.assertTrue(np.all(hierarchical_inference.eval_beta_log_pdf(samples,
			[a[0],b[0],lower[0],upper[0]])==-np.inf))

		samples = np.random.rand(n_points) + 5
		self.assertTrue(np.all(hierarchical_inference.eval_beta_log_pdf(samples,
			[a[0],b[0],lower[0],upper[0]])==-np.inf))

	def test_eval_gen_norm_log_pdf(self):
		# Test that the norm pdf gives what you expect
		n_test = 10
		mu = np.random.rand(n_test)-0.5
		alpha = np.random.rand(n_test)
		p = np.ones((n_test))*10
		lower = np.random.rand(n_test)-2
		upper = np.random.rand(n_test)+1
		n_points = 1000
		samples = np.random.rand(n_points)

		for i in range(n_test):
			pdf = hierarchical_inference.eval_gen_norm_log_pdf(samples,[mu[i],
				alpha[i],p[i],lower[i],upper[i]])
			dist = stats.gennorm(beta=p[i],loc=mu[i],scale=alpha[i])
			scipy_p = dist.logpdf(samples) / (dist.cdf(upper[i]) - dist.cdf(
				lower[i]))
			self.assertAlmostEqual(np.max(np.abs(pdf-scipy_p)),0)

		samples = np.random.rand(n_points)+5
		self.assertTrue(np.all(hierarchical_inference.eval_gen_norm_log_pdf(
			samples,[mu[i],alpha[i],p[i],lower[i],upper[i]])==-np.inf))

		samples = np.random.rand(n_points)-5
		self.assertTrue(np.all(hierarchical_inference.eval_gen_norm_log_pdf(
			samples,[mu[i],alpha[i],p[i],lower[i],upper[i]])==-np.inf))

	def test_build_evaluation_dictionary(self):
		# Check that the eval dictionary is built correctly for an example with
		# and without priors.
		root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		cfg = BaobabConfig.from_file(root_path + 'test_baobab_cfg.py')
		cfg_pr = BaobabConfig.from_file(root_path + 'test_baobab_cfg_prior.py')
		lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		n_lens_param_p_params = [2,5,2,2,4,4,2,2]
		eval_dict = hierarchical_inference.build_evaluation_dictionary(cfg,
			lens_params)
		eval_dict_prior = hierarchical_inference.build_evaluation_dictionary(
			cfg_pr,lens_params,extract_hyperpriors=True)

		# First we test the case without priors.
		self.assertEqual(eval_dict['hyp_len'],23)
		self.assertListEqual(list(eval_dict['hyps']),[-2.73,1.05,0.0,0.5*np.pi,
			10.0,-0.5*np.pi,0.5*np.pi,0.0,0.102,0.0,0.102,4.0,4.0,-0.55,0.55,4.0,
			4.0,-0.55,0.55,0.7,0.1,0.0,0.1])
		self.assertListEqual(eval_dict['hyp_names'],[
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
		for li in range(len(lens_params)):
			lens_param = lens_params[li]
			n_p = n_lens_param_p_params[li]
			self.assertListEqual(list(eval_dict[lens_param]['hyp_ind']),
				list(range(total,total+n_p)))
			if n_p == 2:
				self.assertTrue((eval_dict[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_norm_log_pdf) or (
					eval_dict[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_log_norm_log_pdf))
			if n_p == 4:
				self.assertTrue(eval_dict[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_beta_log_pdf)
			if n_p == 5:
				self.assertTrue(eval_dict[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_gen_norm_log_pdf)
			total += n_p

		# Now we test the case with priors.
		self.assertEqual(eval_dict_prior['hyp_len'],23)
		self.assertListEqual(list(eval_dict_prior['hyps']),[-2.73,1.05,0.0,
			0.5*np.pi,10.0,-0.5*np.pi,0.5*np.pi,0.0,0.102,0.0,0.102,4.0,4.0,-0.55,
			0.55,4.0,4.0,-0.55,0.55,0.7,0.1,0.0,0.1])
		self.assertListEqual(eval_dict_prior['hyp_names'],[
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
		for li in range(len(lens_params)):
			lens_param = lens_params[li]
			n_p = n_lens_param_p_params[li]
			self.assertListEqual(list(eval_dict_prior[lens_param]['hyp_ind']),
				list(range(total,total+n_p)))
			if n_p == 2:
				self.assertTrue((eval_dict_prior[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_norm_log_pdf) or (
					eval_dict_prior[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_log_norm_log_pdf))
			if n_p == 4:
				self.assertTrue(eval_dict_prior[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_beta_log_pdf)
			if n_p == 5:
				self.assertTrue(eval_dict_prior[lens_param]['eval_fn'] is 
					hierarchical_inference.eval_gen_norm_log_pdf)
			total += n_p

		self.assertListEqual(list(eval_dict_prior['hyp_prior'][0]),[-np.inf,0.0,
			-np.inf,0.0,0.0,-np.inf,-np.inf,-np.inf,0.0,-np.inf,0.0,0.0,
			0.0,-np.inf,-np.inf,0.0,0.0,-np.inf,-np.inf,-np.inf,0.0,-np.inf,0.0])
		self.assertListEqual(list(eval_dict_prior['hyp_prior'][1]),[np.inf]*23)


class HierarchicalClassTest(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(HierarchicalClassTest, self).__init__(*args, **kwargs)
		# Open up the config file.
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		with open(self.root_path+'test.json','r') as json_f:
			self.cfg = json.load(json_f)
		self.interim_baobab_omega_path = self.root_path+'test_baobab_cfg.py'
		self.target_baobab_omega_path = self.root_path+'test_baobab_cfg_prior.py'
		self.lens_params = self.cfg['dataset_params']['lens_params']
		self.num_params = len(self.lens_params)
		self.batch_size = self.cfg['training_params']['batch_size']

		self.normalized_param_path = self.root_path + 'new_metadata.csv'
		self.normalization_constants_path = self.root_path + 'norm.csv'
		self.final_params = self.cfg['training_params']['final_params']
		self.cfg['dataset_params']['normalization_constants_path'] = 'norm.csv'
		self.cfg['training_params']['bnn_type'] = 'diag'
		self.tf_record_path = self.root_path+self.cfg['validation_params'][
			'tf_record_path']

		self.hclass = hierarchical_inference.HierarchicalClass(self.cfg,
			self.interim_baobab_omega_path,self.target_baobab_omega_path)

	def test_log_p_omega(self):
		# Test that log_p_omega gives the expected behavior on the boundaries
		hyp = np.ones(23)
		self.assertEqual(self.hclass.log_p_omega(hyp),0)

		hyp = np.array([-2.73,1.05,0.0,0.5*np.pi,10.0,-0.5*np.pi,0.5*np.pi,0.0,
			0.102,0.0,0.102,4.0,4.0,-0.55,0.55,4.0,4.0,-0.55,0.55,0.7,0.1,0.0,
			0.1])
		self.assertEqual(self.hclass.log_p_omega(hyp),0)

		hyp = np.array([-2.73,-1.05,0.0,0.5*np.pi,10.0,-0.5*np.pi,0.5*np.pi,0.0,
			0.102,0.0,0.102,4.0,4.0,-0.55,0.55,4.0,4.0,-0.55,0.55,0.7,0.1,0.0,
			0.1])
		self.assertEqual(self.hclass.log_p_omega(hyp),-np.inf)

		hyp = np.array([-2.73,1.05,0.0,0.5*np.pi,10.0,-0.5*np.pi,0.5*np.pi,0.0,
			0.102,0.0,0.102,4.0,4.0,-0.55,0.55,4.0,4.0,-0.55,100,0.7,0.1,0.0,
			0.1])
		self.assertEqual(self.hclass.log_p_omega(hyp),0)

	def test_log_p_theta_omega(self):
		# Test that the log_p_theta_omega function returns the correct value 
		# for some sample data points.
		hyp = np.array([-2.73,1.05,0.0,0.5*np.pi,10.0,-0.5*np.pi,0.5*np.pi,0.0,
			0.102,0.0,0.102,4.0,4.0,-0.55,0.55,4.0,4.0,-0.55,0.55,0.7,0.1,0.0,
			0.1])
		samples = np.ones((2,2,8))*0.3


		def hand_calc_log_pdf(samples,hyp):
			# Add each one of the probabilities
			scipy_pdf = stats.lognorm.logpdf(samples[:,:,0],scale=np.exp(hyp[0]),
				s=hyp[1])

			dist = stats.gennorm(beta=hyp[4],loc=hyp[2],scale=hyp[3])
			scipy_pdf += dist.logpdf(samples[:,:,1])/(dist.cdf(hyp[6]) - 
				dist.cdf(hyp[5]))

			scipy_pdf += stats.norm.logpdf(samples[:,:,2],loc=hyp[7],
				scale=hyp[8])
			scipy_pdf += stats.norm.logpdf(samples[:,:,3],loc=hyp[9],
				scale=hyp[10])

			scipy_pdf += stats.beta.logpdf(samples[:,:,4],a=hyp[11],b=hyp[12],
				loc=hyp[13],scale=hyp[14]-hyp[13])
			scipy_pdf += stats.beta.logpdf(samples[:,:,5],a=hyp[15],b=hyp[16],
				loc=hyp[17],scale=hyp[18]-hyp[17])

			scipy_pdf += stats.lognorm.logpdf(samples[:,:,6],scale=np.exp(
				hyp[19]),s=hyp[20])
			scipy_pdf += stats.lognorm.logpdf(samples[:,:,7],scale=np.exp(
				hyp[21]),s=hyp[22])

			return scipy_pdf

		self.assertAlmostEqual(np.max(np.abs(self.hclass.log_p_theta_omega(
			samples,hyp,self.hclass.target_eval_dict)-
			hand_calc_log_pdf(samples,hyp))),0)

		samples = np.random.uniform(size=(2,2,8))*0.3
		self.assertAlmostEqual(np.max(np.abs(self.hclass.log_p_theta_omega(
			samples,hyp,self.hclass.target_eval_dict)-
			hand_calc_log_pdf(samples,hyp))),0)

		hyp = np.array([-1.02,1.5,0.1,0.49*np.pi,10.0,-np.pi,np.pi,0.1,
			0.1,0.1,0.1,3.0,3.0,-0.75,0.75,3.0,3.0,-0.75,0.75,0.6,0.11,0.01,
			0.2])
		self.assertAlmostEqual(np.max(np.abs(self.hclass.log_p_theta_omega(
			samples,hyp,self.hclass.target_eval_dict)-
			hand_calc_log_pdf(samples,hyp))),0)

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

		self.hclass.gen_samples(100)

		# First, make sure all of the values of lens_samps were filled out.
		for pi in range(self.num_params):
			self.assertGreater(np.sum(self.hclass.lens_samps[:,:,pi]),0)

		print(self.lens_params)
		print(self.final_params)

		# Check that the parameters that got changed did so in the right way.
		# First check theta_e
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,-1]-
			np.log(self.hclass.lens_samps[:,:,-1]))),0)
		# Now check the catersian to polar for shears.
		gamma = self.hclass.lens_samps[:,:,0]
		ang = self.hclass.lens_samps[:,:,1]
		g1 = gamma*np.cos(2*ang)
		g2 = gamma*np.sin(2*ang)
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,0]-
			g1)),0)
		self.assertAlmostEqual(np.max(np.abs(self.hclass.predict_samps[:,:,1]-
			g2)),0)

		# TODO: I should also be testing pt_omegai since it's generated here.

		# Clean up the files we generated
		os.remove(self.normalization_constants_path)
		os.remove(self.normalized_param_path)
		os.remove(self.tf_record_path)

	def test_log_post_omega(self):
		# TODO: Make this test
		return


