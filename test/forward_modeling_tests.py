import unittest, os
from ovejero import forward_modeling, model_trainer, data_tools
import matplotlib.pyplot as plt
import numpy as np
from baobab import configs
import pandas as pd
import h5py

# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ForwardModelingTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(ForwardModelingTests, self).__init__(*args, **kwargs)
		# Open up the config file.
		# Initialize the class with a test baobab config
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		self.config_path = self.root_path + 'test.json'
		self.baobab_cfg_path = self.root_path + 'test_baobab_cfg.py'

		# Initialize the config
		self.cfg = model_trainer.load_config(self.config_path)

		# Also initialize the baobab config
		self.baobab_cfg = configs.BaobabConfig.from_file(self.baobab_cfg_path)

		# A few bnn_inference testing things that need to be used again here.
		self.lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.normalization_constants_path = self.root_path + 'norms.csv'
		self.tf_record_path = self.root_path + 'tf_record_test'

	def test_class_initialization(self):
		# Just test that the basic variables of the class are initialized as
		# expected.
		fow_model = forward_modeling.ForwardModel(self.cfg)

		# Make sure that the correct lens and source models were initialized
		self.assertEqual(fow_model.ls_lens_model_list,
			self.cfg['forward_mod_params']['lens_model_list'])

		self.assertEqual(fow_model.ls_source_model_list,
			self.cfg['forward_mod_params']['source_model_list'])

		# Test that some basic information in the dictionaries is set
		self.assertEqual(len(fow_model.ls_lens_params),5)
		for lens_dict in fow_model.ls_lens_params:
			self.assertEqual(len(lens_dict),
				len(self.cfg['forward_mod_params']['lens_model_list']))
		self.assertEqual(len(fow_model.ls_source_params),5)
		for source_dict in fow_model.ls_source_params:
			self.assertEqual(len(source_dict),
				len(self.cfg['forward_mod_params']['source_model_list']))

		# Check that the psf kwargs match our expectations
		self.assertEqual(fow_model.ls_kwargs_psf['psf_type'],
			self.baobab_cfg.psf['type'])
		self.assertEqual(fow_model.ls_kwargs_psf['pixel_size'],
			self.baobab_cfg.instrument['pixel_scale'])

		self.assertFalse(fow_model.image_selected)
		self.assertFalse(fow_model.sampler_init)

	def test_select_image(self):
		# Test that the image selection works as intended. The most important
		# thing to test here is that the parameter values and image are what we
		# expect.
		fow_model = forward_modeling.ForwardModel(self.cfg)

		# Use the image index 4
		image_index = 4
		fow_model.select_image(image_index,block=False)
		plt.close()

		# Check that the image is correct
		test_image = np.load(self.root_path + 'X_0000004.npy')
		np.testing.assert_almost_equal(fow_model.true_image,test_image)

		# Check that the noise image is not the same as the test_image but that
		# the difference can be accounted for by noise.
		self.assertFalse(np.array_equal(fow_model.true_image_noise,test_image))
		self.assertLess(np.std(fow_model.true_image_noise-test_image),0.05)

		# Compare the output values and make sure they agree
		df = pd.read_csv(self.root_path+'metadata.csv')
		test_dict = df.loc[image_index].to_dict()
		for key in fow_model.true_values.keys():
			self.assertTrue(key in test_dict.keys())
			self.assertEqual(fow_model.true_values[key],test_dict[key])

		# Check that the data kwargs used by lenstronomy are all set.
		self.assertEqual(len(fow_model.ls_kwargs_data),6)
		np.testing.assert_almost_equal(fow_model.ls_kwargs_data['image_data'],
			fow_model.true_image_noise)

		# Similarly check that multi_band_list has all three dictionaries
		# in it.
		self.assertEqual(len(fow_model.ls_multi_band_list),1)
		for mb_list in fow_model.ls_multi_band_list:
			self.assertEqual(len(mb_list),3)

		self.assertTrue(fow_model.image_selected)

	def test_initialize_sampler(self):
		# Test that we correctly initialize the lenstronomy fitting sequence
		# sampler.
		fow_model = forward_modeling.ForwardModel(self.cfg)

		# If no image has been selected, initializing the sampler should
		# raise an error.
		walker_ratio = 10
		chains_save_path = self.root_path + 'test_chains.h5'
		with self.assertRaises(RuntimeError):
			fow_model.initialize_sampler(walker_ratio,chains_save_path)

		# Now select an image and run the same function.
		image_index = 4
		fow_model.select_image(image_index,block=False)
		plt.close()
		fow_model.initialize_sampler(walker_ratio,chains_save_path)

		self.assertTrue(fow_model.sampler_init)

	def test_run_sampler(self):
		# Test that running the sampler for a few iterations doesn't return
		# any errors. Since all the work is being done by the lenstronomy
		# fitting sequence, these tests are light.
		fow_model = forward_modeling.ForwardModel(self.cfg)
		image_index = 4
		fow_model.select_image(image_index,block=False)
		plt.close()

		# If the sampler hasn't been initialized, this should throw an error
		n_samps = 5
		with self.assertRaises(RuntimeError):
			fow_model.run_sampler(n_samps)

		# Repeat after the sampler has been initialized
		walker_ratio = 3
		chains_save_path = self.root_path + 'test_chains.h5'
		fow_model.initialize_sampler(walker_ratio,chains_save_path)
		fow_model.run_sampler(n_samps)
		# Check that restarting the samples doesn't cause any issues
		fow_model.run_sampler(n_samps)

		# First check that the h5 file was saved correctly.
		n_params = 14
		chains_h5_file = h5py.File(chains_save_path,'r')
		self.assertTrue('lenstronomy_mcmc_emcee' in chains_h5_file.keys())
		self.assertEqual(
			chains_h5_file['lenstronomy_mcmc_emcee']['chain'].shape,
			(n_samps*2,n_params*walker_ratio,n_params))

		# Make sure we got the right chains.
		np.testing.assert_almost_equal(fow_model.chains,
			chains_h5_file['lenstronomy_mcmc_emcee']['chain'])
		self.assertEqual(len(fow_model.chain_params),n_params)

		os.remove(chains_save_path)

	def test_plot_chains(self):
		# Check that the plot chains function does not crash.
		fow_model = forward_modeling.ForwardModel(self.cfg)
		image_index = 4
		fow_model.select_image(image_index,block=False)
		plt.close()
		walker_ratio = 3
		chains_save_path = self.root_path + 'test_chains.h5'
		fow_model.initialize_sampler(walker_ratio,chains_save_path)
		n_samps = 5
		fow_model.run_sampler(n_samps)
		fow_model.plot_chains(burnin=None,block=False)
		plt.close()
		fow_model.plot_chains(burnin=2,block=False)
		plt.close()
		os.remove(chains_save_path)

	def test_correct_chains(self):
		# Test that the internal _correct_chains function applies the
		# transformations from the dataset function.
		# Generate a fake set of chains
		chains = np.random.uniform(size=(100,8))
		new_chains = np.copy(chains)
		true_values = np.ones(8)
		new_true_values = np.copy(true_values)
		params = np.array(['lens_mass_theta_E', 'lens_mass_gamma', 'lens_mass_e1',
			'lens_mass_e2', 'lens_mass_center_x', 'lens_mass_center_y',
			'external_shear_gamma_ext', 'external_shear_psi_ext'])

		# Initialize our forward model
		fow_model = forward_modeling.ForwardModel(self.cfg)

		# Run our internal function
		new_params = fow_model._correct_chains(new_chains,params,new_true_values)

		# Check that the parameters that shouldn't be changed weren't
		unchanged_params = ['lens_mass_gamma', 'lens_mass_e1','lens_mass_e2',
			'lens_mass_center_x', 'lens_mass_center_y']
		for param in unchanged_params:
			np.testing.assert_almost_equal(chains[:,params==param],
				new_chains[:,params==param])
			np.testing.assert_almost_equal(true_values[params==param],
				new_true_values[params==param])

		# Check the log parameter
		np.testing.assert_almost_equal(np.log(
			chains[:,params=='lens_mass_theta_E']),
			new_chains[:,new_params=='lens_mass_theta_E_log'])
		np.testing.assert_almost_equal(np.log(
			true_values[params=='lens_mass_theta_E']),
			new_true_values[new_params=='lens_mass_theta_E_log'])

		# Finally check the parameters that cartesian parameters are cartesian
		gamma = chains[:,params=='external_shear_gamma_ext']
		ang = chains[:,params=='external_shear_psi_ext']
		g1 = gamma*np.cos(2*ang)
		g2 = gamma*np.sin(2*ang)
		np.testing.assert_almost_equal(g1,
			new_chains[:,new_params=='external_shear_g1'])
		np.testing.assert_almost_equal(g2,
			new_chains[:,new_params=='external_shear_g2'])
		gamma = true_values[params=='external_shear_gamma_ext']
		ang = true_values[params=='external_shear_psi_ext']
		g1 = gamma*np.cos(2*ang)
		g2 = gamma*np.sin(2*ang)
		np.testing.assert_almost_equal(g1,
			new_true_values[new_params=='external_shear_g1'])
		np.testing.assert_almost_equal(g2,
			new_true_values[new_params=='external_shear_g2'])

	def test_plot_posterior_contours(self):
		# Check that nothing crashes when we try to plot the posterior
		# contours
		# Check that the plot chains function does not crash.
		fow_model = forward_modeling.ForwardModel(self.cfg)
		image_index = 4
		fow_model.select_image(image_index,block=False)
		plt.close()
		walker_ratio = 3
		chains_save_path = self.root_path + 'test_chains.h5'
		fow_model.initialize_sampler(walker_ratio,chains_save_path)
		n_samps = 20
		fow_model.run_sampler(n_samps)
		burnin = 0
		num_samples = 20

		model_trainer.prepare_tf_record(self.cfg, self.root_path,
			self.tf_record_path,self.lens_params,'train')
		dpi = 20
		fow_model.plot_posterior_contours(burnin,num_samples,dpi=dpi,
			block=False)
		plt.close()

		os.remove(self.normalization_constants_path)
		os.remove(self.tf_record_path)
		os.remove(self.root_path + self.cfg['dataset_params']['new_param_path'])
		os.remove(chains_save_path)
