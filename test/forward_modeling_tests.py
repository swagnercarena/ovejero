import unittest, os
from ovejero import forward_modeling, model_trainer
import matplotlib.pyplot as plt
import numpy as np
from baobab import configs
from baobab.sim_utils import instantiate_PSF_models, get_PSF_model, \
	generate_image
import pandas as pd


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

	def test_class_initialization(self):
		# Just test that the basic variables of the class are initialized as
		# expected.
		fow_model = forward_modeling.ForwardModel(self.cfg)

		self.assertFalse(fow_model.image_selected)
		self.assertFalse(fow_model.sampler_init)

	def test_select_image(self):
		# Test that the image selection works as intended. The most important
		# thing to test here is that the PSF selection, parameter values,
		# and image are what we expect.
		fow_model = forward_modeling.ForwardModel(self.cfg)
		psf_models = instantiate_PSF_models(self.baobab_cfg.psf,
			self.baobab_cfg.instrument.pixel_scale)

		# Use the image index 4
		image_index = 4
		fow_model.select_image(image_index,block=False)
		plt.close()

		# Check that the psf model is what we expect
		test_psf = psf_models[4]
		np.testing.assert_equal(fow_model.psf_model.kernel_point_source,
			test_psf.kernel_point_source)

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
		for key in fow_model.image_dict.keys():
			self.assertTrue(key in test_dict.keys())
			self.assertEqual(fow_model.image_dict[key],test_dict[key])

		# Finally check that the emcee_initial_values are correct and sorted by
		# the params_list.
		test_array = np.array([5.27842949e-02,1.34493664e-02,-7.15014652e-02,
			-1.23646169e-01,6.96736656e-02, 6.85339174e-02, 2.03898627e+00,
			8.50127074e-01, 6.59991409e-01, -6.28755531e-02, 3.39002899e-02,
			-8.00603754e-02, 2.63076203e-01, 1.92940327e+01, 2.19642026e+00])
		np.testing.assert_almost_equal(fow_model.emcee_initial_values,
			test_array)

		self.assertTrue(fow_model.image_selected)
