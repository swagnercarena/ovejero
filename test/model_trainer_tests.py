import unittest, json, glob, os, sys, shutil
# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
from ovejero import model_trainer, data_tools
from helpers import dataset_comparison


class DataPrepTests(unittest.TestCase):

	def setUp(self):
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		self.lens_params = ['external_shear_g1','external_shear_g2',
			'lens_mass_center_x','lens_mass_center_y','lens_mass_e1',
			'lens_mass_e2','lens_mass_gamma','lens_mass_theta_E_log']
		self.lens_params_path = self.root_path + 'new_metadata.csv'
		self.tf_record_path = self.root_path + 'tf_record_test'

	def test_config_checker(self):
		# Test that the config checker doesn't fail correct configuration files
		# and does fail configs with missing fields.
		with open(self.root_path+'test.json','r') as json_f:
			cfg = json.load(json_f)
		model_trainer.config_checker(cfg)

		del cfg['training_params']
		with self.assertRaises(RuntimeError):
			model_trainer.config_checker(cfg)

		with open(self.root_path+'test.json','r') as json_f:
			cfg = json.load(json_f)
		del cfg['training_params']['kernel_regularizer']
		with self.assertRaises(RuntimeError):
			model_trainer.config_checker(cfg)
		self.assertTrue('training_params' in cfg)

		with open(self.root_path+'test.json','r') as json_f:
			cfg = json.load(json_f)
		del cfg['validation_params']
		with self.assertRaises(RuntimeError):
			model_trainer.config_checker(cfg)

		with open(self.root_path+'test.json','r') as json_f:
			cfg = json.load(json_f)
		del cfg['dataset_params']['gampsi']['gampsi_parameter_prefixes']
		with self.assertRaises(RuntimeError):
			model_trainer.config_checker(cfg)
		self.assertTrue('dataset_params' in cfg)
		self.assertTrue('gampsi' in cfg['dataset_params'])

	def test_load_config(self):
		# Test that load config returns a config file and fails the config check
		# when it should.
		cfg = model_trainer.load_config(self.root_path+'test.json')

		del cfg['validation_params']
		temp_cfg_path = self.root_path + 'temp.json'

		with open(temp_cfg_path,'w') as json_f:
			json.dump(cfg,json_f,indent=4)

		with self.assertRaises(RuntimeError):
			model_trainer.load_config(temp_cfg_path)

		os.remove(temp_cfg_path)

	def test_prepare_tf_record(self):
		# Test that the prepare_tf_record function works as expected.
		with open(self.root_path+'test.json','r') as json_f:
			cfg = json.load(json_f)
		with self.assertRaises(ValueError):
			train_or_test='test'
			model_trainer.prepare_tf_record(cfg, self.root_path,
				self.tf_record_path,self.lens_params,train_or_test)
		train_or_test='train'
		model_trainer.prepare_tf_record(cfg, self.root_path, self.tf_record_path,
			self.lens_params,train_or_test)

		# Check the TFRecord and make sure the number of parameters and values
		# all seems reasonable.
		num_npy = len(glob.glob(self.root_path+'X*.npy'))
		self.assertTrue(os.path.exists(self.tf_record_path))

		# Open up this TFRecord file and take a look inside
		raw_dataset = tf.data.TFRecordDataset(self.tf_record_path)

		# Define a mapping function to parse the image
		def parse_image(example):
			data_features = {
				'image': tf.io.FixedLenFeature([],tf.string),
				'height': tf.io.FixedLenFeature([],tf.int64),
				'width': tf.io.FixedLenFeature([],tf.int64),
				'index': tf.io.FixedLenFeature([],tf.int64),
			}
			for lens_param in self.lens_params:
				data_features[lens_param] = tf.io.FixedLenFeature(
					[],tf.float32)
			return tf.io.parse_single_example(example,data_features)
		batch_size = 10
		dataset = raw_dataset.map(parse_image).batch(batch_size)
		dataset_comparison(self,dataset,batch_size,num_npy)

		train_or_test='test'
		model_trainer.prepare_tf_record(cfg, self.root_path, self.tf_record_path,
			self.lens_params,train_or_test)

		# Check the TFRecord and make sure the number of parameters and values
		# all seems reasonable.
		num_npy = len(glob.glob(self.root_path+'X*.npy'))
		self.assertTrue(os.path.exists(self.tf_record_path))

		# Open up this TFRecord file and take a look inside
		raw_dataset = tf.data.TFRecordDataset(self.tf_record_path)
		batch_size = 10
		dataset = raw_dataset.map(parse_image).batch(batch_size)
		dataset_comparison(self,dataset,batch_size,num_npy)

		# Clean up the file now that we're done
		os.remove(self.tf_record_path)
		os.remove(self.root_path+'new_metadata.csv')
		os.remove(self.root_path+'norms.csv')

	def test_model_loss_builder_gmm(self):
		# Test that the model and loss returned from model_loss_builder
		# agree with what is expected.
		cfg = model_trainer.load_config(self.root_path+'test.json')
		cfg['training_params']['dropout_type'] = 'concrete'
		final_params = cfg['training_params']['final_params']
		num_params = len(final_params)

		tf.keras.backend.clear_session()

		model, loss = model_trainer.model_loss_builder(cfg)
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,2*(num_params +
			int(num_params*(num_params+1)/2))+1))
		yptf = tf.constant(y_pred,dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)

		# Check that the loss function has the right dimensions. More rigerous
		# tests of the loss function can be found in the test_bnn_alexnet.
		loss(yttf,yptf)
		self.assertEqual(len(model.layers),13)
		self.assertEqual(model.layers[-1].output_shape[-1],y_pred.shape[-1])

	def test_model_loss_builder_full(self):
		# Test that the model and loss returned from model_loss_builder
		# agree with what is expected.
		cfg = model_trainer.load_config(self.root_path+'test.json')
		cfg['training_params']['dropout_type'] = 'concrete'
		final_params = cfg['training_params']['final_params']
		num_params = len(final_params)

		tf.keras.backend.clear_session()

		cfg['training_params']['bnn_type'] = 'full'
		model, loss = model_trainer.model_loss_builder(cfg)
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,num_params + int(num_params*(num_params+1)/2)))
		yptf = tf.constant(y_pred,dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)

		# Check that the loss function has the right dimensions. More rigerous
		# tests of the loss function can be found in the test_bnn_alexnet.
		loss(yttf,yptf)
		self.assertEqual(len(model.layers),13)
		self.assertEqual(model.layers[-1].output_shape[-1],y_pred.shape[-1])

	def test_model_loss_builder_diag(self):
		# Test that the model and loss returned from model_loss_builder
		# agree with what is expected.
		cfg = model_trainer.load_config(self.root_path+'test.json')
		cfg['training_params']['dropout_type'] = 'concrete'
		final_params = cfg['training_params']['final_params']
		num_params = len(final_params)

		tf.keras.backend.clear_session()

		cfg['training_params']['bnn_type'] = 'diag'
		model, loss = model_trainer.model_loss_builder(cfg)
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,2*num_params))
		yptf = tf.constant(y_pred,dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)

		# Check that the loss function has the right dimensions. More rigerous
		# tests of the loss function can be found in the test_bnn_alexnet.
		loss(yttf,yptf)
		self.assertEqual(len(model.layers),13)
		self.assertEqual(model.layers[-1].output_shape[-1],y_pred.shape[-1])

	def test_model_loss_builder_diag_stand(self):
		# Test that the model and loss returned from model_loss_builder
		# agree with what is expected.
		cfg = model_trainer.load_config(self.root_path+'test.json')
		cfg['training_params']['dropout_type'] = 'concrete'
		final_params = cfg['training_params']['final_params']
		num_params = len(final_params)

		tf.keras.backend.clear_session()

		cfg['training_params']['bnn_type'] = 'diag'
		cfg['training_params']['dropout_type'] = 'standard'
		model, loss = model_trainer.model_loss_builder(cfg)
		y_true = np.ones((1,num_params))
		y_pred = np.ones((1,2*num_params))
		yptf = tf.constant(y_pred,dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)

		# Check that the loss function has the right dimensions. More rigerous
		# tests of the loss function can be found in the test_bnn_alexnet.
		loss(yttf,yptf)
		self.assertEqual(len(model.layers),21)
		self.assertEqual(model.layers[-1].output_shape[-1],y_pred.shape[-1])

	def test_get_normed_pixel_scale(self):
		# Test if get_normed_pixel scale rescales the pixel_scale as we would
		# expect.
		cfg = model_trainer.load_config(self.root_path+'test.json')
		# The original pixel scale
		pixel_scale = 0.051
		# Test if normalizing the lens parameters works correctly.
		normalized_param_path = self.root_path + 'normed_metadata.csv'
		normalization_constants_path = self.root_path + 'norms.csv'
		train_or_test='train'
		lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		lens_params_path = self.root_path + 'metadata.csv'
		data_tools.normalize_lens_parameters(lens_params,
			lens_params_path,normalized_param_path,
			normalization_constants_path,train_or_test=train_or_test)

		# New pixel scale
		normed_pixel_scale = model_trainer.get_normed_pixel_scale(cfg,
			pixel_scale)

		lens_params_csv = pd.read_csv(lens_params_path, index_col=None)
		norm_params_csv = pd.read_csv(normalized_param_path, index_col=None)

		self.assertAlmostEqual(np.std(lens_params_csv['lens_mass_center_x'] /
			pixel_scale - norm_params_csv['lens_mass_center_x'] /
			normed_pixel_scale['lens_mass_center_x']),0)
		self.assertAlmostEqual(np.std(lens_params_csv['lens_mass_center_y'] /
			pixel_scale - norm_params_csv['lens_mass_center_y'] /
			normed_pixel_scale['lens_mass_center_y']),0)

		# Clean up the file now that we're done
		os.remove(normalized_param_path)
		os.remove(normalization_constants_path)

	def test_main(self):
		# Test that the main function works.

		# Make a copy of the previous test config file with fewer epochs
		with open(self.root_path+'test.json') as json_file:
			old_config = json.load(json_file)
		old_config['training_params']['n_epochs'] = 1
		with open(self.root_path+'test_temp.json','w') as json_file:
			json.dump(old_config,json_file)

		sys.argv = ['model_trainer',self.root_path+'test_temp.json']
		model_trainer.main()

		# Check that the expected directories were created
		self.assertTrue(os.path.isfile(self.root_path+'new_metadata.csv'))
		self.assertTrue(os.path.isfile(self.root_path+'norms.csv'))
		self.assertTrue(os.path.isfile(self.root_path+'tf_record_test'))
		self.assertTrue(os.path.isfile(self.root_path+'tf_record_test_val'))
		self.assertTrue(os.path.isfile(self.root_path+'test_model.h5'))

		# Clean up from the model training
		os.remove(self.root_path+'new_metadata.csv')
		os.remove(self.root_path+'norms.csv')
		os.remove(self.root_path+'tf_record_test')
		os.remove(self.root_path+'tf_record_test_val')
		os.remove(self.root_path+'test_model.h5')
		os.remove(self.root_path+'test_temp.json')
		shutil.rmtree(self.root_path + 'test.log')
