# TODO: Change the imports once this is a package!!
import unittest
import sys, os, glob
import tensorflow as tf
sys.path.append("../")
import data_tools
import numpy as np
import pandas as pd

class TFRecordTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TFRecordTests, self).__init__(*args, **kwargs)
		self.root_path = 'test_data/'
		self.lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.lens_params_path = self.root_path + 'metadata.csv'
		self.tf_record_path = self.root_path + 'test_record'

	def dataset_comparison(self,dataset,batch_size,num_npy):
		# Run the same test as above
		lens_params_csv = pd.read_csv(self.lens_params_path, index_col=None)
		index_array = []
		npy_counts = 0
		for batch in iter(dataset):
			# Read the image out
			height = batch['height'].numpy()[0]
			width = batch['width'].numpy()[0]
			batch_images = tf.io.decode_raw(batch['image'],
				out_type=np.float32).numpy().reshape(batch_size,
					height,width)
			npy_indexs = batch['index'].numpy()
			lens_params_batch = []
			for lens_param in self.lens_params:
				lens_params_batch.append(batch[lens_param].numpy())
			# Load the original image and lens parameters and make sure that they
			# match
			for batch_index in range(batch_size):
				npy_index = npy_indexs[batch_index]
				index_array.append(npy_index)
				image = batch_images[batch_index]
				original_image = np.load(self.root_path+'X_{0:07d}.npy'.format(
					npy_index+1)).astype(np.float32)
				self.assertEqual(np.sum(np.abs(image-original_image)),0)
				lpi = 0
				for lens_param in self.lens_params:
					lens_param_value = lens_params_batch[lpi][batch_index]
					self.assertAlmostEqual(lens_param_value,lens_params_csv[
						lens_param][npy_index],places=4)
					lpi += 1
				npy_counts += 1
		# Ensure the total number of files is correct
		self.assertEqual(npy_counts,num_npy)

	def test_normalize_lens_parameters(self):
		# Test if normalizing the lens parameters works correctly.
		normalized_param_path = self.root_path + 'normed_metadata.csv'
		normalization_constants_path = self.root_path + 'norm_'
		train_or_test='train'
		data_tools.normalize_lens_parameters(self.lens_params,
			self.lens_params_path,normalized_param_path,
			normalization_constants_path,train_or_test=train_or_test)

		lens_params_csv = pd.read_csv(self.lens_params_path, index_col=None)
		norm_params_csv = pd.read_csv(normalized_param_path, index_col=None)
		norm_constants_csv = pd.read_csv(normalization_constants_path)

		for lens_param in self.lens_params:
			# Assert that the two lists agree once we factor for normalization
			self.assertAlmostEqual(np.sum(np.abs(lens_params_csv[lens_param] - 
				(norm_params_csv[lens_param]*norm_constants_csv[lens_param][1]+
				norm_constants_csv[lens_param][0]))),0)

		# Repeat the same, but with the tet time functionality
		normalized_param_path = self.root_path + 'test_normalization.csv'
		normalization_constants_path = self.root_path + 'norm_'
		train_or_test='test'
		data_tools.normalize_lens_parameters(self.lens_params,
			self.lens_params_path,normalized_param_path,
			normalization_constants_path,train_or_test=train_or_test)

		lens_params_csv = pd.read_csv(self.lens_params_path, index_col=None)
		norm_params_csv = pd.read_csv(normalized_param_path, index_col=None)
		for lens_param in self.lens_params:
			# Assert that the two lists agree once we factor for normalization
			self.assertAlmostEqual(np.sum(np.abs(lens_params_csv[lens_param] - 
				(norm_params_csv[lens_param]*norm_constants_csv[lens_param][1]+
				norm_constants_csv[lens_param][0]))),0)

		# Clean up the file now that we're done
		os.remove(normalized_param_path)	
		os.remove(normalization_constants_path)	

	def test_write_parameters_in_log_space(self):
		# Test if putting the lens parameters in log space works correctly.
		new_lens_params_path = self.root_path + 'metadata_log.csv'
		data_tools.write_parameters_in_log_space(['lens_mass_theta_E'],
			self.lens_params_path,new_lens_params_path)

		lens_params_csv = pd.read_csv(new_lens_params_path, index_col=None)

		self.assertTrue('lens_mass_theta_E_log' in lens_params_csv)
		# Assert that the two parameters agree once we factor for log
		self.assertAlmostEqual(np.sum(np.abs(
			lens_params_csv['lens_mass_theta_E_log'] - 
			np.log(lens_params_csv['lens_mass_theta_E']))),0)

		# Clean up the file now that we're done
		os.remove(new_lens_params_path)	

	def test_ratang_2_exc(self):
		# Test if putting the lens parameters in excentricities works correctly.
		new_lens_params_path = self.root_path + 'metadata_e1e2.csv'
		data_tools.ratang_2_exc('external_shear_gamma_ext',
			'external_shear_psi_ext',self.lens_params_path,new_lens_params_path,
			'external_shear')

		lens_params_csv = pd.read_csv(new_lens_params_path, index_col=None)

		self.assertTrue('external_shear_e1' in lens_params_csv)
		self.assertTrue('external_shear_e2' in lens_params_csv)
		# Assert that the two parameters agree once we factor for log
		rat = lens_params_csv['external_shear_gamma_ext']
		ang = lens_params_csv['external_shear_psi_ext']
		e1 = (1.-rat)/(1.+rat)*np.cos(2*ang)
		e2 = (1.-rat)/(1.+rat)*np.sin(2*ang)
		self.assertAlmostEqual(np.sum(np.abs(e1 - 
			lens_params_csv['external_shear_e1'])),0)
		self.assertAlmostEqual(np.sum(np.abs(e2 - 
			lens_params_csv['external_shear_e2'])),0)

		# Clean up the file now that we're done
		os.remove(new_lens_params_path)	

	def test_generate_tf_record(self):
		# Test that the generate_tf_record code succesfully generates a TFRecord 
		# object and that the images and labels generated correspond to what
		# is in the npy file and the metadata csv.

		# Probe the number of npy files to make sure the total number of files
		# each epoch matches what is expected
		num_npy = len(glob.glob(self.root_path+'*.npy'))

		data_tools.generate_tf_record(self.root_path,self.lens_params,
			self.lens_params_path,self.tf_record_path)

		self.assertTrue(os.path.exists(self.tf_record_path))

		# Open up this TFRecord file and take a look inside
		raw_dataset = tf.data.TFRecordDataset(self.tf_record_path)
		# Define a mapping function to parse the image
		def parse_image(example):
			data_features = {
				'image' : tf.io.FixedLenFeature([],tf.string),
				'height' : tf.io.FixedLenFeature([],tf.int64),
				'width' : tf.io.FixedLenFeature([],tf.int64),
				'index' : tf.io.FixedLenFeature([],tf.int64),
			}
			for lens_param in self.lens_params:
					data_features[lens_param] = tf.io.FixedLenFeature(
						[],tf.float32)
			return tf.io.parse_single_example(example,data_features)
		batch_size = 10
		dataset = raw_dataset.map(parse_image).batch(batch_size)
		self.dataset_comparison(dataset,batch_size,num_npy)

		# Clean up the file now that we're done
		os.remove(self.tf_record_path)

	def test_build_tf_dataset(self):
		# Test that build_tf_dataset has the correct batching behaviour and 
		# returns the same data contained in the npy files and csv.
		num_npy = len(glob.glob(self.root_path+'*.npy'))

		data_tools.generate_tf_record(self.root_path,self.lens_params,
			self.lens_params_path,self.tf_record_path)
		
		# Try batch size 10
		batch_size = 10
		n_epochs = 1
		dataset = data_tools.build_tf_dataset(self.tf_record_path,
			self.lens_params,batch_size,n_epochs)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,100,100])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,8])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Try batch size 5 and n_epochs 2
		batch_size = 5
		n_epochs = 2
		dataset = data_tools.build_tf_dataset(self.tf_record_path,
			self.lens_params,batch_size,n_epochs)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,100,100])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,8])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

