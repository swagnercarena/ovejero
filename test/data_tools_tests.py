# TODO: Change the imports once this is a package!!
import unittest
import os, glob
# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from ovejero import data_tools
import numpy as np
import pandas as pd
from helpers import dataset_comparison

class TFRecordTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TFRecordTests, self).__init__(*args, **kwargs)
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		self.lens_params = ['external_shear_gamma_ext','external_shear_psi_ext',
			'lens_mass_center_x','lens_mass_center_y',
			'lens_mass_e1','lens_mass_e2',
			'lens_mass_gamma','lens_mass_theta_E']
		self.lens_params_path = self.root_path + 'metadata.csv'
		self.tf_record_path = self.root_path + 'test_record'

	def test_normalize_lens_parameters(self):
		# Test if normalizing the lens parameters works correctly.
		normalized_param_path = self.root_path + 'normed_metadata.csv'
		normalization_constants_path = self.root_path + 'norm.csv'
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
		normalization_constants_path = self.root_path + 'norm.csv'
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
		num_npy = len(glob.glob(self.root_path+'X*.npy'))

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
		dataset_comparison(self,dataset,batch_size,num_npy)

		# Clean up the file now that we're done
		os.remove(self.tf_record_path)

	def test_build_tf_dataset(self):
		# Test that build_tf_dataset has the correct batching behaviour and 
		# returns the same data contained in the npy files and csv.
		num_npy = len(glob.glob(self.root_path+'X*.npy'))

		data_tools.generate_tf_record(self.root_path,self.lens_params,
			self.lens_params_path,self.tf_record_path)
		
		# Try batch size 10
		batch_size = 10
		n_epochs = 1
		norm_images = False
		dataset = data_tools.build_tf_dataset(self.tf_record_path,
			self.lens_params,batch_size,n_epochs,norm_images=norm_images)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,128,128,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,8])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Try batch size 5 and n_epochs 2
		batch_size = 5
		n_epochs = 2
		dataset = data_tools.build_tf_dataset(self.tf_record_path,
			self.lens_params,batch_size,n_epochs,norm_images=norm_images)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,128,128,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,8])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Try normalizing the data
		batch_size = 5
		n_epochs = 2
		norm_images=True
		dataset = data_tools.build_tf_dataset(self.tf_record_path,
			self.lens_params,batch_size,n_epochs,norm_images=norm_images)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,128,128,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,8])
			for image in batch[0].numpy():
				self.assertAlmostEqual(np.std(image),1,places=4)
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Try raising a shift error
		shift_pixels = 2
		with self.assertRaises(RuntimeError):
			dataset_shifted = data_tools.build_tf_dataset(self.tf_record_path,
				self.lens_params,batch_size,n_epochs,norm_images=norm_images,
				shift_pixels=shift_pixels)

		# Try raising a different shift error
		shift_pixels = 2
		shift_params = (['lens_mass_center_x'],['lens_mass_center_y'])
		with self.assertRaises(RuntimeError):
			dataset_shifted = data_tools.build_tf_dataset(self.tf_record_path,
				self.lens_params,batch_size,n_epochs,norm_images=norm_images,
				shift_pixels=shift_pixels,shift_params=shift_params)

		# Try doing the shifting right
		batch_size = 20
		n_epochs = 1
		shift_pixels = 8
		# Set the pixel scales to be different to make sure it works as expected
		normed_pixel_scale = {'lens_mass_center_x':0.051,
			'lens_mass_center_y':0.062}
		shift_params = (['lens_mass_center_x'],['lens_mass_center_y'])
		tf.random.set_seed(20)
		dataset_shifted = data_tools.build_tf_dataset(self.tf_record_path,
				self.lens_params,batch_size,n_epochs,norm_images=norm_images,
				shift_pixels=shift_pixels, shift_params=shift_params,
				normed_pixel_scale=normed_pixel_scale)
		tf.random.set_seed(20)
		dataset = data_tools.build_tf_dataset(self.tf_record_path,
				self.lens_params,batch_size,n_epochs,norm_images=norm_images)
		npy_counts = 0
		for batch in dataset:
			for batch_shifted in dataset_shifted:
				self.assertListEqual(batch[0].get_shape().as_list(),
					[batch_size,128,128,1])
				self.assertListEqual(batch[1].get_shape().as_list(),
					[batch_size,8])
				for image in batch[0].numpy():
					self.assertAlmostEqual(np.std(image),1,places=4)
				self.assertListEqual(batch_shifted[0].get_shape().as_list(),
					[batch_size,128,128,1])
				self.assertListEqual(batch_shifted[1].get_shape().as_list(),
					[batch_size,8])
				for image in batch_shifted[0].numpy():
					self.assertAlmostEqual(np.std(image),1,places=4)

				for image_i in range(len(batch_shifted[0].numpy())):
					image = batch[0].numpy()[image_i]
					image_shifted = batch_shifted[0].numpy()[image_i]
					# We'll have to compare the new and old positions to see what
					# type of shift we caused.
					x_pos,y_pos = batch[1].numpy()[image_i,2:4]
					x_pos_shifted,y_pos_shifted = batch_shifted[1].numpy()[
						image_i,2:4]
					x_pix_shift = int(round((x_pos_shifted - x_pos)/
						normed_pixel_scale['lens_mass_center_x']))
					y_pix_shift = int(round((y_pos_shifted - y_pos)/
						normed_pixel_scale['lens_mass_center_y']))
					# Now compare the two image with the shift canceled out.
					if x_pix_shift > 0:
						image = image[:,:-x_pix_shift]
						image_shifted = image_shifted[:,x_pix_shift:]
					elif x_pix_shift < 0:
						image = image[:,-x_pix_shift:]
						image_shifted = image_shifted[:,:x_pix_shift]
					if y_pix_shift > 0:
						image = image[:-y_pix_shift]
						image_shifted = image_shifted[y_pix_shift:]
					elif y_pix_shift < 0:
						image = image[-y_pix_shift:]
						image_shifted = image_shifted[:y_pix_shift]
					self.assertAlmostEqual(np.sum(np.abs(image-image_shifted)),0,
						places=2)
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)
