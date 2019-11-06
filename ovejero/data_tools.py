# -*- coding: utf-8 -*-
"""
Manipulate the baobab data and prepare it for the model.

This module contains functions that will normalize and reparametrize the data.
It also contains the functions neccesary to build a TFDataset that can be used
for efficient parallelization in training.

See the script model_trainer.py for examples of how to use these functions.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import glob
from tqdm import tqdm

def normalize_lens_parameters(lens_params,lens_params_path,normalized_param_path,
	normalization_constants_path,train_or_test='train'):
	"""
	Normalize the lens parameters such that they have mean 0 and standard
	deviation 1.

	Parameters
	----------
		lens_params ([str,....]): A list of strings containing the lens params 
			that should be written out as features
		lens_params_path (str):  The path to the csv file containing the lens 
			parameters
		normalized_param_path (str): The path to the csv file where the 
			normalized parameters will be written
		normalization_constants_path (str): The path to the csv file where the
			mean and std used for normalization will be written / read
		train_or_test (str): Whether this is a train time or test time operation.
			At test time the normalization values will be read from the
			normalization constants file instead of written to it.
	"""
	# Read the lens parameters from the csv file
	lens_params_csv = pd.read_csv(lens_params_path, index_col=None)
	# Initialize the data structures that will contain our normalized data
	# and the normalization constants
	norm_dict = {'img_filename':lens_params_csv['img_filename']}

	# If this if for train, we must save the normalization constants
	if train_or_test == 'train':
		norm_const_dict = {'constant':['means','std']}
	else:
		norm_const_dict = pd.read_csv(normalization_constants_path, 
			index_col=None)

	for lens_param in lens_params:
		# Store the normalized data and constants
		if train_or_test == 'train':
			norm_const_dict[lens_param] = [np.mean(lens_params_csv[lens_param]),
				np.std(lens_params_csv[lens_param])]
		norm_dict[lens_param] = ((lens_params_csv[lens_param]-
			norm_const_dict[lens_param][0])/norm_const_dict[lens_param][1])
	# Turn data into a DataFrame to save as csv
	df = pd.DataFrame(data=norm_dict)
	# Don't include an index to be consistent with baobab csv files.
	df.to_csv(normalized_param_path,index=False)
	# Repeat the same for the mean and std information
	df_const = pd.DataFrame(data=norm_const_dict)
	df_const.to_csv(normalization_constants_path,index=False)

def write_parameters_in_log_space(lens_params,lens_params_path,
	new_lens_params_path):
	"""
	Convert lens parameters to log space (important for parameters that cannot 
	be negative)

	Parameters
	----------
		lens_params ([str,...]): The parameters that will be convereted to log 
			space
		lens_params_path (str):  The path to the csv file containing the lens 
			parameters
		new_lens_params_path (str): The path to the csv file where the old 
			parameters and the log parameter will be written. Can be the same as 
			lens_params_path

	Notes
	-----
		New values of parameters will be written to csv file with the name
		'lens parameter name'_log
	"""
	# Read the lens parameters from the csv file
	lens_params_csv = pd.read_csv(lens_params_path, index_col=None)

	for lens_param in lens_params:
		lens_params_csv[lens_param+'_log'] = np.log(lens_params_csv[lens_param])

	# Don't include an index to be consistent with baobab csv files.
	lens_params_csv.to_csv(new_lens_params_path,index=False)

def ratang_2_exc(lens_param_rat,lens_param_ang,lens_params_path,
	new_lens_params_path,new_lens_parameter_prefix):
	"""
	Convert one lens parameter pair of ratio and angle to excentricities.

	Parameters
	----------
		lens_param_rat (str): The ratio parameter name
		lens_param_ang (str): The angle parameter name
		lens_params_path (str):  The path to the csv file containing the lens 
			parameters
		new_lens_params_path (str): The path to the csv file where the old 
			parameters and the new excentricities will be written
		new_lens_parameter_prefix (str): The prefix for the new lens parameter 
			name (for example external_shear)

	Notes
	-------
		New values of parameters will be written to csv file with the names
		'lens new_lens_parameter_prefix name'_e1/e2
	"""
	# Read the lens parameters from the csv file
	lens_params_csv = pd.read_csv(lens_params_path, index_col=None)

	# Calcualte the value of these parameters from their ratio and angle
	rat = lens_params_csv[lens_param_rat]
	ang = lens_params_csv[lens_param_ang]
	e1 = (1.-rat)/(1.+rat)*np.cos(2*ang)
	e2 = (1.-rat)/(1.+rat)*np.sin(2*ang)

	# Save the values to the new csv (which may also be the old csv)
	lens_params_csv[new_lens_parameter_prefix+'_e1'] = e1
	lens_params_csv[new_lens_parameter_prefix+'_e2'] = e2
	# Don't include an index to be consistent with baobab csv files.
	lens_params_csv.to_csv(new_lens_params_path,index=False)

def generate_tf_record(root_path,lens_params,lens_params_path,tf_record_path):
	"""
	Generate a TFRecord file from a directory of numpy files.

	Parameters
	----------
		root_path (str): The path to the folder containing all of the numpy files
		lens_params (str): A list of strings containing the lens params that 
			should be written out as features
		lens_params_path (str):  The path to the csv file containing the lens 
			parameters
		tf_record_path (str): The path to which the tf_record will be saved
	"""
	# Pull the list of numpy filepaths from the directory
	npy_file_list =  glob.glob(root_path+'X*.npy')
	# Open label csv
	lens_params_csv = pd.read_csv(lens_params_path, index_col=None)
	# Initialize the writer object and write the lens data
	with tf.io.TFRecordWriter(tf_record_path) as writer:
		for npy_file in tqdm(npy_file_list):
			# Pull the index from the filename
			index = int(npy_file[-11:-4])
			image_shape = np.load(npy_file).shape
			# The image must be converted to a tf string feature
			image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
				value=[np.load(npy_file).astype(np.float32).tostring()]))
			# Initialize a feature dictionary with the image, the height,
			# and the width
			feature = {
				'image' : image_feature,
				'height': tf.train.Feature(
					int64_list=tf.train.Int64List(value=[image_shape[0]])),
				'width': tf.train.Feature(
					int64_list=tf.train.Int64List(value=[image_shape[1]])),
				'index': tf.train.Feature(
					int64_list=tf.train.Int64List(value=[index]))
			}
			# Add all of the lens parameters to the feature dictionary
			for lens_param in lens_params:
				feature[lens_param] = tf.train.Feature(
					float_list=tf.train.FloatList(
						value=[lens_params_csv[lens_param][index]]))
			# Create the tf example object
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			# Write out the example to the TFRecord file
			writer.write(example.SerializeToString())

def build_tf_dataset(tf_record_path,lens_params,batch_size,n_epochs):
	"""
	Return a TFDataset for use in training the model.

	Parameters
	----------
		tf_record_path (str): The path to the TFRecord file that will be turned
			into a TFDataset
		lens_params ([str,...]): A list of strings containing the lens params 
			that were written out as features
		batch_size (int): The batch size that will be used for training
		n_epochs (int): The number of training epochs. The dataset object will deal
			with iterating over the data for repeated epochs.

	Returns
	-------
		(tf.TFDataset): A TFDataset object for use in training

	"""

	# Read the TFRecord
	raw_dataset = tf.data.TFRecordDataset(tf_record_path)

	# Create the feature decoder that will be used
	def parse_image_features(example):
		data_features = {
			'image' : tf.io.FixedLenFeature([],tf.string),
			'height' : tf.io.FixedLenFeature([],tf.int64),
			'width' : tf.io.FixedLenFeature([],tf.int64),
			'index' : tf.io.FixedLenFeature([],tf.int64),
		}
		for lens_param in lens_params:
				data_features[lens_param] = tf.io.FixedLenFeature(
					[],tf.float32)
		parsed_dataset = tf.io.parse_single_example(example,data_features)
		parsed_dataset = tf.io.parse_single_example(example,data_features)
		image = tf.io.decode_raw(parsed_dataset['image'],out_type=float)
		image = tf.reshape(image,(parsed_dataset['height'],
			parsed_dataset['width'],1))
		lens_param_values = tf.stack([parsed_dataset[lens_param] for lens_param 
			in lens_params])
		return image,lens_param_values

	# Select the buffer size to be slightly larger than the batch
	buffer_size = int(batch_size*1.2)

	# Set the feature decoder as the mapping function. Drop the remainder
	# in the case that batch_size does not divide the number of training
	# points exactly
	dataset = raw_dataset.map(parse_image_features).repeat(n_epochs).shuffle(
		buffer_size=buffer_size).batch(batch_size)
	return dataset

