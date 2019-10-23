import numpy as np
import tensorflow as tf
import pandas as pd
import glob

def normalize_lens_parameters(lens_params,lens_params_path,normalized_param_path,
	normalization_constants_path,train_or_test='train'):
	"""
	Normalize the lens parameters such that they have mean 0 and standard
	deviation 1.

	Parameters:
		lens_params: A list of strings containing the lens params that should
			be written out as features
		lens_params_path:  The path to the csv file containing the lens parameters
		normalized_param_path: The path to the csv file where the normalized
			parameters will be written
		normalization_constants_path: The path to the csv file where the
			mean and std used for normalization will be written / read
		train_or_test: Whether this is a train time or test time operation.
			At test time the normalization values will be read from the
			normalization constants file instead of written to it.

	Returns:
		None
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

def generate_tf_record(root_path,lens_params,lens_params_path,tf_record_path):
	"""
	Generates a TFRecord file from a directory of numpy files.

	Parameters:
		root_path: The path to the folder containing all of the numpy files
		lens_params: A list of strings containing the lens params that should
			be written out as features
		lens_params_path:  The path to the csv file containing the lens parameters
		tf_record_path: The path to which the tf_record will be saved

	Returns:
		None
	"""
	# Pull the list of numpy filepaths from the directory
	npy_file_list =  glob.glob(root_path+'*.npy')
	# Open label csv
	lens_params_csv = pd.read_csv(lens_params_path, index_col=None)
	# Initialize the writer object and write the lens data
	with tf.io.TFRecordWriter(tf_record_path) as writer:
		for npy_file in npy_file_list:
			# Pull the index from the filename (assume things are indexed from 1)
			index = int(npy_file[-11:-4])-1
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
	Returns a TFDataset for use in training the model.

	Parameters:
		tf_record_path: The path to the TFRecord file that will be turned
			into a TFDataset
		lens_params: A list of strings containing the lens params that were
			written out as features
		batch_size: The batch size that will be used for training
		n_epochs: The number of training epochs. The dataset object will deal
			with iterating over the data for repeated epochs.

	Returns:
		A TFDataset object for us in training

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
			parsed_dataset['width']))
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

