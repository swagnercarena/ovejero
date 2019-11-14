# -*- coding: utf-8 -*-
"""
This script will initialize and train a BNN model on a strong lensing image
dataset.

Examples
--------
python -m model_trainer configs/t1.json

"""

# Import some backend stuff
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import argparse, json, os
import pandas as pd

# Import the code to construct the bnn and the data pipeline
from ovejero import bnn_alexnet, data_tools

def config_checker(cfg):
	"""
	Check that configuration file meets ovejero requirements. Throw an error
	if configuration file is invalid.

	Parameters
	----------
	cfg: The dictionary attained from reading the json config.
	"""

	def recursive_key_checker(dict_check,dict_ref):
		"""
		Check that dictionary has all of the keys in a reference dictionary, and
		that the same is true for any sub-dictionaries. Raise an error if not
		identical.

		Parameters
		----------
			dict_check (dict): The dictionary to check
			dict_ref (dict): The reference dictionary

		"""
		for key in dict_ref:
			if key not in dict_check:
				raise RuntimeError('Input config does not contain %s'%(key))
			if isinstance(dict_ref[key],dict):
				recursive_key_checker(dict_check[key],dict_ref[key])

	# Load the check json file
	root_path = os.path.dirname(os.path.abspath(__file__))
	with open(root_path+'/check.json','r') as json_f:
		cfg_ref = json.load(json_f)

	recursive_key_checker(cfg,cfg_ref)

def load_config(config_path):
	"""
	Load a configuration file from the path and check that it meets the 
	requirements.

	Parameters
	----------
		config_path (str): The path to the config file to be loaded 

	Returns
	-------
		(dict): A dictionary object with the config file.
	"""
	# Load the config
	with open(config_path,'r') as json_f:
		cfg = json.load(json_f)

	# Check that it's up to snuff
	config_checker(cfg)

	# Return it
	return cfg


def prepare_tf_record(cfg,root_path,tf_record_path,final_params,train_or_test):
	"""
	Perpare the tf record using the config file values.

	Parameters
	----------
		cfg (dict): The dictionary attained from reading the json config
			file.
		root_path (str): The root path that will contain all of the data
			including the lens parameters, the npy files, and the TFRecord.
		tf_record_path (str): The path where the TFRecord will be saved.
		final_params ([str,...]): The parameters we expect to be in the final
			set of lens parameters.
		train_or_test (string): If test, the normalizations will be
			saved. If train, the training normalizations will be used.
	"""
	# Path to csv containing lens parameters.
	lens_params_path = root_path + cfg['dataset_params']['lens_params_path']
	# The list of lens parameters that should be trained on
	lens_params = cfg['dataset_params']['lens_params']
	# Where to save the lens parameters to after the preprocessing 
	# transformations
	new_param_path = root_path + cfg['dataset_params']['new_param_path']
	# Where to save the normalization constants to. Note that we take the
	# root path associated with the training_params here, even if validation
	# params root path was passed in. This is because we always want to use
	# the training norms!
	normalization_constants_path = cfg['training_params'][
		'root_path'] + cfg['dataset_params'][
		'normalization_constants_path']
	# Parameters to convert to log space
	if 'lens_params_log' in cfg['dataset_params']:
		lens_params_log = cfg['dataset_params']['lens_params_log']
	else:
		lens_params_log = None
	# Parameters to convert from ratio and ang to excentricities
	if 'ratang' in cfg['dataset_params']:
		cfg_ratang = cfg['dataset_params']['ratang']
		# New prefix for those parameters
		ratang_parameter_prefixes = cfg_ratang['ratang_parameter_prefixes']
		# The parameter names of the ratios
		ratang_params_rat = cfg_ratang['ratang_params_rat']
		# The parameter names of the angles
		ratang_params_ang = cfg_ratang['ratang_params_ang']
	else:
		ratang_parameter_prefixes=None; ratang_params_rat=None;
		ratang_params_ang=None

	# First write desired parameters in log space.
	if lens_params_log is not None:
		data_tools.write_parameters_in_log_space(lens_params_log,
			lens_params_path,new_param_path)
		# Add log version of parameter.
		for lens_param_log in lens_params_log:
			lens_params.append(lens_param_log+'_log')
		# Parameters should be read from this path from now on.
		lens_params_path = new_param_path

	# Now convert ratio and angle parameters to excentricities.
	if ratang_parameter_prefixes is not None:
		for ratangi in range(len(ratang_parameter_prefixes)):
			data_tools.ratang_2_exc(ratang_params_rat[ratangi],
				ratang_params_ang[ratangi],lens_params_path,new_param_path,
				ratang_parameter_prefixes[ratangi])
			# Update lens_params
			lens_params.append(ratang_parameter_prefixes[ratangi]+'_e1')
			lens_params.append(ratang_parameter_prefixes[ratangi]+'_e2')
			# Parameters should be read from this path from now on.
			lens_params_path = new_param_path

	# Now normalize all of the lens parameters
	data_tools.normalize_lens_parameters(lens_params,lens_params_path,
		new_param_path,normalization_constants_path,
		train_or_test=train_or_test)

	# Quickly check that all the desired lens_params ended up in the final
	# csv file.
	for final_param in final_params:
		if final_param not in lens_params:
			raise RuntimeError('Desired lens parameters and lens parameters in'+
				' final csv do not match')

	# Finally, generate the TFRecord
	data_tools.generate_tf_record(root_path,lens_params,new_param_path,
		tf_record_path)

def get_normed_pixel_scale(cfg,pixel_scale):
	"""
	Return a dictionary with the pixel scale normalized according to the 
	normalization of each shift parameter.

	Parameters
	----------
		cfg (dict): The dictionary attained from reading the json config file.
		pixel_scale (float): The pixel scale used for the original images.

	Returns
	-------
		(dict): A dictionary of the pixel scales renormalized in the same way as 
			the shift parameters.
	"""
	# Get the parameters we need to read the normalization from
	shift_params = cfg['training_params']['shift_params']
	# Adjust the pixel scale by the normalization
	normalization_constants_path = cfg['training_params']['root_path'] + cfg[
		'dataset_params']['normalization_constants_path']
	norm_const_dict = pd.read_csv(normalization_constants_path, index_col=None)
	# Set the normed pixel scale for each parameter
	normed_pixel_scale = {}
	for shift_param in shift_params[0]:
		normed_pixel_scale[shift_param] = pixel_scale/norm_const_dict[
			shift_param][1]
	for shift_param in shift_params[1]:
		normed_pixel_scale[shift_param] = pixel_scale/norm_const_dict[
			shift_param][1]
	return normed_pixel_scale

def model_loss_builder(cfg, verbose=False):
	"""
	Build a model according to the specifications in configuration dictionary
	and return both the initialized model and the loss function.

	Parameters
	----------
		cfg (dict): The dictionary attained from reading the json config file.
		verbose (bool): If True, will be verbose as model is built.

	Returns
	-------
		(tf.keras.model, function): A bnn model of the type specified in config
			and a callable function to construct the tesnorflow graph for the 
			loss. 
	"""
	# Load the parameters we need from the config file. Some of these will
	# be repeats from the main script.

	# The final parameters that need to be in tf_record_path
	final_params = cfg['training_params']['final_params']
	num_params = len(final_params)
	# The learning rate
	learning_rate = cfg['training_params']['learning_rate']
	# The decay rate for Adam
	decay = cfg['training_params']['decay']
	# Image dimensions
	img_dim = cfg['training_params']['img_dim']
	# Weight and dropout regularization parameters for the concrete dropout
	# model.
	kr = cfg['training_params']['kernel_regularizer']
	dr = cfg['training_params']['dropout_regularizer']
	# If the any of the parameters contain excentricities then the e1/e2
	# pair should be included in the flip list for the correct loss function
	# behavior. See the example config files.
	flip_pairs = cfg['training_params']['flip_pairs']
	# The type of BNN output (either diag, full, or gmm).
	bnn_type = cfg['training_params']['bnn_type']
	# The path to the model weights. If they already exist they will be loaded
	model_weights = cfg['training_params']['model_weights']
	# Finally set the random seed we will use for training
	random_seed = cfg['training_params']['random_seed']


	# Initialize the log function according to bnn_type
	loss_class = bnn_alexnet.LensingLossFunctions(flip_pairs,num_params)
	if bnn_type == 'diag':
		loss = loss_class.diagonal_covariance_loss
		num_outputs = num_params*2
	elif bnn_type == 'full':
		loss = loss_class.full_covariance_loss
		num_outputs = num_params + int(num_params*(num_params+1)/2)
	elif bnn_type == 'gmm':
		loss = loss_class.gm_full_covariance_loss
		num_outputs = 2*(num_params + int(num_params*(num_params+1)/2))+1
	else:
		raise RuntimeError('BNN type %s does not exist'%(bnn_type))
	# The mse loss doesn't depend on model type.
	mse_loss = loss_class.mse_loss

	model = bnn_alexnet.concrete_alexnet((img_dim, img_dim, 1), num_outputs,
		kernel_regularizer=kr,dropout_regularizer=dr,random_seed=random_seed)

	adam = Adam(lr=learning_rate,amsgrad=False,decay=decay)
	model.compile(loss=loss, optimizer=adam, metrics=[loss,mse_loss])
	if verbose:
		print('Is model built: ' + str(model.built))

	try:
		model.load_weights(model_weights)
		if verbose:
			print('Loaded weights %s'%(model_weights))
	except:
		if verbose:
			print('No weights found. Saving new weights to %s'%(model_weights))

	return model, loss

def main():
	"""
	Initializes and trains a BNN network. Path to config file are read from 
	command line arguments.
	"""
	# Initialize argument parser to pull neccesary paths
	parser = argparse.ArgumentParser()
	parser.add_argument('config',help='json config file containing BNN type ' + 
		'and data/model paths')
	args = parser.parse_args()

	cfg = load_config(args.config)

	# Extract neccesary parameters from the json config
	# The batch size used for training
	batch_size = cfg['training_params']['batch_size']
	# The number of epochs of training
	n_epochs = cfg['training_params']['n_epochs']
	# The root path that will contain all of the training data including the lens
	# parameters, the npy files, and the TFRecord for training.
	root_path_t = cfg['training_params']['root_path']
	# The same but for validation
	root_path_v = cfg['validation_params']['root_path']
	# The filename of the TFRecord for training data
	tf_record_path_t = root_path_t+cfg['training_params']['tf_record_path']
	# The same but for validation
	tf_record_path_v = root_path_v+cfg['validation_params']['tf_record_path']
	# The final parameters that need to be in tf_record_path
	final_params = cfg['training_params']['final_params']
	# The path to the model weights. If they already exist they will be loaded
	model_weights = cfg['training_params']['model_weights']
	# The path for the Tensorboard logs
	tensorboard_log_dir = cfg['training_params']['tensorboard_log_dir']

	# The parameters govern the augmentation of the data
	# Whether or not the images should be normalzied to have standard
	# deviation 1
	norm_images = cfg['training_params']['norm_images']
	# The number of pixels to uniformly shift the images by and the
	# parameters that need to be rescaled to account for this shift
	shift_pixels = cfg['training_params']['shift_pixels']
	shift_params = cfg['training_params']['shift_params']
	# What the pixel_scale of the images is. This will be adjusted for the
	# normalization.
	pixel_scale = cfg['training_params']['pixel_scale']

	# Finally set the random seed we will use for training
	random_seed = cfg['training_params']['random_seed']
	tf.random.set_seed(random_seed)

	print('Checking for training data.')
	if not os.path.exists(tf_record_path_t):
		print('Generating new TFRecord at %s'%(tf_record_path_t))
		prepare_tf_record(cfg,root_path_t,tf_record_path_t,final_params,
			train_or_test='train')
	else:
		print('TFRecord found at %s'%(tf_record_path_t))

	print('Checking for validation data.')
	if not os.path.exists(tf_record_path_v):
		print('Generating new TFRecord at %s'%(tf_record_path_v))
		prepare_tf_record(cfg,root_path_v,tf_record_path_v,final_params,
			train_or_test='test')
	else:
		print('TFRecord found at %s'%(tf_record_path_v))

	# Get the normalzied pixel scale (will fail if tf_record has not been
	# correctly created.)
	normed_pixel_scale = get_normed_pixel_scale(cfg,pixel_scale)

	# We let keras deal with epochs instead of the tf dataset object.
	tf_dataset_t = data_tools.build_tf_dataset(tf_record_path_t,final_params,
		batch_size,1,norm_images=norm_images,shift_pixels=shift_pixels,
		shift_params=shift_params, normed_pixel_scale=normed_pixel_scale)
	# Validation dataset will, by default, have no augmentation but will have
	# the images normalized if requested.
	tf_dataset_v = data_tools.build_tf_dataset(tf_record_path_v,final_params,
		batch_size,1,norm_images=norm_images)

	print('Initializing the model')

	model, loss = model_loss_builder(cfg,verbose=True)

	tensorboard = TensorBoard(log_dir=tensorboard_log_dir,update_freq='batch')
	modelcheckpoint = ModelCheckpoint(model_weights)

	# TODO add validation data.
	model.fit(tf_dataset_t,callbacks=[tensorboard, modelcheckpoint],
		epochs = n_epochs, validation_data=tf_dataset_v)

if __name__ == '__main__':
    main()

