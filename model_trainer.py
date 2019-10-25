"""
This script will initialize and train a BNN model on a strong lensing image
dataset.

"""

# Import some backend stuff
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import argparse, json, os
import tensorflow.keras.backend as K

# Import the code to construct the bnn and the data pipeline
import bnn_alexnet
import data_tools

def prepare_tf_record(cfg,root_path,tf_record_path,final_params):
	"""
	Perpare the tf record using the config file values.

	Parameters:
		cfg: The dictionary attained from reading the json config
			file.
		root_path: The root path that will contain all of the training data
			including the lens parameters, the npy files, and the TFRecord.
		tf_record_path: The path where the TFRecord will be saved.
		final_params: The parameters we expect to be in the final
			set of lens parameters.
	Returns:
		None. The tf recrod will be generated.
	"""
	# Path to csv containing lens parameters.
	lens_params_path = root_path + cfg['dataset_params']['lens_params_path']
	# The list of lens parameters that should be trained on
	lens_params = cfg['dataset_params']['lens_params']
	# Where to save the lens parameters to after the preprocessing 
	# transformations
	new_param_path = root_path + cfg['dataset_params']['new_param_path']
	# Where to save the normalization constants to
	normalization_constants_path = root_path + cfg['dataset_params'][
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
		new_param_path,normalization_constants_path,train_or_test='train')

	# Quickly check that all the desired lens_params ended up in the final
	# csv file.
	for final_param in final_params:
		if final_param not in lens_params:
			raise RuntimeError('Desired lens parameters and lens parameters in'+
				' final csv do not match')

	# Finally, generate the TFRecord
	data_tools.generate_tf_record(root_path,lens_params,new_param_path,
		tf_record_path)

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

	with open(args.config,'r') as json_f:
		cfg = json.load(json_f)

	# Extract neccesary parameters from the json config
	# The batch size used for training
	batch_size = cfg['training_params']['batch_size']
	# The number of epochs of training
	n_epochs = cfg['training_params']['n_epochs']
	# The learning rate
	learning_rate = cfg['training_params']['learning_rate']
	# The decay rate for Adam
	decay = cfg['training_params']['decay']
	# The root path that will contain all of the training data including the lens
	# parameters, the npy files, and the TFRecord.
	root_path = cfg['training_params']['root_path']
	# The filename of the TFRecord
	tf_record_path = root_path+cfg['training_params']['tf_record_path']
	# The final parameters that need to be in tf_record_path
	final_params = cfg['training_params']['final_params']
	num_params = len(final_params)
	# The path to the model weights. If they already exist they will be loaded
	model_weights = cfg['training_params']['model_weights']
	# Image dimensions
	img_dim = cfg['training_params']['img_dim']
	# Weight and dropout regularization parameters for the concrete dropout
	# model.
	wr = cfg['training_params']['weight_regularizer']
	dr = cfg['training_params']['dropout_regularizer']
	# The path for the Tensorboard logs
	tensorboard_log_dir = cfg['training_params']['tensorboard_log_dir']

	# If the any of the parameters contain excentricities then the e1/e2
	# pair should be included in the flip list for the correct loss function
	# behavior. See the example config files.
	flip_pairs = cfg['training_params']['flip_pairs']
	# The type of BNN output (either diag, full, or gmm).
	bnn_type = cfg['training_params']['bnn_type']

	if not os.path.exists(tf_record_path):
		print('Generating new TFRecord at %s'%(tf_record_path))
		prepare_tf_record(cfg,root_path,tf_record_path,final_params)
	else:
		print('TFRecord found at %s'%(tf_record_path))

	tf_dataset = data_tools.build_tf_dataset(tf_record_path,final_params,
		batch_size,n_epochs)

	print('Initializing the model')

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

	model = bnn_alexnet.concrete_alexnet((img_dim, img_dim, 1), num_outputs,
		weight_regularizer=wr,dropout_regularizer=dr)

	adam = Adam(lr=learning_rate,amsgrad=False,decay=decay)
	model.compile(loss=loss, optimizer=adam)
	print('Is model built: ' + str(model.built))

	try:
		model.load_weights(model_weights)
		print('Loaded weights %s'%(model_weights))
	except:
		print('No weights found. Saving new weights to %s'%(model_weights))

	tensorboard = TensorBoard(log_dir=tensorboard_log_dir,update_freq='batch')
	modelcheckpoint = ModelCheckpoint(model_weights)

	for layer in model.layers:
		if 'concrete' in layer.name:
			print(layer.name,K.sigmoid(layer.weights[0]))

	# TODO add validation data.
	model.fit_generator(tf_dataset,callbacks=[tensorboard, modelcheckpoint])

	for layer in model.layers:
		if 'concrete' in layer.name:
			print(layer.name,K.sigmoid(layer.weights[0]))

if __name__ == '__main__':
    main()

