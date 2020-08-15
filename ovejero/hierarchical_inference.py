# -*- coding: utf-8 -*-
"""
Given lens posteriors from a model, conduct hierarchical inference on the lens
parameter distributions.

This module contains the functions neccesary for interfacing with the baobab
distributions and running an mcmc to extract posteriors on the lens parameter
distributions.

Examples
--------
The demo Test_Hierarchical_Inference.ipynb gives a number of examples on how to
use this module.
"""
import numpy as np
from scipy import special
from baobab import configs
from baobab import distributions
from ovejero import bnn_inference, data_tools, model_trainer
from inspect import signature
import emcee, os, glob, corner, copy, numba
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
from importlib import import_module

# This is ugly, but we have to set lens samples as a global variable for the
# pooling to work well. Otherwise the data is pickled and unpickled each time
# one of the cpus calls the log likelihood function, slowing things down.
lens_samps = None


def load_prior_config(cfg_path):
	"""
	Mirrors the behavior of the baobab config loader without requiring
	all the baobab config values to be set.

	Parameters
	----------
		user_cfg_path (str): Path to the ovejero distribution configuration
			file.

	Returns
	-------
		(addict.Dict): A dictionary with the prior parameters for
			hierarchical inference.
	"""
	dirname, filename = os.path.split(os.path.abspath(cfg_path))
	module_name, ext = os.path.splitext(filename)
	sys.path.insert(0, dirname)
	user_cfg_script = import_module(module_name)
	user_cfg = getattr(user_cfg_script, 'cfg')
	return user_cfg


def convert_tril_to_cov(tril):
	"""
	Convert a list of lower triangular matrix entries to the corresponding
	covariance matrix.

	Parameters
	----------
		tril (np.array): A 1D array with the lower triangular values

	Returns
	-------
		(np.array): The covariance matrix
	"""
	# Format the lower triangular matrix and return the dot product.
	n_params = int(0.5*(np.sqrt(1+8*len(tril))-1))
	tril_mask = np.tri(n_params,dtype=bool, k=0)
	tril_mat = np.zeros((n_params,n_params))
	tril_mat[tril_mask] = tril
	return np.dot(tril_mat,tril_mat.T)


def convert_trils_to_covs(tril):
	"""
	Convert a list of lower triangular matrix entries to the corresponding
	covariance matrix.

	Parameters
	----------
		tril (np.array): A 1D array with the lower triangular values

	Returns
	-------
		(np.array): The covariance matrix
	"""
	# Format the lower triangular matrix and return the dot product.
	n_params = int(0.5*(np.sqrt(1+8*tril.shape[-1])-1))
	tril_mask = np.tri(n_params,dtype=bool, k=0)
	tril_mats = np.zeros((tril.shape[0],n_params,n_params))
	cov_mats = np.zeros((tril.shape[0],n_params,n_params))
	for i in range(tril.shape[0]):
		tril_mats[i,tril_mask] = tril[i]

	# We'll do the dot product in numba to speed things up a bit.
	@numba.njit()
	def tril_to_cov_mats(tril_mats,cov_mats):
		for i in range(tril_mats.shape[0]):
			cov_mats[i] = np.dot(tril_mats[i],tril_mats[i].T)

	tril_to_cov_mats(tril_mats,cov_mats)
	return cov_mats


def build_eval_dict(cfg_dict,lens_params,baobab_config=True):
	"""
	Map between the baobab config and a dictionary that contains the
	evaluation function and hyperparameter indices for each parameters.

	Parameters
	----------
		cfg_dict (dict): A configuration dictionary containing the
			entries for each lens parameter. This is either a baobab config
			(in which case no distributions or priors are extracted) or it
			is an ovejero distribution config.
		lens_params ([str,...]): The parameters to pull from the configuration
			file.
		baobab_config (bool): If true the input config is a baobab config. If
			False the input config is an ovejero distribution config.
	Returns
	-------
		dict: The evaluation dictionary used by HierarchicalClass functions to
			calculate the log pdfs.

	Notes
	-----
		This will be automatically forward compatible with new distributions in
		baobab.
	"""

	if baobab_config:
		# In this case we only need to build the evaluation for each
		# lens parameter and the hyperparameter values.
		eval_dict = dict(hyp_len=0, hyp_values=[], hyp_names=[])
	else:
		# Initialize eval_dict with empty lists for hyperparameters and
		# hyperpriors.
		eval_dict = dict(hyp_len=0, hyp_init=[], hyp_sigma=[], hyp_prior=[],
			hyp_names=[])
	# For each lens parameter add the required hyperparameters and evaluation
	# function.
	for lens_param in lens_params:
		# Skip lens parameters that are in the covariance matrix
		if ('cov_info' in cfg_dict.bnn_omega and
			lens_param in cfg_dict.bnn_omega['cov_info']['cov_params_list']):
			eval_dict[lens_param] = None
			continue
		# Make a new entry for the parameter
		eval_dict[lens_param] = dict()

		# Load the config entry associated to this parameter
		lens_split = lens_param.split('_')
		# This is a bit ugly, but it's the quickest way to map between the
		# parameter name and the location in the config file.
		dist = cfg_dict.bnn_omega['_'.join(lens_split[:2])][
			'_'.join(lens_split[2:])]

		# Get the function in question from baobab.
		eval_fn = getattr(distributions,'eval_{}_logpdf_approx'.format(
			dist['dist']))
		eval_sig = signature(eval_fn)
		fn_name = dist['dist']

		# Hyperparameters is number of parameters-1 because the first parameter
		# is where to evaluate.
		n_hyps = len(eval_sig.parameters)-1

		# Initialize the dict of lens_param evaluation function kwargs
		eval_dict[lens_param]['eval_fn_kwargs'] = {}

		# Add the value of the hyperparameters in the config file. Again skip
		# the first one.
		for hyperparam_name in list(eval_sig.parameters.keys())[1:]:
			# Sometimes hyperparameters are not specified
			if hyperparam_name not in dist or dist[hyperparam_name] == {}:
				n_hyps -= 1
			# For the baobab config we just populate the hyp values and names
			elif baobab_config:
				eval_dict['hyp_values'].extend([dist[hyperparam_name]])
				eval_dict['hyp_names'].extend([lens_param+':'+hyperparam_name])
			# For the ovejero distribution config, we want to deal with fixed
			# hyperparameters. We will do so by passing them in as kwargs to
			# our evaluation function. Hyperparameters are fixed if they
			# have a sigma of 0.
			elif dist[hyperparam_name]['sigma'] == 0:
				n_hyps -= 1
				eval_dict[lens_param]['eval_fn_kwargs'][hyperparam_name] = (
					dist[hyperparam_name]['init'])
			# For non fixed hyperparameters we need to get their initial value
			# the spread (sigma) we want to use, and the functional form of the
			# prior.
			else:
				eval_dict['hyp_init'].extend([dist[hyperparam_name]['init']])
				eval_dict['hyp_sigma'].extend([dist[hyperparam_name]['sigma']])
				eval_dict['hyp_names'].extend([lens_param+':'+hyperparam_name])
				eval_dict['hyp_prior'].extend([dist[hyperparam_name]['prior']])

		# Record the indices we saved the hyperparameter values for this
		# lens parameter to.
		eval_dict[lens_param]['hyp_ind'] = np.arange(eval_dict['hyp_len'],
			eval_dict['hyp_len']+n_hyps)
		eval_dict['hyp_len'] += n_hyps

		# Finally, actually include the evaluation function.
		eval_dict[lens_param]['eval_fn'] = eval_fn
		eval_dict[lens_param]['fn_name'] = fn_name

	# Add covariance matrix information if that's relevant
	if 'cov_info' in cfg_dict.bnn_omega:
		cov_dict = cfg_dict.bnn_omega['cov_info']
		if baobab_config:
			cov_dim = len(cov_dict['cov_omega']['mu'])
		else:
			cov_dim = len(cov_dict['cov_omega']['mu']['init'])
		# Add the indices for the mu and tril parameters seperately
		eval_dict['cov_params_list'] = cov_dict['cov_params_list']
		eval_dict['cov_params_is_log'] = cov_dict['cov_omega']['is_log']
		n_hyps = cov_dim
		eval_dict['cov_mu_hyp_ind'] = np.arange(eval_dict['hyp_len'],
			eval_dict['hyp_len']+n_hyps)
		eval_dict['hyp_len'] += n_hyps
		n_hyps = int(cov_dim*(cov_dim+1)/2)
		eval_dict['cov_tril_hyp_ind'] = np.arange(eval_dict['hyp_len'],
			eval_dict['hyp_len']+n_hyps)
		eval_dict['hyp_len'] += n_hyps
		# Now extract the desired dictionary values depending on whether or not
		# we have passed in a baobab_config.
		if baobab_config:
			eval_dict['hyp_values'].extend(cov_dict['cov_omega']['mu'])
			# The baobab config specifies the covariance matrix, but our
			# hyperparameters are the lower triangular matrix. Here we
			# correct for that.
			cov = np.array(cov_dict['cov_omega']['cov_mat'])
			tril = np.linalg.cholesky(cov).astype(np.float)
			tril_mask = np.tri(cov_dim,dtype=bool, k=0)
			eval_dict['hyp_values'].extend(tril[tril_mask])
			for i in range(cov_dim):
				eval_dict['hyp_names'].extend(['cov_mu_%d'%(i)])
			for i in range(int(cov_dim*(cov_dim+1)/2)):
				eval_dict['hyp_names'].extend(['cov_tril_%d'%(i)])
		else:
			eval_dict['hyp_init'].extend(cov_dict['cov_omega']['mu']['init'])
			eval_dict['hyp_init'].extend(cov_dict['cov_omega']['tril']['init'])
			eval_dict['hyp_sigma'].extend(cov_dict['cov_omega']['mu']['sigma'])
			eval_dict['hyp_sigma'].extend(cov_dict['cov_omega']['tril']['sigma'])
			for i in range(cov_dim):
				eval_dict['hyp_names'].extend(['cov_mu_%d'%(i)])
			for i in range(int(cov_dim*(cov_dim+1)/2)):
				eval_dict['hyp_names'].extend(['cov_tril_%d'%(i)])
			eval_dict['hyp_prior'].extend(cov_dict['cov_omega']['mu']['prior'])
			eval_dict['hyp_prior'].extend(cov_dict['cov_omega']['tril']['prior'])

	# Transform list of values into np array
	if baobab_config:
		eval_dict['hyp_values'] = np.array(eval_dict['hyp_values'])
	else:
		eval_dict['hyp_init'] = np.array(eval_dict['hyp_init'])
		eval_dict['hyp_sigma'] = np.array(eval_dict['hyp_sigma'])

	return eval_dict


def log_p_xi_omega(samples, hyp, eval_dict,lens_params):
	"""
	Calculate log p(xi|omega) - the probability of the lens parameters in
	the data given the proposed lens parameter distribution.

	Parameters
	----------
		samples (np.array): A numpy array with dimensions (n_params,n_samps,
			batch_size)
		hyp (np.array): A numpy array with dimensions (n_hyperparameters).
			These are the hyperparameters that will be used for evaluation.
		eval_dict (dict): A dictionary from build_evaluation_dictionary to
			query for the evaluation functions.
		lens_params ([str,...]): A list of strings of the lens parameters
			generated by baobab.

	Returns
	-------
		np.array: A numpy array of the shape (n_samps,batch_size) containing
			the log p(xi|omega) for each sample.
	"""

	# We iterate through each lens parameter and carry out the evaluation
	logpdf = np.zeros((samples.shape[1],samples.shape[2]))

	for li, lens_param in enumerate(lens_params):
		# Skip covariance parameters.
		if ('cov_params_list' in eval_dict and
			lens_param in eval_dict['cov_params_list']):
			continue
		logpdf += eval_dict[lens_param]['eval_fn'](samples[li],
			*hyp[eval_dict[lens_param]['hyp_ind']],
			**eval_dict[lens_param]['eval_fn_kwargs'])

	# Calculate covariate parameters
	if 'cov_params_list' in eval_dict:
		# Identify the samples associated with the covariance parameters
		cov_samples_index = []
		for cov_lens_param in eval_dict['cov_params_list']:
			cov_samples_index.append(lens_params.index(cov_lens_param))
		cov_samples = samples[cov_samples_index]
		for ili, is_log in enumerate(eval_dict['cov_params_is_log']):
			if is_log:
				cov_samples[ili,cov_samples[ili]<=0] = 1e-22
				cov_samples[ili] = np.log(cov_samples[ili])

		# Get the mean and covariance we want to use
		mu = hyp[eval_dict['cov_mu_hyp_ind']]
		tril = hyp[eval_dict['cov_tril_hyp_ind']]
		cov = convert_tril_to_cov(tril)

		# Reshape the covariance samples to feed into the logpdf function
		orig_shape = cov_samples.T.shape
		cov_samples = cov_samples.T.reshape(-1,len(mu))

		logpdf_cov = stats.multivariate_normal(mean=mu,cov=cov).logpdf(
			cov_samples)

		# This is a hardcode, but for axis ratio we want to renormalize by
		# the area cut by q<=1.
		if 'lens_mass_q' in eval_dict['cov_params_list']:
			qi = eval_dict['cov_params_list'].index('lens_mass_q')
			logpdf_cov -= np.log(stats.norm(mu[qi],np.sqrt(cov[qi,qi])).cdf(1))

		logpdf += logpdf_cov.reshape(orig_shape[:-1]).T

	# Clean up any lingering nans.
	logpdf[np.isnan(logpdf)] = -np.inf

	return logpdf


def log_p_omega(hyp, eval_dict):
	"""
	Calculate log p(omega) - the probability of the hyperparameters given
		the hyperprior.

	Parameters
	----------
		hyp (np.array): A numpy array with dimensions (n_hyperparameters).
			These are the hyperparameters that will be used for evaluation.
		eval_dict (dict): A dictionary from build_evaluation_dictionary to
			query for the evaluation functions.

	Returns
	-------
		np.float: The value of log p(omega)
	"""

	# We iterate through each hyperparamter and evaluate it on its prior.
	logpdf = 0
	for hpi, hyper_param in enumerate(hyp):
		logpdf += eval_dict['hyp_prior'][hpi](hyper_param)

	# Give -np.inf in the case of a nan.
	if np.sum(np.isnan(logpdf))>0:
		logpdf = -np.inf

	return logpdf


class ProbabilityClass:
	"""
	A companion class to HierarchicalClass that does all of the probability
	calculations. These funcitons are placed in a seperate class to allow
	for the functions to be pickled.
	Parameters
	----------
		target_eval_dict (dict): A dictionary from build_evaluation_dictionary
				containing the target distribution evaluation functions.
		interim_eval_dict (dict): A dictionary from build_evaluation_dictionary
				containing the interim distribution evaluation functions.
		lens_params_train ([str,....]): A list of strings containing the lens
			params used in training.
		lens_params_test ([str,...]): A list of strings contraining the lens
			params used in the test set.
	"""
	def __init__(self,target_eval_dict,interim_eval_dict,lens_params_train,
		lens_params_test):
		# Save the parameters we'll need.
		self.target_eval_dict = copy.deepcopy(target_eval_dict)
		self.interim_eval_dict = copy.deepcopy(interim_eval_dict)
		self.lens_params_train = copy.copy(lens_params_train)
		self.lens_params_test = copy.copy(lens_params_test)
		# The samples_init flag.
		self.samples_init = False
		# The probability of the data given the interim prior. Will be
		# calculated when the samples are passed in.
		self.pt_omegai = None

	def set_samples(self):
		"""
		Set the lens samples that will be used for the log_post_omega
			calculation.
		"""
		# Now that we have the samples, we can calculate the probability
		# of the samples given the interim prior.
		global lens_samps
		self.pt_omegai = log_p_xi_omega(lens_samps,
			self.interim_eval_dict['hyp_values'], self.interim_eval_dict,
			self.lens_params_train)
		# Set initialziation variable to True.
		self.samples_init = True

	def log_post_omega(self, hyp):
		"""
		Given generated data, calculate the log posterior of omega.

		Parameters
		----------
			hyp (np.array): A numpy array with dimensions (n_hyperparameters).
				These are the values of omega's parameters.

		Returns
		-------
			(float): The log posterior of omega given the data.

		Notes
		-----
			Constant factors with respect to omega are ignored.
		"""
		if self.samples_init is False:
			raise RuntimeError('Must generate samples before fitting')
		global lens_samps
		# Add the prior on omega
		lprior = log_p_omega(hyp,self.target_eval_dict)

		if lprior == -np.inf:
			return lprior

		# Calculate the probability of each datapoint given omega
		pt_omega = log_p_xi_omega(lens_samps, hyp, self.target_eval_dict,
			self.lens_params_test)
		# We've already calculated the value of pt_omegai when we generated
		# the samples. Now we just need to sum them correctly. Note that the
		# first axis in samples has dimension number of samples, so that is
		# what we want to log sum exp over.
		like_ratio = pt_omega-self.pt_omegai
		like_ratio[np.isinf(self.pt_omegai)] = -np.inf
		like_ratio = special.logsumexp(like_ratio,axis=0)
		like_ratio[np.isnan(like_ratio)] = -np.inf

		return lprior + np.sum(like_ratio)


class HierarchicalClass:
	"""
	A class that contains all of the functions needed to conduct a hierarchical
	calculation of the lens parameter distributions.
	Parameters
	----------
		cfg (dict): The dictionary attained from reading the json config file.
		interim_baobab_omega_path (str): The string specifying the path to the
			baobab config for the interim omega.
		target_ovejero_omega_path (str): The string specifying the path to the
			ovejero distribution config for the target omega.
		test_dataset_path (str): The path to the dataset on which hiearchical
			inference will be conducted
		test_dataset_tf_record_path (str): The path where the TFRecord will be
			saved. If it already exists it will be loaded.
		target_ovejero_omega_path (str): The string specifying the path to the
			baobab distribution config for the target omega. This will be used
			to get the true values of the ditribution for plotting. If set to
			None then no true values for the test distribution will be shown.
		train_to_test_param_map (dict): If a mapping is required between the
			lens parameters of the training distribution and the lens
			parameters of the test distribution this dict should specify
			them. It should include three entire: orig_params - the original
			lens parameter names, new_params - the new parameter names,
			and transform_func - the function to map between them.
		lite_class (bool): If True, do not bother loading the BNN model weights.
			This allows the user to save on memory, but will cause an error
			if the BNN samples have not already been drawn.
	"""
	def __init__(self,cfg,interim_baobab_omega_path,target_ovejero_omega_path,
		test_dataset_path,test_dataset_tf_record_path,
		target_baobab_omega_path=None,train_to_test_param_map=None,
		lite_class=False):
		# Initialzie our class.
		self.cfg = cfg
		# Pull the needed param information from the config file.
		self.lens_params_train = cfg['dataset_params']['lens_params']
		self.lens_params_test = copy.deepcopy(self.lens_params_train)
		# We will need to encode the difference between the test and train
		# parameter names.
		if train_to_test_param_map is not None:
			self.lens_params_change_ind = []
			# Go through each parameter, mark its index, and make the swap
			for li, lens_param in enumerate(
				train_to_test_param_map['orig_params']):
				self.lens_params_change_ind.append(self.lens_params_train.index(
					lens_param))
				self.lens_params_test[self.lens_params_change_ind[-1]] = (
					train_to_test_param_map['new_params'][li])

		self.lens_params_log = cfg['dataset_params']['lens_params_log']
		self.gampsi = cfg['dataset_params']['gampsi']
		self.final_params = cfg['training_params']['final_params']

		# Read the config files and turn them into evaluation dictionaries
		self.interim_baobab_omega = configs.BaobabConfig.from_file(
			interim_baobab_omega_path)
		self.target_baobab_omega = load_prior_config(target_ovejero_omega_path)
		self.interim_eval_dict = build_eval_dict(self.interim_baobab_omega,
			self.lens_params_train,baobab_config=True)
		self.target_eval_dict = build_eval_dict(self.target_baobab_omega,
			self.lens_params_test,baobab_config=False)
		self.train_to_test_param_map = train_to_test_param_map

		# Get the number of parameters and set the batch size to the full
		# test set.
		self.num_params = len(self.lens_params_train)
		self.norm_images = cfg['training_params']['norm_images']
		n_npy_files = len(glob.glob(os.path.join(test_dataset_path,'X*.npy')))
		self.cfg['training_params']['batch_size'] = n_npy_files

		# Make our inference class we'll use to generate samples.
		self.infer_class = bnn_inference.InferenceClass(self.cfg,lite_class)

		# The inference class will load the validation set from the config
		# file. We do not want this. Therefore we must reset it here.
		if not os.path.exists(test_dataset_tf_record_path):
			print('Generating new TFRecord at %s'%(test_dataset_tf_record_path))
			model_trainer.prepare_tf_record(cfg,test_dataset_path,
				test_dataset_tf_record_path,self.final_params,
				train_or_test='test')
		else:
			print('TFRecord found at %s'%(test_dataset_tf_record_path))
		self.tf_dataset = data_tools.build_tf_dataset(
			test_dataset_tf_record_path,self.final_params,n_npy_files,1,
			target_baobab_omega_path,norm_images=self.norm_images)
		self.infer_class.tf_dataset_v = self.tf_dataset

		# Track if the sampler has been initialzied yet.
		self.sampler_init = False

		# Initialize our probability class
		self.prob_class = ProbabilityClass(self.target_eval_dict,
			self.interim_eval_dict,self.lens_params_train,self.lens_params_test)

		# If a baobab config path was provided for the test set we will extract
		# the true values of the hyperparameters from it
		if target_baobab_omega_path is not None:
			temp_config = configs.BaobabConfig.from_file(
				target_baobab_omega_path)
			temp_eval_dict = build_eval_dict(temp_config,self.lens_params_test,
				baobab_config=True)
			# Go through the target_eval_dict and extract the true values
			# from the temp_eval_dict (i.e. the eval dict generated by the
			# baobab config used to make the test set).
			self.true_hyp_values = []
			for name in self.target_eval_dict['hyp_names']:
				temp_index = temp_eval_dict['hyp_names'].index(name)
				self.true_hyp_values.append(temp_eval_dict['hyp_values'][
					temp_index])
		else:
			self.true_hyp_values = None

	def gen_samples(self,num_samples,sample_save_dir=None,subsample=None):
		"""
		Generate samples of lens parameters xi for use in hierarchical
		inference.

		Parameters
		----------
			num_samples (int): The number of samples to draw per lens.
			sample_save_dir (str): A path to a folder to save/load the samples.
				If None samples will not be saved. Do not include .npy, this will
				be appended (since several files will be generated).
			subsample (int): How many of the lenses is the test set to use
				for hierarchical inference. If None use all the lenses.
		"""

		if sample_save_dir is None or not os.path.isdir(sample_save_dir):
			if sample_save_dir is not None:
				print('No samples found. Saving samples to %s'%(
					sample_save_dir))
			# Most of the work will be done by the InferenceClass. The only
			# additional work we'll do here is undoing the polar to cartesian
			# transformation and the log transformation.
			if sample_save_dir is not None:
				self.infer_class.gen_samples(num_samples,sample_save_dir)
			else:
				self.infer_class.gen_samples(num_samples)

			# We'll steal the samples from bnn_infer and transform them
			# back into the original parameter space.
			self.predict_samps = self.infer_class.predict_samps
			global lens_samps
			lens_samps = np.zeros_like(self.predict_samps)

			# Even for parameters that have not been changed between baobab
			# and training time, we still need to make sure their position
			# in the new array is correct.
			for f_param in self.final_params:
				if f_param in self.lens_params_train:
					p_l = self.lens_params_train.index(f_param)
					p_f = self.final_params.index(f_param)
					lens_samps[:,:,p_l] = self.predict_samps[:,:,p_f]

			# Go through all the parameters that were put into log space and
			# put them back into their original form.
			for log_param in self.lens_params_log:
				log_p_l = self.lens_params_train.index(log_param)
				log_p_f = self.final_params.index(log_param+'_log')
				lens_samps[:,:,log_p_l] = np.exp(
					self.predict_samps[:,:,log_p_f])

			# Go through all the parameters that were put into cartesian
			# coordinates and put them back in polar coordinates.
			for gamps_pref, param_rat, param_ang in zip(
				self.gampsi['gampsi_parameter_prefixes'],
				self.gampsi['gampsi_params_rat'],
				self.gampsi['gampsi_params_ang']):
				param_g1 = gamps_pref+'_g1'
				param_g2 = gamps_pref+'_g2'
				pg1i = self.final_params.index(param_g1)
				pg2i = self.final_params.index(param_g2)
				rati = self.lens_params_train.index(param_rat)
				angi = self.lens_params_train.index(param_ang)

				lens_samps[:,:,rati] = np.sqrt(np.square(
					self.predict_samps[:,:,pg1i]) + np.square(
					self.predict_samps[:,:,pg2i]))
				lens_samps[:,:,angi] = np.arctan2(
					self.predict_samps[:,:,pg2i],
					self.predict_samps[:,:,pg1i])/2

			if sample_save_dir is not None:
				np.save(os.path.join(sample_save_dir,'lens_samps.npy'),
					lens_samps)

		else:
			print('Loading samples from %s'%(sample_save_dir))
			self.infer_class.gen_samples(num_samples,sample_save_dir)
			lens_samps = np.load(os.path.join(sample_save_dir,'lens_samps.npy'))

		# If subsample is passed in, truncate down to the desired number of
		# lenses.
		if subsample is not None:
			lens_samps = lens_samps[:,:subsample,:]
		# For numba, we need to change the order of the samples for fast
		# evaluation.
		lens_samps = np.ascontiguousarray(np.transpose(lens_samps,axes=(2,0,1)))

		# Pass the lens samples to our eval class.
		self.prob_class.set_samples()

		# Now we can modify our samples to agree with the test parameterization
		# if need be.
		if self.train_to_test_param_map is not None:
			lens_samps[self.lens_params_change_ind] = (
				self.train_to_test_param_map['transform_func'](
					*lens_samps[self.lens_params_change_ind]))

	def log_post_omega(self, hyp):
		"""
		Given generated data, calculate the log posterior of omega.

		Parameters
		----------
			hyp (np.array): A numpy array with dimensions (n_hyperparameters).
				These are the values of omega's parameters.

		Returns
		-------
			(float): The log posterior of omega given the data.

		Notes
		-----
			Constant factors with respect to omega are ignored.
		"""
		return self.prob_class.log_post_omega(hyp)

	def initialize_sampler(self,n_walkers,save_path,pool=None):
		"""
		Initialize the sampler to be used by run_samples.


		Parameters
		----------
			n_walkers (int): The number of walkers used by the sampler.
				Must be at least twice the number of hyperparameters.
			save_path (str): A pickle path specifying where to save the
				sampler chains. If a sampler chain is already present in the
				path it will be loaded.
			pool (Pool): An instance of multiprocessing.Pool which will be
				used for multiprocessing during the sampling.
		"""
		self.n_walkers = n_walkers
		ndim = self.target_eval_dict['hyp_len']
		if os.path.isfile(save_path):
			print('Loaded chains found at %s'%(save_path))
			self.cur_state = None
		else:
			print('No chains found at %s'%(save_path))
			# Start samples at initial value randomly distributed around
			# +- sigma.
			self.cur_state = ((np.random.rand(n_walkers, ndim)*2-1)*
				self.target_eval_dict['hyp_sigma'] +
				self.target_eval_dict['hyp_init'])

			# Ensure that no walkers start at a point with log probability
			# - np.inf
			all_finite = False
			while all_finite is False:
				all_finite = True
				f_counter = 0.0
				for w_i in range(n_walkers):
					if self.log_post_omega(self.cur_state[w_i]) == -np.inf:
						all_finite = False
						f_counter +=1
						self.cur_state[w_i] = self.cur_state[np.random.randint(
							n_walkers)]
				if f_counter > n_walkers*0.7:
					raise RuntimeError('Too few (%.3f) of the initial'%(
						1-f_counter/n_walkers)+'walkers have finite probability!')

		# Initialize backend hdf5 file that will store samples as we go
		self.backend = emcee.backends.HDFBackend(save_path)

		# Very important I pass in prob_class.log_post_omega here to allow
		# pickling.
		self.sampler = emcee.EnsembleSampler(n_walkers, ndim,
			self.prob_class.log_post_omega, backend=self.backend,pool=pool)

		self.sampler_init = True

	def run_sampler(self,n_samps,progress=True):
		"""
		Run an emcee sampler to get a posterior on the hyperparameters.

		Parameters
		----------
			n_samps (int): The number of samples to take
		"""
		if self.prob_class.samples_init is False:
			raise RuntimeError('Must generate samples before inference')
		if self.sampler_init is False:
			raise RuntimeError('Must initialize sampler before running sampler')

		self.sampler.run_mcmc(self.cur_state,n_samps,progress=progress)

		self.cur_state = None

	def plot_chains(self,burnin=None,hyperparam_plot_names=None,
		block=True):
		"""
		Plot the chains resulting from the emcee to figure out what
		the correct burnin is.

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			hyperparam_plot_names ([str,...]): A list containing the names
				of the hyperparameters to be used during plotting
			block (bool): If true, block excecution after plt.show() command
		"""
		if hyperparam_plot_names is None:
			hyperparam_plot_names = self.target_eval_dict['hyp_names']

		# Extract and plot the chains
		chains = self.sampler.get_chain()
		if burnin is not None:
			chains = chains[burnin:]
		for ci, chain in enumerate(chains.T):
			plt.plot(chain.T,'.')
			plt.title(hyperparam_plot_names[ci])
			plt.ylabel(hyperparam_plot_names[ci])
			plt.xlabel('sample')
			if self.true_hyp_values is not None:
				plt.axhline(self.true_hyp_values[ci],c='k')
			plt.show(block=block)

	def plot_auto_corr(self,hyperparam_plot_names=None,block=True):
		"""
		Plot the auto correlation time resulting from the emcee samples to
		check for convergence.
		Parameters
		----------
			hyperparam_plot_names ([str,...]): A list containing the names
				of the hyperparameters to be used during plotting
			block (bool): If true, block excecution after plt.show() command
		"""
		if hyperparam_plot_names is None:
			hyperparam_plot_names = self.target_eval_dict['hyp_names']

		# Both of these functions are pulled from the emcee examples and based
		# on work in Goodman and Weare 2010
		def auto_window(taus, c):
			m = np.arange(len(taus)) < c * taus
			if np.any(m):
				return np.argmin(m)
			return len(taus) - 1

		def autocorr_gw2010(y, c=5.0):
			f = emcee.autocorr.function_1d(np.mean(y, axis=0))
			taus = 2.0 * np.cumsum(f) - 1.0
			window = auto_window(taus, c)
			return taus[window]

		chains = self.sampler.get_chain()
		ns = np.logspace(1,np.log10(len(chains)),100)
		for ci in range(chains.shape[-1]):
			gw2010_auto_est = []
			for n in ns:
				trunc_chains = chains[:int(n),:,ci].T
				gw2010_auto_est.append(autocorr_gw2010(trunc_chains))
			plt.plot(ns,gw2010_auto_est)
			plt.plot(ns,ns/50)
			plt.yscale('log')
			plt.xscale('log')
			plt.ylabel('Autocorrelation Estimate')
			plt.xlabel('Number of Samples')
			plt.title('Autocorrelation Time for %s'%(hyperparam_plot_names[ci]))
			plt.legend(['Autocorrelation Estimate',
				r'Convergence Threshold ($\tau=N/50$)'])
			plt.show(block)

	def plot_corner(self,burnin,hyperparam_plot_names=None,block=True,
		color='#FFAA00',truth_color='#000000',plot_range=None,dpi=200):
		"""
		Plot the corner plot of chains resulting from the emcee

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			hyperparam_plot_names ([str,...]): A list containing the names
				of the hyperparameters to be used during plotting
			block (bool): If true, block excecution after plt.show() command
			color (str): The color to use for plotting the contour.
			truth_color (str): The color to use for plotting the truths in the
				corner plot.
			dpi (int): The dpi of the figure.
		"""
		if hyperparam_plot_names is None:
			hyperparam_plot_names = self.target_eval_dict['hyp_names']

		# Get the chains from the samples
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(hyperparam_plot_names))

		# Iterate through groups of hyperparameters and make the plots
		for lens_param in self.lens_params_test:
			# In the case of a multinormal for the test distribution, some
			# parameters will have empty dictionary entries
			if self.target_eval_dict[lens_param] is None:
				continue
			hyp_ind = self.target_eval_dict[lens_param]['hyp_ind']
			if hyp_ind.size == 0:
				continue
			hyp_s = np.min(hyp_ind)
			hyp_e = np.max(hyp_ind)+1
			truths = None
			if self.true_hyp_values is not None:
				truths = self.true_hyp_values[hyp_s:hyp_e]
			hist_kwargs = {'density':True,'color':color}
			corner.corner(chains[:,hyp_s:hyp_e],
				labels=hyperparam_plot_names[hyp_s:hyp_e],
				bins=20,show_titles=False, plot_datapoints=False,
				label_kwargs=dict(fontsize=10),
				truths=truths,
				levels=[0.68,0.95],color=color,fill_contours=True,
				truth_color=truth_color,range=plot_range,
				hist_kwargs=hist_kwargs,dpi=dpi)
			plt.show(block=block)

	def plot_single_corner(self,burnin,plot_param,hyperparam_plot_names=None,
		block=True,color='#FFAA00',truth_color='#000000',figure=None,
		plot_range=None,dpi=200,fontsize=13):
		"""
		Plot the corner plot of chains resulting from the emcee for a specific
		parameter

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			plot_param (str): The lens parameter to plot the hyperprior
				posteriors for.
			hyperparam_plot_names ([str,...]): A list containing the names
				of the hyperparameters to be used during plotting
			block (bool): If true, block excecution after plt.show() command
			color (str): The color to use for plotting the contour.
			truth_color (str): The color to use for plotting the truths in the
				corner plot.
			figure (matplotlib.pyplot.figure): A figure that was previously
				returned by plot_single_corner to overplot onto.
			dpi (int): The dpi of the figure.
			fontsize (int): The fontsize for the corner plot labels.
		"""
		if hyperparam_plot_names is None:
			hyperparam_plot_names = self.target_eval_dict['hyp_names']

		# Get the chains from the samples
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(hyperparam_plot_names))

		# Iterate through groups of hyperparameters and make the plots
		hyp_ind = self.target_eval_dict[plot_param]['hyp_ind']
		hyp_s = np.min(hyp_ind)
		hyp_e = np.max(hyp_ind)+1
		truths = None
		if self.true_hyp_values is not None:
			truths = self.true_hyp_values[hyp_s:hyp_e]
		hist_kwargs = {'density':True,'color':color}
		figure = corner.corner(chains[:,hyp_s:hyp_e],
			labels=hyperparam_plot_names[hyp_s:hyp_e],
			dpi=dpi,bins=20,show_titles=False, plot_datapoints=False,
			label_kwargs=dict(fontsize=fontsize),
			truths=truths,levels=[0.68,0.95],color=color,fill_contours=True,
			truth_color=truth_color,fig=figure,range=plot_range,
			hist_kwargs=hist_kwargs)
		return figure

	def plot_cov_corner(self,burnin,hyperparam_plot_names=None,
		block=True,color='#FFAA00',truth_color='#000000',figures=None,
		plot_range=None,dpi=200,fontsize=13):
		"""
		Plot the corner plot of chains resulting from the emcee of the
		covariance matrix parameters for population inference.

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			plot_param (str): The lens parameter to plot the hyperprior
				posteriors for.
			hyperparam_plot_names ([str,...]): A list containing the names
				of the hyperparameters to be used during plotting
			block (bool): If true, block excecution after plt.show() command
			color (str): The color to use for plotting the contour.
			truth_color (str): The color to use for plotting the truths in the
				corner plot.
			figure [(matplotlib.pyplot.figure),...]: A 2 elements list of
				figures that were previously returned by plot_cov_corner to
				overplot onto. Two figure are required since we will make a
				corner plot for the mean and the covariance.
			dpi (int): The dpi of the figure.
			fontsize (int): The fontsize for the corner plot labels.
		"""
		if hyperparam_plot_names is None:
			hyperparam_plot_names = self.target_eval_dict['hyp_names']

		# We need the Nones to be in list form
		if figures is None:
			figures = [None,None]

		# Get the chains from the samples
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(hyperparam_plot_names))

		# First plot the posterior on the covariance matrix mean.
		mu_ind = self.target_eval_dict['cov_mu_hyp_ind']
		mu_hyp_s = np.min(mu_ind)
		mu_hyp_e = np.max(mu_ind)+1
		truths = None
		if self.true_hyp_values is not None:
			truths = self.true_hyp_values[mu_hyp_s:mu_hyp_e]
		hist_kwargs = {'density':True,'color':color}
		figures[0] = corner.corner(chains[:,mu_hyp_s:mu_hyp_e],
			labels=hyperparam_plot_names[mu_hyp_s:mu_hyp_e],
			dpi=dpi,bins=20,show_titles=False, plot_datapoints=False,
			label_kwargs=dict(fontsize=fontsize),
			truths=truths,levels=[0.68,0.95],color=color,fill_contours=True,
			truth_color=truth_color,fig=figures[0],range=plot_range,
			hist_kwargs=hist_kwargs)

		# Now repeat the same but for the covariance parameters. We will
		# convert from the lower triangular matrix to the covariance matrix.
		tril_ind = self.target_eval_dict['cov_tril_hyp_ind']
		tril_hyp_s = np.min(tril_ind)
		tril_hyp_e = np.max(tril_ind)+1
		truths = None
		if self.true_hyp_values is not None:
			truths = self.true_hyp_values[tril_hyp_s:tril_hyp_e]

		# Convert posterior samples and truths to of the lower triangular
		# matrix to covariance matrix samples.
		tril_mask = np.tri(len(mu_ind),dtype=bool, k=0)
		cov_truth = convert_tril_to_cov(truths)
		cov_truth = cov_truth[tril_mask]
		chains_covs = convert_trils_to_covs(chains[:,tril_hyp_s:tril_hyp_e])
		chains_covs = chains_covs[:,tril_mask]

		hist_kwargs = {'density':True,'color':color}
		figures[1]= corner.corner(chains_covs,
			labels=hyperparam_plot_names[tril_hyp_s:tril_hyp_e],
			dpi=dpi,bins=20,show_titles=False, plot_datapoints=False,
			label_kwargs=dict(fontsize=fontsize),
			truths=cov_truth,levels=[0.68,0.95],color=color,fill_contours=True,
			truth_color=truth_color,fig=figures[1],range=plot_range,
			hist_kwargs=hist_kwargs)

		return figures

	def plot_distributions(self,burnin,param_plot_names=None,block=True,
		color_map=["#a1dab4","#41b6c4","#2c7fb8","#253494"],bnn_name='BNN',
		dpi=200,plot_param=None,save_fig_path=None):
		"""
		Plot the posteriors from our MCMC sampling of Omega.

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			param_plot_names ([str,...]): A list containing the names
				of the parameters to be used during plotting
			block (bool): If true, block excecution after plt.show() command
			color_map ([str,...]): The colors for the samples, the posterior
				distribution samples, the true distribution, and the interim
				distribution respectively.
			bnn_name (str): The BNN name to use for the plot titles.
			dpi (int): The resolution of the output figures
			plot_param (str): The lens parameter to plot the distributions
				for. If None all lens parameters will be used.
			save_fig_path (str): The path to save the figure to. If no
				plot_param is specified only the last parameter will be saved.
		"""
		hyperparam_plot_names = self.target_eval_dict['hyp_names']
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(hyperparam_plot_names))
		global lens_samps

		for li, lens_param in enumerate(self.lens_params_test):
			# If only a specific lens parameter is specified skip the others
			if plot_param is not None and not lens_param == plot_param:
				continue
			# In the case of a multinormal for the test distribution, some
			# parameters will have empty dictionary entries
			if self.target_eval_dict[lens_param] is None:
				continue
			# Grab the samples and indices of the hyperparameters
			samples = lens_samps[li].flatten()
			hyp_ind = self.target_eval_dict[lens_param]['hyp_ind']
			if hyp_ind.size == 0:
				continue
			plt_min = max(np.mean(samples)-6*np.std(samples),
				np.min(samples)-0.1)
			plt_max = min(np.mean(samples)+6*np.std(samples),
				np.max(samples)+0.1)
			plt.figure(figsize=(8.5,5), dpi=dpi)

			# Select the range of parameter values to evaluate the pdf at
			eval_pdf_at = np.linspace(plt_min,plt_max,1000)

			# Plot the interim distribution these parameters were being drawn
			# from.
			hyp_ind_int = self.interim_eval_dict[lens_param]['hyp_ind']
			hyp_s = np.min(hyp_ind_int)
			hyp_e = np.max(hyp_ind_int)+1
			truth_eval = np.exp(self.interim_eval_dict[lens_param]['eval_fn'](
				eval_pdf_at,*self.interim_eval_dict['hyp_values'][hyp_s:hyp_e],
				**self.interim_eval_dict[lens_param]['eval_fn_kwargs']))
			plt.plot(eval_pdf_at,truth_eval,color=color_map[0], lw=2.5,
				ls=':')

			# Plot the samples for the parameter
			hyp_s = np.min(hyp_ind)
			hyp_e = np.max(hyp_ind)+1
			plt.hist(samples,bins=100,density=True,align='mid',
				color=color_map[1],range=(plt_min,plt_max))

			# Sample 100 chains and plot them
			n_chains_plot = 50
			for chain in chains[np.random.randint(len(chains),size=n_chains_plot)]:
				chain_eval = np.exp(self.target_eval_dict[lens_param]['eval_fn'](
					eval_pdf_at,*chain[hyp_s:hyp_e],
					**self.target_eval_dict[lens_param]['eval_fn_kwargs']))
				plt.plot(eval_pdf_at,chain_eval,color=color_map[2], lw=2,
					alpha=5/n_chains_plot)

			# Plot the true distribution these parameters were being drawn
			# from.
			truth_eval = np.exp(self.target_eval_dict[lens_param]['eval_fn'](
				eval_pdf_at,*self.true_hyp_values[hyp_s:hyp_e],
				**self.target_eval_dict[lens_param]['eval_fn_kwargs']))
			plt.plot(eval_pdf_at,truth_eval,color=color_map[3], lw=2.5,
				ls='--')

			# Construct the legend.
			custom_lines = [Line2D([0], [0], color=color_map[0], lw=4),
				Line2D([0], [0], color=color_map[1], lw=4),
				Line2D([0], [0], color=color_map[2], lw=4),
				Line2D([0], [0], color=color_map[3], lw=4)]
			plt.legend(custom_lines, [
				r'Training Distribution: $p(\xi^{\star}|\Omega_\mathrm{int})$',
				r'Stacked BNN Samples: $p(\{\xi\}|\{d\},\Omega_\mathrm{int})$',
				r'Posterior Samples: $p(\xi^{\star}|\Omega)p(\Omega|\{d\}) $',
				r'Test Distribution: $p(\xi^{\star}|\Omega_\mathrm{test})$'])

			if param_plot_names is None:
				plt.xlabel(lens_param)
				plt.ylabel('p(%s)'%(lens_param))
				plt.title('%s Distribtuions for p(%s)'%(bnn_name,lens_param))
			else:
				plt.xlabel(param_plot_names[li])
				plt.ylabel('p(%s)'%(param_plot_names[li]))
				plt.title('%s Distribtuions for p(%s)'%(bnn_name,
					param_plot_names[li]))
			plt.xlim([plt_min,plt_max])

			if save_fig_path is not None:
				plt.savefig(save_fig_path)

			plt.show(block=block)

	def calculate_sample_log_weights(self,n_p_omega_samps,burnin):
		"""
		Calculate the weights from the posterior on Omega needed for
		reweighting.

		Parameters
		----------
			n_p_omega_samps (int): The number of samples from p(Omega|{d}) to
				use in the reweighting.

		Returns
		-------
			(np.array): The log of the weights for the lens samples.
		"""
		# Define the global variables we'll use.
		global lens_samps
		# First we'll get samples from the chain we'll use to reweight our
		# contour.
		samples = self.sampler.get_chain()[burnin:].reshape(-1,
			self.target_eval_dict['hyp_len'])
		samples = samples[np.random.choice(samples.shape[0],
			size=n_p_omega_samps,replace=False)]
		# Calculate the log posterior on xi required for the reweighting term
		# for each sample.
		lpos = np.zeros((n_p_omega_samps,lens_samps.shape[1],
			lens_samps.shape[2]))
		for s_i, sample in enumerate(samples):
			lpos[s_i] = log_p_xi_omega(lens_samps, sample, self.target_eval_dict,
				self.lens_params_test)
		lpi = self.prob_class.pt_omegai
		# Calculate the log weights using the log posterior samples
		log_weights = special.logsumexp(lpos-lpi,axis=0)-np.log(n_p_omega_samps)
		return log_weights

	def plot_reweighted_lens_posterior(self,burnin,image_index,plot_limits=None,
		n_p_omega_samps=100, color_map=["#FFAA00","#41b6c4"],
		block=True,truth_color='#000000'):
		"""
		Plot the original and reweighted posterior contours for a specific image
		along with the image itself.

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			image_index (int): The integer index of the image in the validation
				set to plot the posterior of.
			plot_limits ([(float,float),..]): A list of float tuples that define
				the maximum and minimum plot range for each posterior parameter.
			n_p_omega_samps (int): The number of samples from p(Omega|{d}) to
				use in the reweighting.
			color_map ([str,...]): The colors to use in the contour plotting.
			block (bool): If true, block excecution after plt.show() command
			truth_color (str): The color to use for plotting the truths in the
				corner plot.
		"""
		# Plot the contours without the reweighting first.
		fig = corner.corner(self.infer_class.predict_samps[:,image_index,:],
				bins=20, labels=self.infer_class.final_params_print_names,
				show_titles=False,plot_datapoints=False,
				label_kwargs=dict(fontsize=13),
				truths=self.infer_class.y_test[image_index],levels=[0.68,0.95],
				dpi=1600, color=color_map[0],fill_contours=True,
				range=plot_limits,truth_color=truth_color)

		log_weights = self.calculate_sample_log_weights(n_p_omega_samps,burnin)
		log_weights -= special.logsumexp(log_weights,axis=0)
		weights = np.exp(log_weights)[:,image_index]

		corner.corner(self.infer_class.predict_samps[:,image_index,:],bins=20,
				labels=self.infer_class.final_params_print_names,
				show_titles=False,plot_datapoints=False,
				label_kwargs=dict(fontsize=13),
				truths=self.infer_class.y_test[image_index],levels=[0.68,0.95],
				dpi=1600, color=color_map[1],fill_contours=True,
				weights=weights, fig=fig,range=plot_limits,
				truth_color=truth_color)

	def plot_reweighted_calibration(self,burnin,n_perc_points=20,
		n_p_omega_samps=100,color_map=['#000000','#1b9e77','#d95f02'],
		legend=['Perfect Calibration','Bare Network','Reweighted Network'],
		ls_list=['-','--'],figure=None):
		"""
		Plot the calibration plot reweighted using the samples of Omega

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			n_perc_point (int): The number of percentages to probe in the
				plotting.
			n_p_omega_samps (int): The number of samples from p(Omega|{d}) to
				use in the reweighting.
			color_map ([str,...]): The colors to use in the calibration
				plots. Must include 3 colors. They are the colors for the
				perfect calibration line, the orginial calibration (with
				the interim prior assumption), and the reweighted calibration.
			legend ([str,...]): The legend to use when labeling the
				output lines
			ls_list ([str,...]): A list of line styles to use for each
				line type.
			figure (matplotlib.pyplot.figure): A figure that was previously
				returned by plot_calibration to overplot onto.

		Returns
		-------
			(matplotlib.pyplot.figure): The figure object that contains the
				plot
		"""
		# For the first plot we can just use the original BNN code.
		figure = self.infer_class.plot_calibration(color_map=color_map,
			n_perc_points=n_perc_points,show_plot=False,legend=legend,
			ls=ls_list[0],figure=figure)

		# Grab a sample of the weights we will use on the posteriors.
		log_weights = self.calculate_sample_log_weights(n_p_omega_samps,burnin)
		log_weights -= special.logsumexp(log_weights,axis=0)-np.log(
			len(log_weights))
		weights = np.exp(log_weights)

		# Recreate the same calibration plot with the new weights folded
		# in.
		figure = self.infer_class.plot_calibration(color_map=color_map[1:],
			n_perc_points=n_perc_points,figure=figure,show_plot=False,
			weights=weights,legend=legend,ls=ls_list[1])

		return figure

	def plot_parameter_distribtuion(self,burnin,lens_params,n_p_omega_samps=100,
		samps_per_omega=100, param_print_names=None,color='#000000',
		fontsize=13,plot_limits=None,figure=None, cov_params=False):
		"""
		For the desired lens parameters, plot a corner plot of the distribution
		of each parameter.

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			lens_params ([str,...]): The lens params to plot the distributions
				for.
			n_p_omega_samps (int): The number of samples from p(Omega|{d}) to
				use.
			samps_per_omega (int): How many samples to pull for each omega.
			param_print_names ([str,...]): A list of string matching lens_params
				with the string that should be used for the parameter name
				in plotting (allows pretty labels).
			color (str): The color to use for plotting the contour.
			fontsize (int): The fontsize to use for labeling.
			plot_limits ([(float,float),..]): A list of float tuples that define
				the maximum and minimum plot range for each posterior parameter.
			figure (matplotlib.pyplot.figure): A figure that was previously
				returned by plot_single_corner to overplot onto.
			cov_params (bool): If true, will also attempt to print
				the distribution for the covariate parameters.
		"""
		# Grab the chains with burnin
		chains = self.sampler.get_chain()[burnin:]
		chains = chains.reshape(-1,chains.shape[-1])
		corner_param_samples = []

		for li, lens_param in enumerate(lens_params):
			# In the case of a multinormal for the test distribution, some
			# parameters will have empty dictionary entries
			if self.target_eval_dict[lens_param] is None:
				continue

			# Grab the indices of the hyperparameters
			hyp_ind = self.target_eval_dict[lens_param]['hyp_ind']
			hyp_s = np.min(hyp_ind)
			hyp_e = np.max(hyp_ind)+1

			# Pull the function to sample from
			samp_fn = getattr(distributions,'sample_{}_vectorize'.format(
				self.target_eval_dict[lens_param]['fn_name']))
			param_samps = []

			# Sample the desired number of population hyperparameter values and
			# sample the desired number of lens parameter value for each one.
			for chain in chains[np.random.randint(len(chains),
				size=n_p_omega_samps)]:
				param_samps.append(samp_fn(samps_per_omega,*chain[hyp_s:hyp_e],
					**self.target_eval_dict[lens_param]['eval_fn_kwargs']))
			corner_param_samples.append(np.array(param_samps).flatten())

		# Numpy arrays are easier to work with.
		corner_param_samples = np.array(corner_param_samples).T

		# Deal with covariate parameters
		if cov_params:
			# Get the indices we'll use for sampling
			mu_ind = self.target_eval_dict['cov_mu_hyp_ind']
			tril_ind = self.target_eval_dict['cov_tril_hyp_ind']
			param_samps = []

			# We will want to impose a cut on any samples above 1 in qi
			qi = self.target_eval_dict['cov_params_list'].index('lens_mass_q')

			# Sample the desired number of population hyperparameter values and
			# sample the desired number of lens parameter value for each one.
			for chain in chains[np.random.randint(len(chains),
				size=n_p_omega_samps)]:
				# Grab mu and tril and turn the latter into the covariance
				# matrix.
				mu = chain[mu_ind]
				tril = chain[tril_ind]
				cov = convert_tril_to_cov(tril)
				cov_samps = np.exp(np.random.multivariate_normal(mu,cov,
					size=samps_per_omega))

				# Resample and points with q>1.
				while np.sum(cov_samps[:,qi]>1)>0:
					cov_samps[cov_samps[:,qi]>1] = np.exp(
						np.random.multivariate_normal(mu,cov,
							size=np.sum(cov_samps[:,qi]>1)))

				# Append the samples to our list once we've dealth with the
				# resampling.
				param_samps.append(cov_samps)

			param_samps = np.array(param_samps).reshape(-1,len(mu))
			corner_param_samples = np.concatenate([corner_param_samples,
				param_samps],axis=1)

		# Numpy arrays are easier to work with.
		corner_param_samples = np.array(corner_param_samples)

		# Now plot our samples
		if param_print_names is None:
			param_print_names = lens_params

		hist_kwargs = {'density':True,'color':color}
		corner.corner(corner_param_samples,labels=param_print_names,bins=30,
			show_titles=False, plot_datapoints=False,
			label_kwargs=dict(fontsize=fontsize),color=color,levels=[0.68,0.95],
			fill_contours=True,fig=figure,range=plot_limits,
			hist_kwargs=hist_kwargs)
