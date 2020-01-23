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
from ovejero import bnn_inference
from inspect import signature
from tqdm import tqdm
import emcee, pickle, os

def build_evaluation_dictionary(baobab_cfg,lens_params,
	extract_hyperpriors=False):
	"""
	Map between the baobab config and a dictionary that contains the 
	evaluation function and hyperparameter indices for each parameters.

	Parameters
	----------
		baobab_cfg (dict): The baobab configuration file containing the 
			distributions we want to consider for each parameter.
		lens_params ([str,...]): The parameters to pull the distributions from.
		extract_hyperpriors (bool): If set to true, the function will attempt to
			read the hyperprios from the config file. For this to work, the
			config file must include priors for each parameter.
	Returns
	-------
		dict: The evaluation dictionary used by HierarchicalClass functions to
			calculate the log pdf.

	Notes
	-----
		This will be automatically forward compatible with new distributions in 
		baobab. Only uniform hyperpriors are supported for now.
	"""

	# Initialize eval_dict with empty lists for hyperparameters and hyperpriors.
	eval_dict = dict(hyp_len = 0, hyps = [], hyp_prior = [], hyp_names=[])
	# For each lens parameter add the required hyperparameters and evaluation
	# function.
	for lens_param in lens_params:
		# Make a new entry for the parameter
		eval_dict[lens_param] = dict()

		# Load the config entry associated to this parameter
		lens_split = lens_param.split('_')
		# This is a bit ugly, but it's the quickest way to map between the
		# parameter name and the location in the config file.
		dist = baobab_cfg.bnn_omega['_'.join(lens_split[:2])][
			'_'.join(lens_split[2:])]

		# Get the function in question from lenstronomy.
		eval_fn = getattr(distributions,'eval_{}_logpdf'.format(dist['dist']))
		eval_sig = signature(eval_fn)

		# Hyperparameters is number of parameters-1 because the first parameter
		# is where to evaluate.
		n_hyps = len(eval_sig.parameters)-1

		# Add the value of the hyperparameters in the config file. Again skip
		# the first one.
		for hyperparam_name in list(eval_sig.parameters.keys())[1:]:
			# Sometimes hyperparameters are not specified
			if dist[hyperparam_name] == {}:
				n_hyps -= 1
			else:
				eval_dict['hyps'].extend([dist[hyperparam_name]])
				eval_dict['hyp_names'].extend([lens_param+':'+hyperparam_name])
				if extract_hyperpriors:
					hypp = dist['hyper_prior']
					eval_dict['hyp_prior'].extend([hypp[hyperparam_name]])

		eval_dict[lens_param]['hyp_ind'] = np.arange(eval_dict['hyp_len'],
			eval_dict['hyp_len']+n_hyps)
		eval_dict['hyp_len'] += n_hyps

		# Finally, actually include the evaluation function.
		eval_dict[lens_param]['eval_fn'] = eval_fn

	# Transform list of initial values into np array
	eval_dict['hyps'] = np.array(eval_dict['hyps'])
	eval_dict['hyp_prior'] = np.array(eval_dict['hyp_prior']).T

	return eval_dict


class HierarchicalClass:
	"""
	A class that contains all of the functions needed to conduct a hierarchical
	calculation of the lens parameter distributions.
	"""
	def __init__(self,cfg,interim_baobab_omega_path,target_baobab_omega_path):
		"""
		Initialize the HierarchicalClass instance using the parameters of the
		configuration file.

		Parameters
		----------
			cfg (dict): The dictionary attained from reading the json config file.
			interim_baobab_omega_path (str): The string specifying the path to the
				baobab config for the interim omega. 
			target_baobab_omega_path (str): The string specifying the path to the 
				baobab config for the target omega. The exact value of the
				distribution parameters in omega will be used as
				intitialization points for the mc chains.
		"""
		self.cfg = cfg
		# Pull the needed param information from the config file.
		self.lens_params = cfg['dataset_params']['lens_params']
		self.lens_params_log = cfg['dataset_params']['lens_params_log']
		self.gampsi = cfg['dataset_params']['gampsi']
		self.final_params = cfg['training_params']['final_params']
		self.interim_baobab_omega = configs.BaobabConfig.from_file(
			interim_baobab_omega_path)
		self.target_baobab_omega = configs.BaobabConfig.from_file(
			target_baobab_omega_path)
		self.num_params = len(self.lens_params)
		# Build the evaluation dictionaries from the 
		self.interim_eval_dict = build_evaluation_dictionary(
			self.interim_baobab_omega, self.lens_params, 
			extract_hyperpriors=False)
		self.target_eval_dict = build_evaluation_dictionary(
			self.target_baobab_omega, self.lens_params, 
			extract_hyperpriors=True)
		# Make our inference class we'll use to generate samples.
		self.infer_class = bnn_inference.InferenceClass(self.cfg)
		self.samples_init = False
		# The probability of the data given the interim prior. Will be
		# generated along with the samples.
		self.pt_omegai = None
		# Track if the sampler has been initialzied yet.
		self.sampler_init = False

	def log_p_theta_omega(self, samples, hyp, eval_dict):
		"""
		Calculate log p(theta|omega) - the probability of the lens parameters in
		the data given the proposed lens parameter distribution.

		Parameters
		----------
			samples (np.array): A numpy array with dimensions (n_samps,batch_size,
				n_params)
			hyp (np.array): A numpy array with dimensions (n_hyperparameters). 
				These are the hyperparameters that will be used for evaluation.
			eval_dict (dict): A dictionary from build_evaluation_dictionary to
				query for the evaluation functions.
			lens_params ([str,...]): A list of strings of the lens parameters
				generated by baobab.

		Returns
		-------
			np.array: A numpy array of the shape (n_samps,batch_size) containing
				the log p(theta | omega) for each sample.
		"""
		
		# We iterate through each lens parameter and carry out the evaluation

		# TODO: Make this faster. .1 seconds is a bit slow.
		logpdf = np.zeros((samples.shape[0],samples.shape[1]))

		for li in range(len(self.lens_params)):
			lens_param = self.lens_params[li]
			logpdf += eval_dict[lens_param]['eval_fn'](samples[:,:,li],
				*hyp[eval_dict[lens_param]['hyp_ind']])

		return logpdf

	def log_p_omega(self,hyp):
		"""
		Calculate log p(omega) - the probability of the hyperparameters given
			the hyperprior.

		Parameters
		----------
			hyp (np.array): A numpy array with dimensions (n_hyperparameters). 
				These are the hyperparameters that will be used for evaluation.

		Returns
		-------
			np.float: The value of log p(omega)
		"""

		# Currently this only supports the uniform priors from eval_dict.
		if np.any(hyp<self.target_eval_dict['hyp_prior'][0]):
			return -np.inf
		elif np.any(hyp>self.target_eval_dict['hyp_prior'][1]):
			return -np.inf
		# No need to calculate the volume since this is just a constant.
		return 0

	def gen_samples(self,num_samples):
		"""
		Generate samples of lens parameters theta for use in hierarchical
		inference.

		Parameters
		----------
			num_samples (int): The number of samples to draw per lens.
		"""

		# Most of the work will be done by the InferenceClass. The only
		# additional work we'll do here is undoing the polar to cartesian
		# transformation and the log transformation.
		self.infer_class.gen_samples(num_samples)

		# We'll steal the samples from bnn_infer and transform them
		# back into the original parameter space.
		self.predict_samps = self.infer_class.predict_samps
		self.lens_samps = np.zeros_like(self.predict_samps)

		# Even for parameters that have not been changed between baobab
		# and training time, we still need to make sure their position
		# in the new array is correct.
		for f_param in self.final_params:
			if f_param in self.lens_params:
				p_l = self.lens_params.index(f_param)
				p_f = self.final_params.index(f_param)
				self.lens_samps[:,:,p_l] = self.predict_samps[:,:,p_f]

		# Go through all the parameters that were put into log space and
		# put them back into their original form.
		for log_param in self.lens_params_log:
			log_p_l = self.lens_params.index(log_param)
			log_p_f = self.final_params.index(log_param+'_log')
			self.lens_samps[:,:,log_p_l] = np.exp(
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
			rati = self.lens_params.index(param_rat)
			angi = self.lens_params.index(param_ang)

			self.lens_samps[:,:,rati] = np.sqrt(np.square(
				self.predict_samps[:,:,pg1i]) + np.square(
				self.predict_samps[:,:,pg2i]))
			self.lens_samps[:,:,angi] = np.arctan2(
				self.predict_samps[:,:,pg2i],
				self.predict_samps[:,:,pg1i])/2

		# TODO: We need a much better way to deal with this! But for now
		# we're just going to force it.
		hard_fix_params = ['lens_mass_e1','lens_mass_e2']
		for lens_param in hard_fix_params:
			fixi = self.lens_params.index(lens_param)
			self.lens_samps[self.lens_samps[:,:,fixi]<-0.55,fixi] = -0.55+1e-5
			self.lens_samps[self.lens_samps[:,:,fixi]>0.55,fixi] = 0.55-1e-5

		self.pt_omegai = self.log_p_theta_omega(self.lens_samps,
			self.target_eval_dict['hyps'], self.target_eval_dict)

		self.samples_init = True

	def log_post_omega(self, hyp):
		"""
		Given generated data, calculate the log posterior of omega.

		Parameters
		----------
			hyp (np.array): A numpy array with dimensions (n_hyperparameters). 
				These are the values of omega.

		Returns
		-------
			(float): The log posterior of omega given the data.

		Notes
		-----
			Constant factors with respect to omega are ignored.
		"""
		# Add the prior on omega
		lprior = self.log_p_omega(hyp)

		# Calculate the probability of each datapoint given omega
		pt_omega = self.log_p_theta_omega(self.lens_samps, hyp, 
			self.target_eval_dict)
		# We've already calculated the value of pt_omegai when we generated
		# the samples. Now we just need to sum them correctly.
		like_ratio = special.logsumexp(pt_omega-self.pt_omegai,axis=0)
		like_ratio[np.isnan(like_ratio)] = -np.inf

		return lprior + np.sum(like_ratio)

	def initialize_sampler(self,n_walkers,save_path):
		"""
		Initialize the sampler to be used by run_samples.

		
		Parameters
		----------
			n_walkers (int): The number of walkers used by the sampler.
				Must be at least twice the number of hyperparameters.
			save_path (str): A pickle path specifying where to save the 
				sampler chains. If a sampler chain is already present in the
				path it will be loaded.
		"""
		nwalkers = 50
		ndim = self.target_eval_dict['hyp_len']
		if os.path.isfile(save_path):
			self.cur_state = None
		else:
			self.cur_state = (np.random.rand(nwalkers, ndim)*0.1 + 
				self.target_eval_dict['hyps'])

		backend = emcee.backends.HDFBackend(save_path)

		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, 
			self.log_post_omega, backend=backend)

		self.sampler_init = True

	def run_samples(self,n_samps,save_path):
		"""
		Run an emcee sampler to get a posterior on the hyperparameters.

		Parameters
		----------
			n_samps (int): The number of samples to take
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before inference')
		if self.sampler_init == False:
			raise RuntimeError('Must initialize sampler before running sampler')

		self.sampler.run_mcmc(self.cur_state,n_samps)





		