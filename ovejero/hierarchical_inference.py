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
from scipy import stats, special
from baobab import configs
from baobab import distributions
from ovejero import bnn_inference

def eval_log_norm_log_pdf(samples,hyp):
	"""
	Evaluate the log normal log pdf with the hyperparameters provided.

	Parameters
	----------
		samples (np.array): A numpy array with dimensions (n_samps,batch_size)
		hyperparameters (np.array): A numpy array containing the mean and 
			sigma

	Returns
	-------
		np.array: The pdf at each point.
	"""
	mu = np.exp(hyp[0])
	sigma = hyp[1]
	dist = stats.lognorm(scale=mu,s=sigma)
	return dist.logpdf(samples)

def eval_norm_log_pdf(samples,hyp):
	"""
	Evaluate the normal log pdf with the hyperparameters provided.

	Parameters
	----------
		samples (np.array): A numpy array with dimensions (n_samps,batch_size)
		hyp (np.array): A numpy array containing the mean and sigma

	Returns
	-------
		np.array: The log pdf at each point.
	"""
	mu,sigma = hyp
	dist = stats.norm(loc=mu,scale=sigma)
	return dist.logpdf(samples)

def eval_beta_log_pdf(samples,hyp):
	"""
	Evaluate the beta log pdf with the hyperparameters provided.

	Parameters
	----------
		samples (np.array): A numpy array with dimensions (n_samps,batch_size)
		hyp (np.array): A numpy array containing [a,b,lower,upper].

	Returns
	-------
		np.array: The log pdf at each point.
	"""
	a,b,lower,upper = hyp
	dist = stats.beta(a=a, b=b, loc=lower,scale=upper-lower)
	return dist.logpdf(samples)

def eval_gen_norm_log_pdf(samples,hyp):
	"""
	Evaluate the generalized normal pdf with the hyperparameters provided.

	Parameters
	----------
		samples (np.array): A numpy array with dimensions (n_samps,batch_size)
		hyp (np.array): A numpy array containing [mu,alpha,p,lower,upper].

	Returns
	-------
		np.array: The pdf at each point.
	"""
	mu, alpha, p, lower, upper = hyp
	dist = stats.gennorm(beta=p, loc=mu, scale=alpha)
	logpdf = dist.logpdf(samples)
	logpdf /= dist.cdf(upper) - dist.cdf(lower)
	logpdf[samples<lower] = -np.inf; logpdf[samples>upper] = -np.inf
	return logpdf


def build_evaluation_dictionary(baobab_cfg,lens_params,extract_hyperpriors=False):
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

		# We are going to read through the config file here and build a mapping
		# between an array of hyperparameter values that our mcmc will return
		# and the pdf on the data. To do this we will have to read through the
		# distribution for each parameter and build a function we can evaluate 
		# to get the pdf. We will also store the values for the distribution
		# parameters in the config file, since these will be important to know
		# for the interim prior. Finally, we also need to know the hyperpriors
		# we're imposing on the distribution parameters in Omega. We will extract
		# this if extract_hyperpriors is set to true. 
		if dist['dist'] == 'normal':
			# Normal has two hyperparameters
			eval_dict[lens_param]['hyp_ind'] = np.arange(eval_dict['hyp_len'],
				eval_dict['hyp_len']+2)
			eval_dict['hyp_len'] += 2
			# Add the value of the parameters in the config file
			eval_dict['hyps'].extend([dist['mu'],dist['sigma']])
			eval_dict['hyp_names'].extend([lens_param+':mu',lens_param+':sigma'])
			if extract_hyperpriors:
				hypp = dist['hyper_prior']
				eval_dict['hyp_prior'].extend([hypp['mu'],hypp['sigma']])
			# For a normal distribution we want the sigma parameter to be 
			# calculated in log space, and we need to check if it is normal
			# or log normal.
			if dist['log'] == True:
				eval_dict[lens_param]['eval_fn'] = eval_log_norm_log_pdf
			else:
				eval_dict[lens_param]['eval_fn'] = eval_norm_log_pdf
			# 

		elif dist['dist'] == 'beta':
			# Beta has four parameters - a, b, lower, and upper
			eval_dict[lens_param]['hyp_ind'] = np.arange(eval_dict['hyp_len'],
				eval_dict['hyp_len']+4)
			eval_dict['hyp_len'] += 4
			# Add the value of the parameters in the config file
			eval_dict['hyps'].extend([dist['a'],dist['b'],dist['lower'],
				dist['upper']])
			eval_dict['hyp_names'].extend([lens_param+':a',lens_param+':b',
				lens_param+':lower',lens_param+':upper'])
			if extract_hyperpriors:
				hypp = dist['hyper_prior']
				eval_dict['hyp_prior'].extend([hypp['a'],hypp['b'],hypp['lower'],
					hypp['upper']])
			eval_dict[lens_param]['eval_fn'] = eval_beta_log_pdf

		elif dist['dist'] == 'generalized_normal':
			# A generalized normal distribution has five parameters - mu, alpha,
			# p, lower, and upper.
			eval_dict[lens_param]['hyp_ind'] = np.arange(eval_dict['hyp_len'],
				eval_dict['hyp_len']+5)
			eval_dict['hyp_len'] += 5
			# Add the value of the parameters in the config file
			eval_dict['hyps'].extend([dist['mu'],dist['alpha'],dist['p'],
				dist['lower'],dist['upper']])
			eval_dict['hyp_names'].extend([lens_param+':mu',lens_param+':alpha',
				lens_param+':p',lens_param+':lower',lens_param+':upper'])
			if extract_hyperpriors:
				hypp = dist['hyper_prior']
				eval_dict['hyp_prior'].extend([hypp['mu'],hypp['alpha'],hypp['p'],
					hypp['lower'],hypp['upper']])
			eval_dict[lens_param]['eval_fn'] = eval_gen_norm_log_pdf

		else:
			raise RuntimeError('Distribution %s is not an option'%(dist['dist']))

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
				hyp[eval_dict[lens_param]['hyp_ind']])

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





		