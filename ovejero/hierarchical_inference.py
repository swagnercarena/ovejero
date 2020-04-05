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
import emcee, os, glob, corner, copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# This is ugly, but we have to set lens samples as a global variable for the
# pooling to work well. Otherwise the data is pickled and unpickled each time
# one of the cpus calls the log likelihood function, slowing things down.
lens_samps = None


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
	eval_dict = dict(hyp_len=0, hyps=[], hyp_prior=[], hyp_names=[])
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
		eval_fn = getattr(distributions,'eval_{}_logpdf_approx'.format(
			dist['dist']))
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

	# TODO: Make this faster. .1 seconds is a bit slow.
	logpdf = np.zeros((samples.shape[1],samples.shape[2]))

	for li in range(len(lens_params)):
		lens_param = lens_params[li]
		logpdf += eval_dict[lens_param]['eval_fn'](samples[li],
			*hyp[eval_dict[lens_param]['hyp_ind']])
	logpdf[np.isnan(logpdf)] = -np.inf

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
		lens_params ([str,....]): A list of strings containing the lens params
			that should be written out as features
	"""
	def __init__(self,target_eval_dict,interim_eval_dict,lens_params):
		# Save the parameters we'll need.
		self.target_eval_dict = copy.deepcopy(target_eval_dict)
		self.interim_eval_dict = copy.deepcopy(interim_eval_dict)
		self.lens_params = copy.copy(lens_params)
		# The samples_init flag.
		self.samples_init = False
		# The probability of the data given the interim prior. Will be
		# calculated when the samples are passed in.
		self.pt_omegai = None

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

	def set_samples(self):
		"""
		Set the lens samples that will be used for the log_post_omega
			calculation.
		"""
		# Now that we have the samples, we can calculate the probability
		# of the samples given the interim prior.
		global lens_samps
		self.pt_omegai = log_p_xi_omega(lens_samps,
			self.interim_eval_dict['hyps'], self.interim_eval_dict,
			self.lens_params)
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
		lprior = self.log_p_omega(hyp)

		if lprior == -np.inf:
			return lprior

		# Calculate the probability of each datapoint given omega
		pt_omega = log_p_xi_omega(lens_samps, hyp, self.target_eval_dict,
			self.lens_params)
		# We've already calculated the value of pt_omegai when we generated
		# the samples. Now we just need to sum them correctly. Note that the
		# first axis in samples has dimension number of samples, so that is
		# what we want to log sum exp over.
		like_ratio = special.logsumexp(pt_omega-self.pt_omegai,axis=0)
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
		target_baobab_omega_path (str): The string specifying the path to the
			baobab config for the target omega. The exact value of the
			distribution parameters in omega will be used as intitialization
			points for the mc chains.
		test_dataset_path (str): The path to the dataset on which hiearchical
			inference will be conducted
		test_dataset_tf_record_path (str): The path where the TFRecord will be
			saved. If it already exists it will be loaded.
	"""
	def __init__(self,cfg,interim_baobab_omega_path,target_baobab_omega_path,
		test_dataset_path,test_dataset_tf_record_path):
		# Initialzie our class.
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
		self.norm_images = cfg['training_params']['norm_images']
		n_npy_files = len(glob.glob(os.path.join(test_dataset_path,'X*.npy')))
		self.cfg['training_params']['batch_size'] = n_npy_files
		# Build the evaluation dictionaries from the
		self.interim_eval_dict = build_evaluation_dictionary(
			self.interim_baobab_omega, self.lens_params,
			extract_hyperpriors=False)
		self.target_eval_dict = build_evaluation_dictionary(
			self.target_baobab_omega, self.lens_params,
			extract_hyperpriors=True)
		# Make our inference class we'll use to generate samples.
		self.infer_class = bnn_inference.InferenceClass(self.cfg)

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
			self.interim_eval_dict,self.lens_params)

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
		return self.prob_class.log_p_omega(hyp)

	def gen_samples(self,num_samples,sample_save_dir=None):
		"""
		Generate samples of lens parameters xi for use in hierarchical
		inference.

		Parameters
		----------
			num_samples (int): The number of samples to draw per lens.
			sample_save_dir (str): A path to a folder to save/load the samples.
				If None samples will not be saved. Do not include .npy, this will
				be appended (since several files will be generated).
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
				if f_param in self.lens_params:
					p_l = self.lens_params.index(f_param)
					p_f = self.final_params.index(f_param)
					lens_samps[:,:,p_l] = self.predict_samps[:,:,p_f]

			# Go through all the parameters that were put into log space and
			# put them back into their original form.
			for log_param in self.lens_params_log:
				log_p_l = self.lens_params.index(log_param)
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
				rati = self.lens_params.index(param_rat)
				angi = self.lens_params.index(param_ang)

				lens_samps[:,:,rati] = np.sqrt(np.square(
					self.predict_samps[:,:,pg1i]) + np.square(
					self.predict_samps[:,:,pg2i]))
				lens_samps[:,:,angi] = np.arctan2(
					self.predict_samps[:,:,pg2i],
					self.predict_samps[:,:,pg1i])/2

			# TODO: We need a much better way to deal with this! But for now
			# we're just going to force it.
			hard_fix_params = ['lens_mass_e1','lens_mass_e2']
			for lens_param in hard_fix_params:
				fixi = self.lens_params.index(lens_param)
				lens_samps[lens_samps[:,:,fixi]<-0.55,fixi] =-0.55+1e-5
				lens_samps[lens_samps[:,:,fixi]>0.55,fixi] = 0.55-1e-5

			if sample_save_dir is not None:
				np.save(os.path.join(sample_save_dir,'lens_samps.npy'),
					lens_samps)

		else:
			print('Loading samples from %s'%(sample_save_dir))
			self.infer_class.gen_samples(num_samples,sample_save_dir)
			lens_samps = np.load(os.path.join(sample_save_dir,'lens_samps.npy'))

		# For numba, we need to change the order of the samples for fast
		# evaluation.
		lens_samps = np.ascontiguousarray(np.transpose(lens_samps,axes=(2,0,1)))

		# Pass the lens samples to our eval class.
		self.prob_class.set_samples()

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
			self.cur_state = (np.random.rand(n_walkers, ndim)*0.05 +
				self.target_eval_dict['hyps'])
			# Inflate lower and upper bounds since this can otherwise
			# cause none of the initial samples to be non -np.inf.
			for hpi in range(len(self.target_eval_dict['hyp_names'])):
				name = self.target_eval_dict['hyp_names'][hpi]
				if 'lower' in name:
					self.cur_state[:,hpi] -= 0.2
				if 'upper' in name:
					self.cur_state[:,hpi] += 0.2
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

	def run_sampler(self,n_samps,progress='notebook'):
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
			plt.axhline(self.target_eval_dict['hyps'][ci],c='k')
			plt.show(block=block)

	def plot_corner(self,burnin,hyperparam_plot_names=None,block=True,
		color='#FFAA00',truth_color='#000000'):
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
		"""
		if hyperparam_plot_names is None:
			hyperparam_plot_names = self.target_eval_dict['hyp_names']

		# Get the chains from the samples
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(hyperparam_plot_names))

		# Iterate through groups of hyperparameters and make the plots
		for lens_param in self.lens_params:
			hyp_ind = self.target_eval_dict[lens_param]['hyp_ind']
			hyp_s = np.min(hyp_ind)
			hyp_e = np.max(hyp_ind)+1
			corner.corner(chains[:,hyp_s:hyp_e],
				labels=hyperparam_plot_names[hyp_s:hyp_e],
				bins=20,show_titles=True, plot_datapoints=False,
				label_kwargs=dict(fontsize=10),
				truths=self.target_eval_dict['hyps'][hyp_s:hyp_e],
				levels=[0.68,0.95],color=color,fill_contours=True,
				truth_color=truth_color)
			plt.show(block=block)

	def plot_single_corner(self,burnin,plot_param,hyperparam_plot_names=None,
		block=True,color='#FFAA00',truth_color='#000000',figure=None):
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
		figure = corner.corner(chains[:,hyp_s:hyp_e],
			labels=hyperparam_plot_names[hyp_s:hyp_e],
			dpi=800,bins=20,show_titles=True, plot_datapoints=False,
			label_kwargs=dict(fontsize=13),
			truths=self.target_eval_dict['hyps'][hyp_s:hyp_e],
			levels=[0.68,0.95],color=color,fill_contours=True,
			truth_color=truth_color,fig=figure)
		return figure

	def plot_distributions(self,burnin,param_plot_names=None,block=True,
		color_map=["#a1dab4","#41b6c4","#2c7fb8","#253494"]):
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
		"""
		hyperparam_plot_names = self.target_eval_dict['hyp_names']
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(hyperparam_plot_names))
		global lens_samps

		for li in range(len(self.lens_params)):
			# Grab the lens parameters, the samples, and indices of the
			# hyperparameters
			lens_param = self.lens_params[li]
			samples = lens_samps[li].flatten()
			hyp_ind = self.target_eval_dict[lens_param]['hyp_ind']
			hyp_s = np.min(hyp_ind)
			hyp_e = np.max(hyp_ind)+1
			plt_min = max(np.mean(samples)-6*np.std(samples),
				np.min(samples)-0.1)
			plt_max = min(np.mean(samples)+6*np.std(samples),
				np.max(samples)+0.1)

			# Select the range of parameter values to evaluate the pdf at
			eval_pdf_at = np.linspace(plt_min,plt_max,1000)

			# Plot the samples for the parameter
			plt.figure(dpi=800)
			plt.hist(samples,bins=100,density=True,align='mid',
				color=color_map[0],range=(plt_min,plt_max))

			# Sample 100 chains and plot them
			n_chains_plot = 50
			for chain in chains[np.random.randint(len(chains),size=n_chains_plot)]:
				chain_eval = np.exp(self.target_eval_dict[lens_param]['eval_fn'](
					eval_pdf_at,*chain[hyp_s:hyp_e]))
				plt.plot(eval_pdf_at,chain_eval,color=color_map[1], lw=2,
					alpha=5/n_chains_plot)

			# Plot the true distribution these parameters were being drawn
			# from.
			truth_eval = np.exp(self.target_eval_dict[lens_param]['eval_fn'](
				eval_pdf_at,*self.target_eval_dict['hyps'][hyp_s:hyp_e]))
			plt.plot(eval_pdf_at,truth_eval,color=color_map[2], lw=2.5,
				ls='--')

			# Plot the interim distribution these parameters were being drawn
			# from.
			hyp_ind = self.interim_eval_dict[lens_param]['hyp_ind']
			hyp_s = np.min(hyp_ind)
			hyp_e = np.max(hyp_ind)+1
			truth_eval = np.exp(self.interim_eval_dict[lens_param]['eval_fn'](
				eval_pdf_at,*self.interim_eval_dict['hyps'][hyp_s:hyp_e]))
			plt.plot(eval_pdf_at,truth_eval,color=color_map[3], lw=2.5,
				ls=':')

			# Construct the legend.
			custom_lines = [Line2D([0], [0], color=color_map[0], lw=4),
				Line2D([0], [0], color=color_map[1], lw=4),
				Line2D([0], [0], color=color_map[2], lw=4),
				Line2D([0], [0], color=color_map[3], lw=4)]
			plt.legend(custom_lines, ['BNN Samples', 'Posterior Samples',
				'True Distribution','Interim Distribution'])

			if param_plot_names is None:
				plt.xlabel(lens_param)
				plt.ylabel('p(%s)'%(lens_param))
			else:
				plt.xlabel(param_plot_names[li])
				plt.ylabel('p(%s)'%(param_plot_names[li]))
			plt.xlim([plt_min,plt_max])
			plt.show(block=block)

	def calculate_sample_weights(self,n_p_omega_samps,burnin):
		"""
		Calculate the weights from the posterior on Omega needed for
		reweighting.

		Parameters
		----------
			n_p_omega_samps (int): The number of samples from p(Omega|{d}) to
				use in the reweighting.
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
				self.lens_params)
		lpi = self.prob_class.pt_omegai
		# Calculate the weights using the log posterior samples
		weights = np.mean(np.exp(lpos-lpi),axis=0)
		return weights

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
				show_titles=True,plot_datapoints=False,
				label_kwargs=dict(fontsize=13),
				truths=self.infer_class.y_test[image_index],levels=[0.68,0.95],
				dpi=1600, color=color_map[0],fill_contours=True,
				range=plot_limits,truth_color=truth_color)

		weights = self.calculate_sample_weights(n_p_omega_samps,burnin)
		weights /= np.sum(weights,axis=0)
		weights = weights[:,image_index]

		corner.corner(self.infer_class.predict_samps[:,image_index,:],bins=20,
				labels=self.infer_class.final_params_print_names,show_titles=True,
				plot_datapoints=False,label_kwargs=dict(fontsize=13),
				truths=self.infer_class.y_test[image_index],levels=[0.68,0.95],
				dpi=1600, color=color_map[1],fill_contours=True,
				weights=weights, fig=fig,range=plot_limits,
				truth_color=truth_color)

	def plot_reweighted_calibration(self,burnin,n_perc_points,
		n_p_omega_samps=100,color_map=['#1b9e77','#d95f02','#7570b3'],
		legend=['Perfect Calibration','Bare Network','Reweighted Network']):
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
				plots. Must include 3 colors.
		"""
		# For the first plot we can just use the original BNN code.
		fig = self.infer_class.plot_calibration(color_map=color_map,
			n_perc_points=n_perc_points,show_plot=False,legend=legend)

		weights = self.calculate_sample_weights(n_p_omega_samps,burnin)
		weights /= np.mean(weights,axis=0)

		fig = self.infer_class.plot_calibration(color_map=color_map[1:],
			n_perc_points=n_perc_points,figure=fig,show_plot=False,
			weights=weights,legend=legend)




