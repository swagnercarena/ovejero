# -*- coding: utf-8 -*-
"""
Given a trained model, compare the model posteriors to those of a forward
model.

This module contains the functions neccesary to compare the inference of a
trained BNN to a traditional forward modeling approach powered by lenstronomy.

Examples:
	The demo Forward_Modeling_Demo.ipynb gives examples on how to use this
	module.

Notes:
	Unlike most of the other modules in this package, this module assumes that
	the images consist only a a lensed source (no lens light or point source).
"""
# TODO: Write tests for these functions after you're done racing.
from ovejero import bnn_inference
from baobab import configs
from baobab.sim_utils import instantiate_PSF_kwargs
from baobab.data_augmentation import noise_tf, noise_lenstronomy
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from matplotlib import cm
import pandas as pd
import numpy as np
import os, corner
import tensorflow as tf
from matplotlib import pyplot as plt
import lenstronomy.Util.util as util
from matplotlib.lines import Line2D


class ForwardModel(bnn_inference.InferenceClass):
	"""
	A class that inherets from InferenceClass and adds the ability to forward
	# model.
	"""
	def __init__(self,cfg,lite_class=False,test_set_path=None):
		"""
		Initialize the ForwardModel instance using the parameters of the
		configuration file.

		Parameters:
			cfg (dict): The dictionary attained from reading the json config file.
			lite_class (bool): If True, do not bother loading the BNN model
				weights. This allows the user to save on memory, but will cause
				an error if the BNN samples have not already been drawn.
			test_set_path (str): The path to the set of images that the
				forward modeling image will be pulled from. If None, the
				path to the validation set images will be used.
		"""
		# Initialize the BNN inference class.
		super(ForwardModel, self).__init__(cfg,lite_class,test_set_path)

		# We will use the baobab code to generate our images and then calculate
		# the likelihood manually.
		# First we get the psf model
		self.baobab_cfg = configs.BaobabConfig.from_file(
			self.baobab_config_path)

		# Add the lens and source models specified in the config. Currently
		# no light model can be specified. Note that any self variable
		# starting with the prefix ls_ is for use with lenstronomy.
		self.ls_lens_model_list = []
		fixed_lens = []
		kwargs_lens_init = []
		kwargs_lens_sigma = []
		kwargs_lower_lens = []
		kwargs_upper_lens = []

		self.ls_source_model_list = []
		fixed_source = []
		kwargs_source_init = []
		kwargs_source_sigma = []
		kwargs_lower_source = []
		kwargs_upper_source = []

		# For now, each of the distribution options are hard coded toghether
		# with reasonable choices for their parameters.

		if 'PEMD' in cfg['forward_mod_params']['lens_model_list']:
			self.ls_lens_model_list.append('PEMD')
			fixed_lens.append({})
			kwargs_lens_init.append({'theta_E': 0.7, 'e1': 0., 'e2': 0.,
				'center_x': 0., 'center_y': 0., 'gamma': 2.0})
			kwargs_lens_sigma.append({'theta_E': .2, 'e1': 0.05, 'e2': 0.05,
				'center_x': 0.05, 'center_y': 0.05, 'gamma': 0.2})
			kwargs_lower_lens.append({'theta_E': 0.01, 'e1': -0.5, 'e2': -0.5,
				'center_x': -10, 'center_y': -10, 'gamma': 0.01})
			kwargs_upper_lens.append({'theta_E': 10., 'e1': 0.5, 'e2': 0.5,
				'center_x': 10, 'center_y': 10, 'gamma': 10})

		if 'SHEAR_GAMMA_PSI' in cfg['forward_mod_params']['lens_model_list']:
			self.ls_lens_model_list.append('SHEAR_GAMMA_PSI')
			fixed_lens.append({'ra_0': 0, 'dec_0': 0})
			kwargs_lens_init.append({'gamma_ext': 0.2, 'psi_ext': 0.0})
			kwargs_lens_sigma.append({'gamma_ext': 0.1, 'psi_ext': 0.1})
			kwargs_lower_lens.append({'gamma_ext': 0, 'psi_ext': -0.5*np.pi})
			kwargs_upper_lens.append({'gamma_ext': 10, 'psi_ext': 0.5*np.pi})

		if 'SERSIC_ELLIPSE' in cfg['forward_mod_params']['source_model_list']:
			self.ls_source_model_list.append('SERSIC_ELLIPSE')
			fixed_source.append({})
			kwargs_source_init.append({'R_sersic': 0.2, 'n_sersic': 1,
				'e1': 0, 'e2': 0, 'center_x': 0., 'center_y': 0})
			kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.1,
				'e1': 0.05, 'e2': 0.05, 'center_x': 0.2, 'center_y': 0.2})
			kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5,
				'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10,
				'center_y': -10})
			kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10,
				'n_sersic': 5., 'center_x': 10, 'center_y': 10})

		# Feed all of the above params into lists
		self.ls_lens_params = [kwargs_lens_init, kwargs_lens_sigma,
			fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
		self.ls_source_params = [kwargs_source_init, kwargs_source_sigma,
			fixed_source, kwargs_lower_source, kwargs_upper_source]
		self.ls_kwargs_params = {'lens_model': self.ls_lens_params,
				'source_model': self.ls_source_params}

		# Some of the likelihood parameters being used by Lenstronomy
		self.ls_kwargs_likelihood = {'source_marg': False}
		self.ls_kwargs_model = {'lens_model_list': self.ls_lens_model_list,
			'source_light_model_list': self.ls_source_model_list}

		# We will also need some of the noise kwargs. We will feed the
		# lenstronomy version straight to lenstronomy and the tensorflow
		# version to our pipeline for selecting the image.
		bandpass = self.baobab_cfg.survey_info.bandpass_list[0]
		detector = self.baobab_cfg.survey_object_dict[bandpass]
		detector_kwargs = detector.kwargs_single_band()
		self.noise_kwargs = self.baobab_cfg.get_noise_kwargs(bandpass)
		self.noise_function = noise_tf.NoiseModelTF(**self.noise_kwargs)

		self.ls_kwargs_psf = instantiate_PSF_kwargs(
			self.baobab_cfg.psf['type'],detector_kwargs['pixel_scale'],
			seeing=detector_kwargs['seeing'],
			kernel_size=detector.psf_kernel_size,
			which_psf_maps=self.baobab_cfg.psf['which_psf_maps'])

		# The kwargs for the numerics. These should match what was used
		# to generate the image.
		self.ls_kwargs_numerics = {
			'supersampling_factor': (
				self.baobab_cfg.numerics.supersampling_factor),
			'supersampling_convolution': False}

		# Pull the needed information from the config file.
		self.lens_params = self.cfg['dataset_params']['lens_params']

		# Get the model parameter kwargs
		self.ls_kwargs_model = {'lens_model_list': self.ls_lens_model_list,
			'source_light_model_list': self.ls_source_model_list}

		# Set flags to make sure things are initialzied.
		self.image_selected = False
		self.sampler_init = False

	def select_image(self,image_index,block=True):
		"""
		Select the image to conduct forward modeling on.

		Parameters:
			image_index (int): The index of the image to use.
		"""
		# Load the metadata file
		metadata = pd.read_csv(os.path.join(
			self.cfg['validation_params']['root_path'],'metadata.csv'))

		# Get the image filename
		img_filename = 'X_{0:07d}.npy'.format(image_index)

		# Load the true image.
		self.true_image = np.load(os.path.join(
			self.cfg['validation_params']['root_path'],
			img_filename)).astype(np.float32)

		# Show the image without noise
		print('True image without noise.')
		plt.imshow(self.true_image,cmap=cm.magma)
		plt.colorbar()
		plt.show(block=block)

		# Set the random seed since we will be using it to add
		# noise
		tf.random.set_seed(self.cfg['training_params']['random_seed'])
		# Add noise and show the new image_index
		self.true_image_noise = self.noise_function.add_noise(
			self.true_image).numpy()
		print('True image with noise.')
		plt.imshow(self.true_image_noise,cmap=cm.magma)
		plt.colorbar()
		plt.show(block=block)

		# Extract the data kwargs (including noise kwargs) being used
		# by lenstronomy.
		_, _, ra_0, dec_0, _, _, Mpix2coord, _ = (
			util.make_grid_with_coordtransform(
				numPix=self.baobab_cfg.image['num_pix'],
				deltapix=self.baobab_cfg.instrument['pixel_scale'],
				center_ra=0, center_dec=0, subgrid_res=1,
				inverse=self.baobab_cfg.image['inverse']))

		# Update the lenstronomy kwargs with the image information
		noise_dict = noise_lenstronomy.get_noise_sigma2_lenstronomy(
			self.true_image_noise,**self.noise_kwargs)
		self.ls_kwargs_data = {
			'background_rms': np.sqrt(noise_dict['sky']+noise_dict['readout']),
			'exposure_time': (
				self.noise_kwargs['exposure_time']*
				self.noise_kwargs['num_exposures']),
			'ra_at_xy_0': ra_0,
			'dec_at_xy_0': dec_0,
			'transform_pix2angle': Mpix2coord,
			'image_data': self.true_image_noise
		}
		self.ls_multi_band_list = [[self.ls_kwargs_data,
			self.ls_kwargs_psf, self.ls_kwargs_numerics]]

		# Find, save, and print the parameters for this image.
		image_data = metadata[metadata['img_filename'] == img_filename]
		self.true_values = image_data.to_dict(orient='index')[image_index]
		print('Image data')
		print(self.true_values)

		# Note that image has been selected.
		self.image_selected = True

	def initialize_sampler(self,walker_ratio,chains_save_path):
		"""
		Initialize the sampler to be used by run_samples.


		Parameters:
			walker_ratio (int): The number of walkers per free parameter.
				Must be at least 2.
			save_path (str): An h5 path specifying where to save the
				sampler chains. If a sampler chain is already present in the
				path it will be loaded.
		"""
		if self.image_selected is False:
			raise RuntimeError('Select an image before starting your sampler')

		# Set up the fitting sequence and fitting kwargs from lenstronomy
		self.walker_ratio = walker_ratio
		self.chains_save_path = chains_save_path
		ls_kwargs_data_joint = {'multi_band_list': self.ls_multi_band_list,
			'multi_band_type': 'multi-linear'}
		ls_kwargs_constraints = {}
		self.fitting_seq = FittingSequence(ls_kwargs_data_joint,
			self.ls_kwargs_model, ls_kwargs_constraints,
			self.ls_kwargs_likelihood, self.ls_kwargs_params)

		self.sampler_init = True

	def run_sampler(self,n_samps):
		"""
		Run an emcee sampler to get a posterior on the hyperparameters.

		Parameters:
			n_samps (int): The number of samples to take
		"""
		if self.sampler_init is False:
			raise RuntimeError('Must initialize sampler before running sampler')

		# Notify user if chains were found
		if os.path.isfile(self.chains_save_path):
			print('Using chains found at %s'%(self.chains_save_path))
			self.start_from_backup = True
		else:
			print('No chains found at %s'%(self.chains_save_path))
			self.start_from_backup = False

		# Initialize the fitting kwargs to be passed to the lenstronomy
		# fitting sequence. We set burnin to 0 since we would like to be
		# responsible for the burnin.
		fitting_kwargs_list = [['MCMC',{'n_burn': 0,
			'n_run': n_samps, 'walkerRatio': self.walker_ratio,
			'sigma_scale': 0.1, 'backup_filename': self.chains_save_path,
			'start_from_backup': self.start_from_backup}]]
		chain_list = self.fitting_seq.fit_sequence(fitting_kwargs_list)

		# Extract the relevant outputs:
		self.chain_params = chain_list[0][2]
		# I want the walkers to be seperate so I can chose my own burnin
		# adventure here.
		self.chains = chain_list[0][1].reshape((-1,
			len(self.chain_params)*self.walker_ratio,len(self.chain_params)))

		# Convert chain_params naming convention to the one used by baobab
		renamed_params = []
		for param in self.chain_params:
			if 'lens0' in param:
				renamed_params.append('lens_mass_'+param[:-6])
			if 'lens1' in param:
				renamed_params.append('external_shear_'+param[:-6])
			if 'source_light0' in param:
				renamed_params.append('src_light_'+param[:-14])
		self.chain_params=renamed_params

	def plot_chains(self,burnin=None,block=True):
		"""
		Plot the chains resulting from the emcee to figure out what
		the correct burnin is.

		Parameters:
			burnin (int): How many of the initial samples to drop as burnin
			block (bool): If true, block excecution after plt.show() command
		"""
		# Extract and plot the chains
		if burnin is not None:
			chains = self.chains[burnin:]
		else:
			chains = self.chains
		for ci, chain in enumerate(chains.T):
			plt.plot(chain.T,'.')
			plt.title(self.chain_params[ci])
			plt.ylabel(self.chain_params[ci])
			plt.xlabel('sample')
			plt.axhline(self.true_values[self.chain_params[ci]],c='k')
			plt.show(block=block)

	def _correct_chains(self,chains,param_names,true_values):
		"""
		Correct the chains and true values so that their convention agrees with
		what was used to train the BNN.

		Parameters:
			chains (np.array): A numpy array containing the chain in the
				original parameter space. Dimensions should be (n_samples,
				n_params).
			param_names ([str,...]): A list of string containing the names
				of each of the parameters in chains.
			true_values (np.array): A numpy array with the true values for
				each parameter in the untransformed parameterization. Should
				have dimensions (n_params).

		Returns:
			[str,...]: A list containing the corrected parameter names.
			Everything else is changed in place.

		TODO: Integrate this directly with the dataset code.
		"""
		# Go through the parameters and find which ones need to be corrected
		param_names = np.array(param_names)
		new_param_names = np.copy(param_names)
		dat_params=self.cfg['dataset_params']

		# First get all of the parameters that were changed to a cartesian
		# format.
		for rat_param,ang_param,param_prefix in zip(dat_params['gampsi'][
			'gampsi_params_rat'],dat_params['gampsi'][
			'gampsi_params_ang'],dat_params['gampsi'][
			'gampsi_parameter_prefixes']):
			# Pull the gamma and angle parameter.
			gamma = chains[:,param_names==rat_param]
			ang = chains[:,param_names==ang_param]
			# Calculate g1 and g2.
			g1 = gamma*np.cos(2*ang)
			g2 = gamma*np.sin(2*ang)
			# Change the name and the values
			new_param_names[param_names==rat_param] = param_prefix+'_g1'
			new_param_names[param_names==ang_param] = param_prefix+'_g2'
			chains[:,param_names==rat_param] = g1
			chains[:,param_names==ang_param] = g2
			# Make the same change in the true values
			gamma = true_values[param_names==rat_param]
			ang = true_values[param_names==ang_param]
			# Calculate g1 and g2.
			g1 = gamma*np.cos(2*ang)
			g2 = gamma*np.sin(2*ang)
			true_values[param_names==rat_param] = g1
			true_values[param_names==ang_param] = g2

		# Now get all of the parameters that were changed to log space.
		for log_param in dat_params['lens_params_log']:
			# Pull the parameter value
			value = chains[:,param_names==log_param]
			# Change the name and value
			new_param_names[param_names==log_param] = log_param+'_log'
			chains[:,param_names==log_param] = np.log(value)
			# Make the same change in the true values.
			true_values[param_names==log_param] = np.log(true_values[
				param_names==log_param])

		return new_param_names

	def plot_posterior_contours(self,burnin,num_samples,block=True,
		sample_save_dir=None,color_map=['#FFAA00','#41b6c4'],
		plot_limits=None,truth_color='#000000',save_fig_path=None,
		dpi=400,fig=None,show_plot=True,plot_fow_model=True,
		add_legend=True,fontsize=12):
		"""
		Plot the corner plot of chains resulting from the emcee for the
		lens mass parameters.

		Parameters:
			burnin (int): How many of the initial samples to drop as burnin
			num_samples (int): The number of bnn samples to use for the
				contour
			block (bool): If true, block excecution after plt.show() command
			sample_save_dir (str): A path to a folder to save/load the samples.
				If None samples will not be saved. Do not include .npy, this will
				be appended (since several files will be generated).
			color_map ([str,...]): A list of strings specifying the colors to
				use in the contour plots.
			plot_limits ([(float,float),..]): A list of float tuples that define
				the maximum and minimum plot range for each posterior parameter.
			truth_color (str): The color to use for plotting the truths in the
				corner plot.
			save_fig_path (str): If specified, the figure will be saved to that
				path.
			dpi (int): The dpi to use when generating the image.
			fig (matplotlib.Figure): The figure to use as a starting point. Best
				to leave this as None unless you're passing in another corner
				plot.
			show_plot (bool): Whether or not to show the plot or just return
				the figure.
			plot_fow_model (bool): Whether or not to plot the forward modeling
				posteriors. This is mostly here  for plotting multiple BNN
				posteriors on one plot.
			add_legend (bool): Whether or not to add an auto-generated legend.
			fontsize (int): The fontsize for the corner plot labels.

		Returns:
			(matplotlib.pyplot.figure): The figure object containing the
			contours.
		"""
		# Get the chains from the samples
		chains = self.chains[burnin:].reshape(-1,len(self.chain_params))

		# Keep only the parameters that our BNN is predicting
		pi_keep = []
		chain_params_keep = []
		for pi, param in enumerate(self.chain_params):
			if param in self.lens_params:
				pi_keep.append(pi)
				chain_params_keep.append(param)

		# Keep only the chains related to the parameters we want to look at.
		chains = chains.T[pi_keep].T
		true_values_list = []
		for param in chain_params_keep:
			true_values_list.append(self.true_values[param])
		true_values_list = np.array(true_values_list)

		chain_params_keep = self._correct_chains(chains,chain_params_keep,
			true_values_list)

		# The final step is a simple reordering
		reordered_chains = np.zeros_like(chains)
		reordered_true_values = np.zeros_like(true_values_list)
		for pi, param in enumerate(chain_params_keep):
			fpi = self.final_params.index(param)
			reordered_chains[:,fpi] = chains[:,pi]
			reordered_true_values[fpi] = true_values_list[pi]

		# Make a corner plot for the BNN samples
		hist_kwargs = {'density':True,'color':color_map[0]}
		self.gen_samples(num_samples,sample_save_dir=sample_save_dir,
			single_image=self.true_image_noise/np.std(self.true_image_noise))
		corner_bnn_samples = self.predict_samps.reshape(-1,
			self.predict_samps.shape[-1])
		fig = corner.corner(corner_bnn_samples,
				bins=20,labels=self.final_params_print_names,show_titles=False,
				plot_datapoints=False,label_kwargs=dict(fontsize=fontsize),
				truths=reordered_true_values,levels=[0.68,0.95],
				dpi=dpi, color=color_map[0],fig=fig,fill_contours=True,
				range=plot_limits,truth_color=truth_color,
				hist_kwargs=hist_kwargs)

		# Now overlay the forward modeling samples
		if plot_fow_model:
			hist_kwargs['color'] = color_map[1]
			fig = corner.corner(reordered_chains,
				labels=self.final_params_print_names,bins=20,show_titles=False,
				plot_datapoints=False,label_kwargs=dict(fontsize=fontsize),
				truths=reordered_true_values,levels=[0.68,0.95],dpi=dpi,
				color=color_map[1],fill_contours=True,range=plot_limits,
				truth_color=truth_color,hist_kwargs=hist_kwargs,fig=fig)

			left, bottom, width, height = [0.5725,0.8, 0.15, 0.18]
			ax2 = fig.add_axes([left, bottom, width, height])
			ax2.imshow(self.true_image_noise,cmap=cm.magma,origin='lower')

		# Add a nice legend to our contours
		handles = [Line2D([0], [0], color=color_map[0], lw=10),
			Line2D([0], [0], color=color_map[1], lw=10)]
		bnn_type = self.cfg['training_params']['bnn_type']
		if bnn_type == 'gmm':
			bnn_type = 'GMM'
		else:
			bnn_type = bnn_type.capitalize()

		if add_legend:
			fig.legend(handles,[bnn_type+' BNN','Forward Modeling'],
				loc=(0.55,0.75),fontsize=20)

		if save_fig_path is not None:
			plt.savefig(save_fig_path)

		if show_plot:
			plt.show(block=block)

		return fig
