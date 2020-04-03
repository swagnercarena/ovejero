# -*- coding: utf-8 -*-
"""
Given a trained model, compare the model posteriors to those of a forward
model.

This module contains the functions neccesary to compare the inference of a
trained BNN to a traditional forward modeling approach powered by lenstronomy.

Examples
--------
The demo Forward_Modeling.ipynb gives examples on how to use this module.

Notes
-----
Unlike most of the other modules in this package, this module assumes that the
images consist only a a lensed source (no lens light or point source).
"""
# TODO: Write tests for these functions after you're done racing.
from ovejero import bnn_inference
from baobab import configs
from baobab.sim_utils import instantiate_PSF_models, get_PSF_model, generate_image
from baobab.data_augmentation import noise_tf
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.data_api import DataAPI
import pandas as pd
import numpy as np
import os, emcee, corner
from matplotlib import pyplot as plt
import lenstronomy.Util.util as util


class ForwardModel(bnn_inference.InferenceClass):
	"""
	A class that inherets from InferenceClass and adds the ability to forward
	# model.
	"""
	def __init__(self,cfg):
		"""
		Initialize the ForwardModel instance using the parameters of the
		configuration file.

		Parameters
		----------
		cfg (dict): The dictionary attained from reading the json config file.
		"""
		# Initialize the BNN inference class.
		super(ForwardModel, self).__init__(cfg)
		# We will use the baobab code to generate our images and then calculate
		# the likelihood manually.
		# First we get the psf model
		self.baobab_cfg = configs.BaobabConfig.from_file(self.baobab_config_path)
		psf_models = instantiate_PSF_models(self.baobab_cfg.psf,
			self.baobab_cfg.instrument.pixel_scale)

		# Now we get the noise function we'll use to add noise to our images
		noise_kwargs = self.baobab_cfg.get_noise_kwargs()
		self.noise_function = noise_tf.NoiseModelTF(**noise_kwargs)

		# Pull the needed information from the config file.
		self.lens_params = self.cfg['dataset_params']['lens_params']

		# Get the model parameter kwargs
		kwargs_model = dict(
			lens_model_list=[
				self.baobab_cfg.bnn_omega.lens_mass.profile,
				self.baobab_cfg.bnn_omega.external_shear.profile],
			source_light_model_list=[self.baobab_cfg.bnn_omega.src_light.profile]
		)

		# initialize all the kwargs and models we'll need to generate the images.
		self.lens_mass_model = LensModel(
			lens_model_list=kwargs_model['lens_model_list'])
		self.src_light_model = LightModel(
			light_model_list=kwargs_model['source_light_model_list'])
		self.lens_eq_solver = LensEquationSolver(self.lens_mass_model)

		self.psf_model = get_PSF_model(psf_models, 1, 0)
		kwargs_detector = util.merge_dicts(self.baobab_cfg.instrument,
			self.baobab_cfg.bandpass, self.baobab_cfg.observation)
		kwargs_detector.update(seeing=self.baobab_cfg.psf.fwhm,
			psf_type=self.baobab_cfg.psf.type,
			kernel_point_source=self.psf_model,
			background_noise=0.0)
		self.data_api = DataAPI(self.baobab_cfg.image.num_pix, **kwargs_detector)

		# Set flags to make sure things are initialzied.
		self.image_selected = False
		self.sampler_init = False

	def select_image(self,image_index):
		"""
		Select the image to conduct forward modeling on.

		Parameters
		----------
			image_index (int): The index of the image to use.
		"""
		# Load the metadata file
		metadata = pd.read_csv(os.path.join(
			self.cfg['validation_params']['root_path'],'metadata.csv'))

		# Get the image filename
		img_filename = ('X_'+'0'*(6-int(np.log10(image_index)))+
			str(image_index)+'.npy')

		# Load the true image.
		self.true_image = np.load(os.path.join(
			self.cfg['validation_params']['root_path'],
			img_filename)).astype(np.float32)

		# Show the image without noise
		print('True image without noise.')
		plt.imshow(self.true_image)
		plt.show()

		# Add noise and show the new image_index
		self.true_image_noise = self.noise_function.add_noise(
			self.true_image).numpy()
		print('True image with noise.')
		plt.imshow(self.true_image_noise)
		plt.show()

		# Find, save, and print the parameters for this image.
		image_data = metadata[metadata['img_filename'] == img_filename]
		self.image_dict = image_data.to_dict(orient='index')[image_index]
		print('Image data')
		print(self.image_dict)

		# We want to save a list of the relevant parameters and the true values
		# (which we will use to initialize our emcee code for speed).
		self.emcee_params_list = []
		self.emcee_initial_values = []
		for samp_key in self.image_dict:
			if ('lens_mass'in samp_key or 'src_light' in samp_key or
				'external_shear' in samp_key) and (
				samp_key != 'external_shear_dec_0' and
				samp_key != 'external_shear_ra_0'):
				self.emcee_params_list.append(samp_key)
				self.emcee_initial_values.append(self.image_dict[samp_key])
		self.emcee_initial_values = np.array(self.emcee_initial_values)
		# It's important to sort this since there is not a defined order to the
		# dict.
		self.emcee_initial_values = self.emcee_initial_values[np.argsort(
			self.emcee_params_list)]
		self.emcee_params_list.sort()

		# Note that image has been selected.
		self.image_selected = True

	def transform_sample_to_dict(self,sample):
		"""
		Take as input a sample array and transform it into the dict used by
		generate_image.

		Parameters
		----------
		sample (np.array): The initial values of the samples. Should be in the
			same order as self.parameter_list (set by select_image).
		"""
		# Just go through the samples one by one and put them in the right
		# place in the dict structure.
		sample_dict = {'lens_mass':dict(),'src_light':dict(),
			'external_shear':dict()}
		for si, param in enumerate(self.emcee_params_list):
			if 'lens_mass' in param:
				sample_dict['lens_mass'][param[10:]] = sample[si]
			if 'src_light' in param:
				sample_dict['src_light'][param[10:]] = sample[si]
			if 'external_shear' in param:
				sample_dict['external_shear'][param[15:]] = sample[si]

		# Add the ra_0 and dec_0 parameters back in.
		sample_dict['external_shear']['ra_0'] = sample_dict['lens_mass'][
			'center_x']
		sample_dict['external_shear']['dec_0'] = sample_dict['lens_mass'][
			'center_y']

		return sample_dict

	def _log_likelihood(self,sample):
		"""
		Calculate the log likehood of a sample.

		Parameters
		----------
		sample (np.array): The initial values of the samples. Should be in the
			same order as self.parameter_list (set by select_image).
		"""
		# Transform the sample into the dict structure expected by generate_image.
		sample_dict = self.transform_sample_to_dict(sample)
		# Generate the image predicted by the parameters.
		pred_img, _ = generate_image(sample_dict, self.psf_model, self.data_api,
			self.lens_mass_model,
			self.src_light_model, self.lens_eq_solver,
			self.baobab_cfg.instrument.pixel_scale,
			self.baobab_cfg.image.num_pix, self.baobab_cfg.components,
			self.baobab_cfg.numerics,min_magnification=0.0,
			lens_light_model=None, ps_model=None)
		pred_img = pred_img.astype(np.float32)

		# Get the noise for each pixel in this image according to our model.
		noise_sigma2 = self.noise_function.get_noise_sigma2(pred_img).numpy()

		# Calculate the log likelihood using this noise.
		log_emcee_like = (pred_img - self.true_image_noise)**2 / noise_sigma2
		log_emcee_like = -np.sum(log_emcee_like)/2

		return log_emcee_like

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
		if self.image_selected is False:
			raise RuntimeError('Select an image before starting your sampler')

		# Set a few parameters associated with the emcee code.
		self.n_walkers = n_walkers
		ndim = len(self.emcee_initial_values)

		# Load chains we've used before.
		if os.path.isfile(save_path):
			print('Loaded chains found at %s'%(save_path))
			self.cur_state = None

		# Otherwise create new chains.
		else:
			print('No chains found at %s'%(save_path))
			self.cur_state = (np.random.rand(n_walkers, ndim)*0.05 +
				self.emcee_initial_values)

		# Initialize backend hdf5 file that will store samples as we go
		self.backend = emcee.backends.HDFBackend(save_path)

		# Very important I pass in prob_class.log_post_omega here to allow
		# pickling.
		self.sampler = emcee.EnsembleSampler(n_walkers, ndim,
			self._log_likelihood, backend=self.backend,pool=pool)

		print('Optimizing the following parameters:')
		print(self.emcee_params_list)

		self.sampler_init = True

	def run_sampler(self,n_samps,progress='notebook'):
		"""
		Run an emcee sampler to get a posterior on the hyperparameters.

		Parameters
		----------
			n_samps (int): The number of samples to take
		"""
		if self.sampler_init is False:
			raise RuntimeError('Must initialize sampler before running sampler')

		self.sampler.run_mcmc(self.cur_state,n_samps,progress=progress)

		self.cur_state = None

	def plot_chains(self,burnin=None,block=True):
		"""
		Plot the chains resulting from the emcee to figure out what
		the correct burnin is.

		Parameters
		----------
			burnin (int): How many of the initial samples to drop as burnin
			block (bool): If true, block excecution after plt.show() command
		"""
		# Extract and plot the chains
		chains = self.sampler.get_chain()
		if burnin is not None:
			chains = chains[burnin:]
		for ci, chain in enumerate(chains.T):
			plt.plot(chain.T,'.')
			plt.title(self.emcee_params_list[ci])
			plt.ylabel(self.emcee_params_list[ci])
			plt.xlabel('sample')
			plt.axhline(self.emcee_initial_values[ci],c='k')
			plt.show(block=block)

	def _correct_chains(self,chains,param_names):
		"""
		Correct the chains and true values so that their convention agrees with
		what was used to train the BNN.

		Parameters
		----------
			chains (np.array): A numpy array containing the chain in the
				original parameter space.
			param_names ([str,...]): A list of string containing the names
				of each of the parameters in chains.

		Returns
		-------
			(np.array,[str,...]): A tuple containing the corrected chains and
				parameter list.

		TODO: Integrate this directly with the dataset code.
		"""
		# Go through the parameters and find which ones need to be corrected
		param_names = np.array(param_names)
		new_param_names = np.copy(param_names)
		dat_params=self.cfg['dataset_params']
		self.emcee_initial_values

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
			gamma = self.true_values[param_names==rat_param]
			ang = self.true_values[param_names==ang_param]
			# Calculate g1 and g2.
			g1 = gamma*np.cos(2*ang)
			g2 = gamma*np.sin(2*ang)
			self.true_values[param_names==rat_param] = g1
			self.true_values[param_names==ang_param] = g2

		# Now get all of the parameters that were changed to log space.
		for log_param in dat_params['lens_params_log']:
			# Pull the parameter value
			value = chains[:,param_names==log_param]
			# Change the name and value
			new_param_names[param_names==log_param] = log_param+'_log'
			chains[:,param_names==log_param] = np.log(value)
			# Make the same change in the true values.
			self.true_values[param_names==log_param] = np.log(self.true_values[
				param_names==log_param])

		return chains,new_param_names

	def plot_posterior_contours(self,burnin,num_samples,block=True,
		sample_save_dir=None,color_map=['#FFAA00','#41b6c4'],
		plot_limits=None):
		"""
		Plot the corner plot of chains resulting from the emcee for the
		lens mass parameters.

		Parameters
		----------
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
		"""
		# Get the chains from the samples
		chains = self.sampler.get_chain()[burnin:].reshape(-1,
			len(self.emcee_initial_values))

		pi_keep = []
		param_names = []
		for pi, param in enumerate(self.emcee_params_list):
			if param in self.lens_params:
				pi_keep.append(pi)
				param_names.append(param)

		# Keep only the chains related to the parameters we want to look at.
		chains = chains.T[pi_keep].T
		self.true_values = self.emcee_initial_values[pi_keep]
		chains,new_param_names = self._correct_chains(chains,param_names)

		for pi, param in enumerate(new_param_names):
			fpi = self.final_params.index(param)
			new_param_names[pi] = self.final_params_print_names[fpi]

		# # Iterate through groups of hyperparameters and make the plots
		fig = corner.corner(chains,
			labels=new_param_names,
			bins=20,show_titles=True, plot_datapoints=False,
			label_kwargs=dict(fontsize=10),truths=self.true_values,
			levels=[0.68,0.95],color=color_map[0],fill_contours=True,
			range=plot_limits)

		# Now overlay the samples from the BNN
		self.gen_samples(num_samples,sample_save_dir=sample_save_dir,
			single_image=self.true_image/np.std(self.true_image))
		corner.corner(self.predict_samps[:,0,:],bins=20,
				labels=self.final_params_print_names,show_titles=True,
				plot_datapoints=False,label_kwargs=dict(fontsize=13),
				truths=self.true_values,levels=[0.68,0.95],
				dpi=1600, color=color_map[1],fig=fig,fill_contours=True,
				range=plot_limits)
		plt.show(block=block)

