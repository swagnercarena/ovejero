# -*- coding: utf-8 -*-
"""
Given a trained model, use the model to infer the posteriors of new lenses.

This module contains the functions neccesary for conducting inference using a
trained bnn.

Examples
--------
The demo Test_Model_Performance.ipynb gives a number of examples on how to use
this module.
"""
from ovejero import model_trainer, data_tools, bnn_alexnet
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import corner
import tensorflow as tf
import tensorflow_probability as tfp
import os

class InferenceClass:
	"""
	A class that contains all of the functions needed to use the bnn_alexnet
	models for inference. This class will output correctly marginalized 
	predictions as well as make important performance plots.
	"""
	def __init__(self,cfg):
		"""
		Initialize the InferenceClass instance using the parameters of the
		configuration file.

		Parameters
		----------
		cfg (dict): The dictionary attained from reading the json config file.
		"""

		self.cfg = cfg
		self.model, self.loss = model_trainer.model_loss_builder(cfg,
			verbose=True)

		# Load the validation set we're going to use.
		self.tf_record_path_v = os.path.join(
			cfg['validation_params']['root_path'],
			cfg['validation_params']['tf_record_path'])
		# Load the parameters and the batch size needed for computation
		self.final_params = cfg['training_params']['final_params']
		self.final_params_print_names = cfg['inference_params'][
			'final_params_print_names']
		self.num_params = len(self.final_params)
		self.batch_size = cfg['training_params']['batch_size']
		self.norm_images = cfg['training_params']['norm_images']
		self.baobab_config_path = cfg['training_params']['baobab_config_path']

		self.tf_dataset_v = data_tools.build_tf_dataset(self.tf_record_path_v,
			self.final_params,self.batch_size,1,self.baobab_config_path,
			norm_images=self.norm_images)

		self.output_shape = self.model.output_shape[1]
		self.bnn_type = cfg['training_params']['bnn_type']

		# This code is borrowed from the LensingLossFunctions initializer
		self.flip_pairs = cfg['training_params']['flip_pairs']
		# Always include no flips as an option.
		self.flip_mat_list = [np.diag(np.ones(self.num_params))]
		for flip_pair in self.flip_pairs:
			# Initialize a numpy array since this is the easiest way
			# to flexibly set the tensor.
			const_initializer = np.ones(self.num_params)
			const_initializer[flip_pair] = -1
			self.flip_mat_list.append(np.diag(const_initializer))

		self.loss_class = bnn_alexnet.LensingLossFunctions(
			self.flip_pairs,self.num_params)

		self.y_pred = None
		self.y_cov = None
		self.y_std = None
		self.y_test = None
		self.predict_samps = None
		self.samples_init = False

	def fix_flip_pairs(self,predict_samps,y_test):
		"""
		Update predict_samps to account for pairs of points (i.e. 
		ellipticities) that give equivalent physical lenses if both values
		are flipped.

		Parameters
		----------
			predict_samps (np.array): A numpy array with dimensions (
				num_samples,batch_size,num_params) that contains multiple
				samples of the model output.
			y_test (np.array): A numpy array with dimensions (batch_size,
				num_params) that contains the true values of each example
				in the batch.

		Notes
		-----
			The flip_pairs will be read from the config file. predict_samps is
			modified in place.
		"""
		# Initialize a matrix that will store every possible flip for each
		# sample.
		predict_flips = np.zeros((len(self.flip_mat_list),)+
			predict_samps[0].shape)
		for samp_i in range(len(predict_samps)):
			predict_samp = predict_samps[samp_i]
			for flip_mi in range(len(self.flip_mat_list)):
				flip_mat = self.flip_mat_list[flip_mi]
				predict_flips[flip_mi]=np.dot(predict_samp,flip_mat)
			min_flip_indices = np.argmin(np.sum(np.abs(predict_flips-y_test),
				axis=-1),axis=0)
			predict_samps[samp_i] = predict_flips[min_flip_indices,
				np.arange(self.batch_size)]

	def undo_param_norm(self,predict_samps,y_test,al_samp):
		"""
		Undo the normalization of the parameters done for training.

		Parameters
		----------
			predict_samps (np.array): A numpy array with dimensions (
				num_samples,batch_size,num_params) that contains multiple
				samples of the model output.
			y_test (np.array): A numpy array with dimensions (batch_size,
				num_params) that contains the true values of each example
				in the batch.
			al_samp (np.array): A numpy array with dimensions (num_samples,
				batch_size,num_params,num_params) that contains the aleatoric 
				uncertainty of each sample in the batch.

		Notes
		-----
			Correction to numpy arrays is done in place
		"""

		# Get the normalization constants used from the csv file.
		normalization_constants_path = os.path.join(
			self.cfg['training_params']['root_path'],
			self.cfg['dataset_params']['normalization_constants_path'])
		norm_const_dict = pd.read_csv(normalization_constants_path, 
			index_col=None)

		# Go through each parameter and undo the normalization.
		for lpi in range(len(self.final_params)):
			lens_param = self.final_params[lpi]
			param_mean = norm_const_dict[lens_param][0]
			param_std = norm_const_dict[lens_param][1]

			# Correct the samples
			predict_samps[:,:,lpi] *= param_std
			predict_samps[:,:,lpi] += param_mean

			# Correct the true values
			y_test[:,lpi] *= param_std
			y_test[:,lpi] += param_mean

			# Correct the row and columns of the al_samp covariance matrices
			al_samp[:,:,lpi,:] *= param_std
			al_samp[:,:,:,lpi] *= param_std

	def gen_samples(self,num_samples,sample_save_dir=None):
		"""
		Generate the y prediction and the associated covariance matrix
		by marginalizing over both network and output uncertainty.

		Parameters
		----------
			num_samples (int): The number of samples used to marginalize over
				the network's uncertainty.
			sample_save_dir (str): A path to a folder to save/load the samples. 
				If None samples will not be saved. Do not include .npy, this will
				be appended (since several files will be generated).
		"""

		if sample_save_dir is None or not os.path.isdir(sample_save_dir):
			if sample_save_dir is not None:
				print('No samples found. Saving samples to %s'%(sample_save_dir))
			# Extract image and output batch from tf dataset
			for image_batch, yt_batch in self.tf_dataset_v.take(1):
				self.images = image_batch
				self.y_test = yt_batch.numpy()

			# This is where we will save the samples for each prediction. We will
			# use this to numerically extract the covariance.
			predict_samps = np.zeros((num_samples,self.batch_size,
				self.num_params))
			# We also want to store a sampling of the aleatoric noise being
			# predicted to get a handle on how it compares to the epistemic
			# uncertainty.
			al_samp = np.zeros((num_samples,self.batch_size,self.num_params,
				self.num_params))
			# Generate our samples
			for samp in tqdm(range(num_samples)):
				output = self.model.predict(self.images) 
				# How we extract uncertanties will depend on the type of network in
				# question.
				if self.bnn_type == 'diag':
					# In the diagonal case we only need to add Gaussian random 
					# noise scaled by the variane.
					y_pred, std_pred = tf.split(output,num_or_size_splits=2,
						axis=-1)
					std_pred = tf.exp(std_pred)
					# Draw a random sample of noise and add that noise to our 
					# predicted value.
					noise = tf.random.normal((self.batch_size,
						self.num_params))*std_pred
					p_samps_tf = y_pred+noise
					a_samps_tf = tf.linalg.diag(std_pred)

					# Extract the numpy from the tensorflow
					predict_samps[samp] = p_samps_tf.numpy()
					al_samp[samp] = a_samps_tf.numpy()

				elif self.bnn_type == 'full':
					# In the full covariance case we need to explicitly 
					# construct our covariance matrix. This mostly follow the 
					# code in bnn_alexnet full_covariance_loss function.
					# Divide our output into the prediction and the precision
					# matrix elements.
					L_elements_len = int(self.num_params*(self.num_params+1)/2)
					y_pred, L_mat_elements = tf.split(output,
						num_or_size_splits=[self.num_params,L_elements_len],
						axis=-1)
					# Transform our matrix elements into a precision matrix and
					# the precision matrix into a covariance matrix.
					prec_mats, _ = self.loss_class.construct_precision_matrix(
						L_mat_elements)
					cov_mats = tf.linalg.inv(prec_mats)
					# Sample noise using our covariance matrices
					mvn = tfp.distributions.MultivariateNormalFullCovariance(
						loc=y_pred,covariance_matrix=cov_mats)
					p_samps_tf = mvn.sample(1)

					# Extract the numpy from the tensorflow
					predict_samps[samp] = p_samps_tf.numpy()
					al_samp[samp] = cov_mats.numpy()

				elif self.bnn_type == 'gmm':
					# In the gmm case, we do the same thing as the bnn case, but
					# draw which of the two posteriors to draw from.
					L_elements_len = int(self.num_params*(self.num_params+1)/2)
					y_pred1,L_mat_elements1,y_pred2,L_mat_elements2,pi_logit = (
						tf.split(output,num_or_size_splits = [self.num_params,
							L_elements_len,self.num_params,L_elements_len,1],
							axis=-1))

					# Set the probability between 0 and 1.
					pi = tf.sigmoid(pi_logit)

					# Now build the precision matrix for our two models and extract the
					# diagonal components used for the loss calculation
					prec_mat1, _ = self.loss_class.construct_precision_matrix(
						L_mat_elements1)
					# We have to flatten the covariance matrices for the condition
					# step later on.
					cov_mats1 = tf.reshape(tf.linalg.inv(prec_mat1),(
						self.batch_size,-1))
					prec_mat2, _ = self.loss_class.construct_precision_matrix(
						L_mat_elements2)
					cov_mats2 = tf.reshape(tf.linalg.inv(prec_mat2),(
						self.batch_size,-1))

					# Use random draws from a uniform distribution to select between the
					# two outputs
					switch = tf.random.uniform((self.batch_size,1))
					y_pred = tf.where(switch<pi,y_pred1,y_pred2)
					cov_mats = tf.reshape(tf.where(switch<pi,cov_mats1,cov_mats2),
						(self.batch_size,self.num_params,self.num_params))

					# Draw from the alleatoric posterior.
					mvn = tfp.distributions.MultivariateNormalFullCovariance(
						loc=y_pred,covariance_matrix=cov_mats)
					p_samps_tf = mvn.sample(1)

					# Extract the numpy from the tensorflow
					predict_samps[samp] = p_samps_tf.numpy()
					al_samp[samp] = cov_mats.numpy()

				else:
					raise NotImplementedError('gen_pred_cov does not yet support'+
						' %s models'%(self.bnn_type))

			self.fix_flip_pairs(predict_samps,self.y_test)
			self.undo_param_norm(predict_samps,self.y_test,al_samp)

			self.predict_samps = predict_samps
			self.al_samp = al_samp

			# Save the samples if desired.
			if sample_save_dir is not None:
				os.mkdir(sample_save_dir)
				np.save(os.path.join(sample_save_dir,'pred.npy'),
					self.predict_samps)
				np.save(os.path.join(sample_save_dir,'al_samp.npy'),
					self.al_samp)
				np.save(os.path.join(sample_save_dir,'images.npy'),
					self.images)
				np.save(os.path.join(sample_save_dir,'y_test.npy'),
					self.y_test)

		else:
			print('Loading samples from %s'%(sample_save_dir))
			self.predict_samps = np.load(os.path.join(sample_save_dir,
				'pred.npy'))
			self.al_samp = np.load(os.path.join(sample_save_dir,'al_samp.npy'))
			self.images = np.load(os.path.join(sample_save_dir,'images.npy'))
			self.y_test = np.load(os.path.join(sample_save_dir,'y_test.npy'))

		self.al_cov = np.mean(self.al_samp,axis=0)
		self.y_pred = np.mean(self.predict_samps,axis=0)
		self.y_std = np.std(self.predict_samps,axis=0)
		self.y_cov = np.zeros((self.batch_size,self.num_params,self.num_params))
		for bi in range(self.batch_size):
			self.y_cov[bi] = np.cov(self.predict_samps[:,bi,:],rowvar=False)

		self.samples_init = True

	def gen_coverage_plots(self,color_map = ["#377eb8", "#4daf4a","#e41a1c",
		"#984ea3"],block=True):
		"""
		Generate plots for the coverage of each of the parameters.

		Parameters
		----------
			color_map ([str,...]): A list of at least 4 colors that will be used
				for plotting the different coverage probabilities.
			block (bool): If true, block excecution after plt.show() command.
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before plotting')
		
		plt.figure(figsize=(16,18))
		error = self.y_pred - self.y_test
		cov_masks = [np.abs(error)<=self.y_std, np.abs(error)<2*self.y_std, 
			np.abs(error)<3*self.y_std, np.abs(error)>=3*self.y_std]
		cov_masks_names = ['1 sigma =', '2 sigma =', '3 sigma =', '>3 sigma =']
		for i in range(len(self.final_params)):
			plt.subplot(3, int(np.ceil(self.num_params/3)), i+1)
			for cm_i in range(len(cov_masks)-1,-1,-1):
				cov_mask = cov_masks[cm_i][:,i]
				yt_plot = self.y_test[cov_mask,i]
				yp_plot = self.y_pred[cov_mask,i]
				ys_plot = self.y_std[cov_mask,i]
				plt.errorbar(yt_plot,yp_plot,yerr=ys_plot, fmt='.', 
					c=color_map[cm_i],
					label=cov_masks_names[cm_i]+'%.2f'%(
						np.sum(cov_mask)/self.batch_size))
			# plot confidence interval
			straight = np.linspace(np.min(self.y_pred[:,i]), 
				np.max(self.y_pred[:,i]),10)
			plt.plot(straight, straight, label='',color='k')
			plt.title(self.final_params_print_names[i])
			plt.ylabel('Prediction')
			plt.xlabel('True Value')
			plt.legend()
		plt.show(block)

	def report_stats(self):
		"""
		Print out performance statistics of the model. So far this includes median
		error on each parameter.
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before statistics are '+
				'reported')
		# Median error on each parameter
		medians = np.median(np.abs(self.y_pred-self.y_test),axis=0)
		med_std = np.median(self.y_std,axis=0)
		print('Parameter, Median Abs Error, Median Std')
		for param_i in range(len(self.final_params)):
			print(self.final_params[param_i],medians[param_i],med_std[param_i])

	def plot_posterior_contours(self,image_index,contour_color='#FFAA00',
		block=True):
		"""
		Plot the posterior contours for a specific image along with the image 
		itself.
		
		Parameters
		----------
			image_index (int): The integer index of the image in the validation 
				set to plot the posterior of.
			contour_color (str): A string specifying the color to use for the
				contours
			block (bool): If true, block excecution after plt.show() command.
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before plotting')

		# First plot the image and print its parameter values
		plt.imshow(self.images[image_index][:,:,0])
		plt.colorbar()
		plt.show(block)
		for pi in range(self.num_params):
			print(self.final_params[pi],self.y_test[image_index][pi])

		# Now show the posterior contour for the image
		corner.corner(self.predict_samps[:,image_index,:],bins=20,
				labels=self.final_params_print_names,show_titles=True, 
				plot_datapoints=False,label_kwargs=dict(fontsize=13),
				truths=self.y_test[image_index],levels=[0.68,0.95],
				dpi=1600, color=contour_color,fill_contours=True)
		plt.show(block)

	def comp_al_ep_unc(self,block=True):
		"""
		Generate plots to compare the aleatoric and epistemic uncertainties
		generated by the model.

		Parameters
		----------
			block (bool): If true, block excecution after plt.show() command.
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before plotting')

		# Plot the two uncertanties side by side with the same scale.
		fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,12))
		al_median = np.median(np.abs(self.al_cov),axis=0)
		y_cov_median = np.median(np.abs(self.y_cov),axis=0)
		# Get the maximum and minimum values
		vmax = max(np.max(al_median),np.max(y_cov_median))
		vmin = min(np.min(al_median),np.min(y_cov_median))
		if vmin == 0:
			vmin = np.min(y_cov_median)
		print(vmin,vmax)
		# Plot the aleatoric covariance
		im = axes[0].imshow(al_median,norm=LogNorm(vmin=vmin, vmax=vmax))
		axes[0].set_title('Median Aleatoric Covariance')
		# Plot the full covariance
		im = axes[1].imshow(y_cov_median,norm=LogNorm(vmin=vmin, vmax=vmax))
		axes[1].set_title('Median Aleatoric + Epistemic Covariance')
		# Reset the labels to agree with our parameters
		for ax in axes:
			ax.set_xticklabels([0]+self.final_params_print_names)
			ax.set_yticklabels([0]+self.final_params_print_names)
		# Make room for a nice colorbar
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.35, 0.05, 0.3])
		fig.colorbar(im, cax=cbar_ax)
		plt.show(block)

		#Now we want to plot the ratio to get an idea for how dominant
		# one is over the other.
		fig = plt.figure(figsize=(6,6))
		plt.imshow(al_median/y_cov_median,vmax=1)
		plt.title('Median Aleatoric / Total Covariance')
		fig.axes[0].set_xticklabels([0]+self.final_params_print_names)
		fig.axes[0].set_yticklabels([0]+self.final_params_print_names)
		plt.colorbar()

	def calc_p_dlt(self):
		"""
		Calculate the percentage of draws from the predicted distribution with
		||draws||_2 > ||truth||_2 for all of the examples in the batch

		Notes
		-----
			p_dlt will be set a property of the class
		"""
		self.p_dlt = np.sum(np.square(self.predict_samps-self.y_pred),
			axis=-1) < np.sum(np.square(self.y_test-self.y_pred),axis=-1)
		self.p_dlt = np.mean(self.p_dlt,axis=0)

	def plot_calibration(self,color_map=["#377eb8", "#4daf4a"],n_perc_points=20,
		figure=None,legend=None,show_plot=True,block=True):
		"""
		Plot the percentage of draws from the predicted distributions with
		||draws||_2 > ||truth||_2 for our different batch examples.

		Parameters
		----------
			color_map ([str,...]): A list of the colors to use in plotting.
			n_perc_point (int): The number of percentages to probe in the
				plotting.
			figure (matplotlib.pyplot.figure): A figure that was previously
				returned by plot_calibration to overplot onto.
			legend ([str,...]): The legend to use for plotting.
			show_plot (bool): If true, call plt.show() at the end of the
				function.
			block (bool): If true, block excecution after plt.show() command.


		Returns
		-------
			(matplotlib.pyplot.figure): The figure object that contains the 
				plot

		Notes
		-----
			If the posterior is correctly predicted, it follows that x% of the
			images should have x% of their draws with ||draws||_2 > ||truth||_2. 
			See the Test_Model_Performance notebook for a discussion of this.
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before plotting')
		# Go through each of our examples and see what percentage of draws have
		# ||draws||_2 < ||truth||_2 (essentially doing integration by radius).
		self.calc_p_dlt()
		
		# Plot what percentage of images have at most x% of draws with 
		# p(draws)>p(true).
		percentages = np.linspace(0.0,1.0,n_perc_points)
		p_images = np.zeros_like(percentages)
		if figure is None:
			figure = plt.figure(figsize=(6,6))
			plt.plot(percentages,percentages,c=color_map[0],ls='--')

		# We'll estimate the uncertainty in our plat using a jacknife method.
		p_images_jn = np.zeros((len(self.p_dlt),n_perc_points))
		for pi in range(n_perc_points):
			percent = percentages[pi]
			p_images[pi] = np.mean(self.p_dlt<=percent)
			for ji in range(len(self.p_dlt)):
				samp_p_dlt = np.delete(self.p_dlt,ji)
				p_images_jn[ji,pi] = np.mean(samp_p_dlt<=percent)
		# Estimate the standard deviation from the jacknife
		p_dlt_std = np.sqrt((len(self.p_dlt)-1)*np.mean(np.square(p_images_jn-
			np.mean(p_images_jn,axis=0)),axis=0))
		plt.plot(percentages,p_images,c=color_map[1])
		# Plot the 1 sigma contours from the jacknife estimate to get an idea of
		# our sample variance.
		plt.fill_between(percentages,p_images+p_dlt_std,p_images-p_dlt_std,
			color=color_map[1],alpha=0.3)
		plt.xlabel('Percentage')
		plt.ylabel('Percent of Images that have x% of draws with d(draws)>d(truth)')
		plt.title('Calibration of Network Posterior')
		if legend is None:
			plt.legend(['Perfect Calibration','Network Calibration'])
		else:
			plt.legend(legend)
		if show_plot:
			plt.show(block)

		return figure



