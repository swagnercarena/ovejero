# -*- coding: utf-8 -*-
"""
This module contains the functions neccesary for conducting inference using a
trained bnn.

Examples
--------
The demo Test_Model_Performance.ipynb gives a number of examples on how to use
this module.
"""
from ovejero import model_trainer, data_tools
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import corner

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
		self.model, self.loss = model_trainer.model_loss_builder(cfg)

		# Load the validation set we're going to use.
		self.tf_record_path_v = (cfg['validation_params']['root_path']+
			cfg['validation_params']['tf_record_path'])
		# Load the parameters and the batch size needed for computation
		self.final_params = cfg['training_params']['final_params']
		self.num_params = len(self.final_params)
		self.batch_size = cfg['training_params']['batch_size']
		self.norm_images = cfg['training_params']['norm_images']

		self.tf_dataset_v = data_tools.build_tf_dataset(self.tf_record_path_v,
			self.final_params,self.batch_size,1,norm_images=self.norm_images)

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

	def undo_param_norm(self,predict_samps,y_test):
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

		Notes
		-----
			Correction to numpy arrays is done in place
		"""

		# Get the normalization constants used from the csv file.
		normalization_constants_path = (self.cfg['training_params']['root_path']+
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

	def gen_samples(self,num_samples):
		"""
		Generate the y prediction and the associated covariance matrix
		by marginalizing over both network and output uncertainty.

		Parameters
		----------
			num_samples (int): The number of samples used to marginalize over
				the network's uncertainty.
		"""
		# Extract image and output batch from tf dataset
		for image_batch, yt_batch in self.tf_dataset_v.take(1):
			self.images = image_batch.numpy()
			self.y_test = yt_batch.numpy()

		# This is where we will save the samples for each prediction. We will
		# use this to numerically extract the covariance.
		predict_samps = np.zeros((num_samples,self.batch_size,self.num_params))
		# We also want to store a sampling of the alleatoric noise being
		# predicted to get a handle on how it compares to the epistemic
		# uncertainty.
		all_samp = np.zeros((num_samples,self.batch_size,self.num_params,
			self.num_params))
		# Generate our samples
		for samp in tqdm(range(num_samples)):
			# How we extract uncertanties will depend on the type of network in
			# question.
			if self.bnn_type=='diag':
				output_sample = self.model.predict(self.images)
				predict_samps[samp] = output_sample[:,:self.num_params] 
				noise = np.random.randn(self.batch_size*self.num_params).reshape(
					(self.batch_size,self.num_params))
				predict_samps[samp] += noise*np.exp(output_sample[:,
					self.num_params:])
				all_samp[samp] = output_sample[:,self.num_params:]
			else:
				raise NotImplementedError('gen_pred_cov does not yet support '+
					'%s models'%(self.bnn_type))

		self.fix_flip_pairs(predict_samps,self.y_test)
		self.undo_param_norm(predict_samps,self.y_test)

		self.predict_samps = predict_samps
		self.all_cov = np.mean(all_samp,axis=0)
		self.y_pred = np.mean(predict_samps,axis=0)
		self.y_std = np.std(predict_samps,axis=0)
		self.y_cov = np.zeros((self.batch_size,self.num_params,self.num_params))
		for bi in range(self.batch_size):
			self.y_cov[bi] = np.cov(predict_samps[:,bi,:],rowvar=False)

		self.samples_init = True

	def gen_coverage_plots(self,color_map = ["#377eb8", "#4daf4a","#e41a1c",
		"#984ea3"]):
		"""
		Generate plots for the coverage of each of the parameters.

		Parameters
		----------
			color_map ([str,...]): A list of at least 4 colors that will be used
				for plotting the different coverage probabilities.
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
			plt.title(self.final_params[i])
			plt.ylabel('Prediction')
			plt.xlabel('True Value')
			plt.legend()
		plt.show()

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

	def plot_posterior_contours(self,image_index,contour_color='#FFAA00'):
		"""
		Plot the posterior contours for a specific image along with the image 
		itself.
		
		Parameters
		----------
			image_index (int): The integer index of the image in the validation 
				set to plot the posterior of.
			contour_color (str): A string specifying the color to use for the
				contours
		"""
		if self.samples_init == False:
			raise RuntimeError('Must generate samples before plotting')

		# First plot the image and print its parameter values
		plt.imshow(self.images[image_index][:,:,0])
		plt.colorbar()
		plt.show()
		for pi in range(self.num_params):
			print(self.final_params[pi],self.y_test[image_index][pi])

		# Now show the posterior contour for the image
		corner.corner(self.predict_samps[:,image_index,:],bins=20,
				labels=self.final_params,show_titles=True, 
				plot_datapoints=False,label_kwargs=dict(fontsize=13),
				truths=self.y_test[image_index],levels=[0.68,0.95],
				dpi=1600, color=contour_color,fill_contours=True)
		plt.show()





