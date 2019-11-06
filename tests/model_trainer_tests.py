import unittest, json, glob, os
# Eliminate TF warning in tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from ovejero import model_trainer
from helpers import dataset_comparison

class TFRecordTests(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TFRecordTests, self).__init__(*args, **kwargs)
		self.root_path = 'test_data/'
		self.lens_params = ['external_shear_e1','external_shear_e2',
			'lens_mass_center_x','lens_mass_center_y','lens_mass_e1',
			'lens_mass_e2','lens_mass_gamma','lens_mass_theta_E_log']
		self.lens_params_path = self.root_path + 'new_metadata.csv'
		self.tf_record_path = self.root_path + 'test_record'

	def test_prepare_tf_record(self):
		# Test that the prepare_tf_record function works as expected.
		with open(self.root_path+'test.json','r') as json_f:
		    cfg = json.load(json_f)
		model_trainer.prepare_tf_record(cfg,self.root_path,
			self.tf_record_path,self.lens_params)

		# Check the TFRecord and make sure the number of parameters and values
		# all seems reasonable.
		num_npy = len(glob.glob(self.root_path+'X*.npy'))
		self.assertTrue(os.path.exists(self.tf_record_path))

		# Open up this TFRecord file and take a look inside
		raw_dataset = tf.data.TFRecordDataset(self.tf_record_path)
		# Define a mapping function to parse the image
		def parse_image(example):
			data_features = {
				'image' : tf.io.FixedLenFeature([],tf.string),
				'height' : tf.io.FixedLenFeature([],tf.int64),
				'width' : tf.io.FixedLenFeature([],tf.int64),
				'index' : tf.io.FixedLenFeature([],tf.int64),
			}
			for lens_param in self.lens_params:
					data_features[lens_param] = tf.io.FixedLenFeature(
						[],tf.float32)
			return tf.io.parse_single_example(example,data_features)
		batch_size = 10
		dataset = raw_dataset.map(parse_image).batch(batch_size)
		dataset_comparison(self,dataset,batch_size,num_npy)

		# Clean up the file now that we're done
		os.remove(self.tf_record_path)
		os.remove(self.root_path+'new_metadata.csv')
		os.remove(self.root_path+'norms.npy')