import tensorflow as tf
import numpy as np
import pandas as pd

def dataset_comparison(unittest_class,dataset,batch_size,num_npy):
	# Test that the dataset matches what is saved in the test directory
	# Run the same test as above
	lens_params_csv = pd.read_csv(unittest_class.lens_params_path, index_col=None)
	index_array = []
	npy_counts = 0
	for batch in iter(dataset):
		# Read the image out
		height = batch['height'].numpy()[0]
		width = batch['width'].numpy()[0]
		batch_images = tf.io.decode_raw(batch['image'],
			out_type=np.float32).numpy().reshape(-1,
				height,width)
		npy_indexs = batch['index'].numpy()
		lens_params_batch = []
		for lens_param in unittest_class.lens_params:
			lens_params_batch.append(batch[lens_param].numpy())
		# Load the original image and lens parameters and make sure that they
		# match
		for batch_index in range(batch_size):
			npy_index = npy_indexs[batch_index]
			index_array.append(npy_index)
			image = batch_images[batch_index]
			original_image = np.load(unittest_class.root_path+
				'X_{0:07d}.npy'.format(npy_index)).astype(np.float32)
			unittest_class.assertEqual(np.sum(np.abs(image-original_image)),0)
			lpi = 0
			for lens_param in unittest_class.lens_params:
				lens_param_value = lens_params_batch[lpi][batch_index]
				unittest_class.assertAlmostEqual(lens_param_value,lens_params_csv[
					lens_param][npy_index],places=4)
				lpi += 1
			npy_counts += 1
	# Ensure the total number of files is correct
	unittest_class.assertEqual(npy_counts,num_npy)