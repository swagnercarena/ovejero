import unittest, json, glob, os
from ovejero import model_trainer

class ConfigTest(unittest.TestCase):

	def test_all_configs(self):
		# Make sure that all the configs in the configuration folder are up to
		# snuff. This will pass so long as no error is raised in the model_trainer
		# function.
		root_path = os.path.dirname(os.path.abspath(__file__))+'/../configs/'
		config_files = glob.glob(root_path + '*.json')
		for config_file in config_files:
			with open(config_file,'r') as json_f:
				cfg = json.load(json_f)
			model_trainer.config_checker(cfg)
