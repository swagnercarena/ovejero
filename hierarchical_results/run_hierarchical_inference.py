from ovejero import model_trainer, hierarchical_inference
import argparse, os
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Run hiearchical inference on'+
	' a specific test set')
parser.add_argument('config_path', type=str,
	help='The path to the ovejero config used to train the model.')
parser.add_argument('target_ovejero_omega_path', type=str,
	help='The ovejero config containing the target distribution')
parser.add_argument('test_dataset_path', type=str,
	help='The path to the test data')
parser.add_argument('test_dataset_tf_record_path', type=str,
	help='The path to the test data tfrecord')
parser.add_argument('sample_save_dir', type=str,
	help='The directory to save samples')
parser.add_argument('chains_save_path', type=str,
	help='The path to save the chains')
parser.add_argument('num_lens_samples', type=int,
	help='The number of lens samples to use')
parser.add_argument('num_emcee_samples', type=int,
	help='The number of emcee samples to take')
parser.add_argument('num_lenses', type=int,
	help='The number of lenses to use from test set. If None use all' +
	'the lenses. 0 is used to indicate all the lenses.')

args = parser.parse_args()
if args.num_lenses == 0:
	args.num_lenses = None

# First specify the config path
root_path = os.getcwd()[:-20]
config_path = args.config_path

# We also need the path to the baobab configs for the interim and target omega
interim_baobab_omega_path = os.path.join(root_path,
	'configs/baobab_configs/train_diagonal.py')
target_ovejero_omega_path = args.target_ovejero_omega_path

test_dataset_path = args.test_dataset_path
test_dataset_tf_record_path = args.test_dataset_tf_record_path

# Check that the config has what you need
cfg = model_trainer.load_config(config_path)

# Correct any path issues.
def recursive_str_checker(cfg_dict):
    for key in cfg_dict:
        if isinstance(cfg_dict[key],str):
            cfg_dict[key] = cfg_dict[key].replace('/home/swagnercarena/ovejero/',
            	root_path)
        if isinstance(cfg_dict[key],dict):
            recursive_str_checker(cfg_dict[key])
recursive_str_checker(cfg)

# The InferenceClass will do all the heavy lifting of preparing the model from
# the configuration file,
# initializing the validation dataset, and providing outputs correctly
# marginalized over the BNN uncertainties.
hier_infer = hierarchical_inference.HierarchicalClass(cfg,
	interim_baobab_omega_path,target_ovejero_omega_path,test_dataset_path,
	test_dataset_tf_record_path)

# Now we just have to ask the InferenceClass to spin up some samples from our BNN.
# The more samples, the more accurate our plots and metrics will be. The right
# value to use unfortunately requires a bit of trial and error.
hier_infer.gen_samples(args.num_lens_samples,args.sample_save_dir,
	args.num_lenses)

n_walkers = 50
pool = Pool()
hier_infer.initialize_sampler(n_walkers,args.chains_save_path,pool=pool)

hier_infer.run_sampler(args.num_emcee_samples,progress=True)

