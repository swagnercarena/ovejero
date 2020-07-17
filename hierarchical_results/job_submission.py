import os
# Submit the jobs from the list
config_path = ['../configs/nn1_hr.json','../configs/nn2_slr.json',
	'../configs/nn3_slr.json','../configs/nn1_hr.json','../configs/nn2_slr.json',
	'../configs/nn3_slr.json','../configs/nn1_hr.json','../configs/nn2_slr.json',
	'../configs/nn3_slr.json','../configs/nn3_slr.json','../configs/nn3_slr.json',
	'../configs/nn3_lr.json','../configs/nn3.json']

target_ovejero_omega_path = [
	'../configs/baobab_configs/cent_narrow_cfg_prior.py',
	'../configs/baobab_configs/cent_narrow_cfg_prior.py',
	'../configs/baobab_configs/cent_narrow_cfg_prior.py',
	'../configs/baobab_configs/shifted_narrow_cfg_prior.py',
	'../configs/baobab_configs/shifted_narrow_cfg_prior.py',
	'../configs/baobab_configs/shifted_narrow_cfg_prior.py',
	'../configs/baobab_configs/empirical_prior.py',
	'../configs/baobab_configs/empirical_prior.py',
	'../configs/baobab_configs/empirical_prior.py',
	'../configs/baobab_configs/cent_narrow_cfg_prior.py',
	'../configs/baobab_configs/cent_narrow_cfg_prior.py',
	'../configs/baobab_configs/cent_narrow_cfg_prior.py',
	'../configs/baobab_configs/cent_narrow_cfg_prior.py']

test_dataset_path = ['../datasets/cent_narrow/','../datasets/cent_narrow/',
	'../datasets/cent_narrow/','../datasets/shifted_narrow/',
	'../datasets/shifted_narrow/','../datasets/shifted_narrow/',
	'../datasets/empirical/','../datasets/empirical/','../datasets/empirical/',
	'../datasets/cent_narrow/','../datasets/cent_narrow/',
	'../datasets/cent_narrow/','../datasets/cent_narrow/']

test_dataset_tf_record_path = ['../datasets/cent_narrow/tf_record_cn',
	'../datasets/cent_narrow/tf_record_cn','../datasets/cent_narrow/tf_record_cn',
	'../datasets/shifted_narrow/tf_record_sn',
	'../datasets/shifted_narrow/tf_record_sn',
	'../datasets/shifted_narrow/tf_record_sn',
	'../datasets/empirical/tf_record_emp',
	'../datasets/empirical/tf_record_emp',
	'../datasets/empirical/tf_record_emp',
	'../datasets/cent_narrow/tf_record_cn','../datasets/cent_narrow/tf_record_cn',
	'../datasets/cent_narrow/tf_record_cn','../datasets/cent_narrow/tf_record_cn']

sample_save_dir = ['cn_nn1_hr_samps','cn_nn2_slr_samps','cn_nn3_slr_samps',
	'sn_nn1_hr_samps','sn_nn2_slr_samps','sn_nn3_slr_samps','emp_nn1_hr_samps',
	'emp_nn2_slr_samps','emp_nn3_slr_samps','cn_nn3_slr_samps',
	'cn_nn3_slr_samps','cn_nn3_lr_samps','cn_nn3_samps']

chains_save_path = ['cn_nn1_hr.h5','cn_nn2_slr.h5','cn_nn3_slr.h5',
	'sn_nn1_hr.h5','sn_nn2_slr.h5','sn_nn3_slr.h5','emp_nn1_hr.h5',
	'emp_nn2_slr.h5','emp_nn3_slr.h5','cn_nn3_slr_256.h5','cn_nn3_slr_64.h5',
	'cn_nn3_lr.h5','cn_nn3.h5']

num_lens_samples = ['1000']*len(chains_save_path)

num_emcee_samples = ['10000']*len(chains_save_path)

num_lenses = ['0','0','0','0','0','0','0','0','0','256','64','0','0']

for arg_list in zip(config_path,target_ovejero_omega_path,test_dataset_path,
	test_dataset_tf_record_path,sample_save_dir,chains_save_path,num_lens_samples,
	num_emcee_samples,num_lenses):
	os.system('sbatch submit_sciprt.sh ' + ' '.join(arg_list))
