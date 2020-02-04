#!/bin/bash
#
# h_inf.sbatch
#
#SBATCH -J h_inf         # A single job name for the array
#SBATCH -p kipac,normal,iric,hns 
#SBATCH -n 4              
#SBATCH -N 1              
#SBATCH -t 0-24         # Running time of 1 day
#SBATCH --mem 8000
#SBATCH -o /home/users/swagnerc/slurm_out/h_inf_%j.out # Standard output
#SBATCH -e /home/users/swagnerc/slurm_out/h_inf_%j.err # Standard error

ml python/3.6.1
ml gcc/9.1.0
cd /home/users/swagnerc/Phil/fastell4py
python setup.py install --user
cd /home/users/swagnerc/Phil/baobab
pip3 install -e . -r requirements.txt
pip3 install tensorflow==2.1.0 --user
cd /home/users/swagnerc/Phil/ovejero
python setup.py install --user
cd /home/users/swagnerc/Phil/ovejero/hierarchical_results
python3 run_hierarchical_inference.py ${1} ${2} ${3} ${4} ${5} ${6} ${7}
