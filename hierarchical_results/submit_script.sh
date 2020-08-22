#!/bin/bash
#
# h_inf.sbatch
#
#SBATCH -J h_inf         # A single job name for the array
#SBATCH -p kipac,normal,iric,hns,owners
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -t 0-24         # Running time of 1 day
#SBATCH --mem 8000
#SBATCH -o /home/users/swagnerc/slurm_out/h_inf_%j.out # Standard output
#SBATCH -e /home/users/swagnerc/slurm_out/h_inf_%j.err # Standard error
#SBATCH --mail-type=FAIL

cd /home/users/swagnerc/Phil/baobab
pip3 install -e . -r requirements.txt
cd /home/users/swagnerc/Phil/ovejero
python3 setup.py install --user
ml python/3.6.1
cd /home/users/swagnerc/Phil/ovejero/hierarchical_results
python3 -m run_hierarchical_inference ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10}
