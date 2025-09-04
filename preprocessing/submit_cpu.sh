#!/bin/bash -l
#
# Multithreading example job script for MPCDF Raven.
# In addition to the Python example shown here, the script
# is valid for any multi-threaded program, including
# Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH -J PYTHON_MT     # Job name
#SBATCH -o ./out.%j      # console output saved in this file
#SBATCH -e ./err.%j      # console errors saved in this file
#SBATCH -D ./            # Set the working directory
#SBATCH --ntasks=1       # maximum number of tasks (default: one task per node)
#SBATCH --cpus-per-task=8  # 8 cores on a shared node
#SBATCH --mem=16000MB      # memory limit for the job
#SBATCH --time=2:00:00     # run for 10 minutes

# load necessary modules/ softwares
module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05
conda deactivate
# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

conda activate my_env
export PYTHONPATH=/u/mp040/conda-envs/my_env/lib/python3.10/site-packages:$PYTHONPATH

srun python3 ./era5_prep.py