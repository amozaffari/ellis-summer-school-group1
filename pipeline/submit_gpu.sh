#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_gpu
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --time=0:10:00    # run for 1 hour

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05
conda deactivate
conda init bash

echo "Job started at: $(date)"
# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#conda activate /u/mp002/conda-envs/neuraltransport
#torchrun --nproc_per_node=1 ./train.py
conda activate ellis
export PYTHONPATH=/u/mp040/conda-envs/ellis/lib/python3.12/site-packages:$PYTHONPATH

#python ./train.py
#python ./train_1.py
#python ./train.py

python train_multi.py --config configs/config_tinycnn.json

# python train_multi.py --config configs/config_deepercnn.json

# python train_multi.py --config configs/config_linear.json

# python train_multi.py --config configs/config_convlstm.json