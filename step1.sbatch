#!/bin/bash
#SBATCH --time=10:00:00                     # Job run time (hh:mm:ss)
#SBATCH --partition=haopeng
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1                 # Number of task (cores/ppn) per node
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --job-name=step1_mistral_7b_v01_bsz32          # Name of batch job
#SBATCH --output=step1_mistral_7b_v01_bsz32.o%j        # Name of batch job output file
##SBATCH --error=step1_mistral_7b_v01_bsz32.e%j        # Name of batch job error file
#SBATCH --mail-user=qirundai@illinois.edu  # Send email notifications
#SBATCH --mail-type=BEGIN,END              # Type of email notifications to send


module purge                            # Clear all loaded modules
module load gcc/11.2.0
module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate less                     # Activate your specific Conda environment


# User scripting: any number of any Linux command (python/shell, etc.) can be executed here
./step1.sh
