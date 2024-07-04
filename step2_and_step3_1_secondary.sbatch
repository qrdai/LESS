#!/bin/bash
#SBATCH --time=4:00:00                     # Job run time (hh:mm:ss)
#SBATCH --partition=secondary               # switch between haopeng and secondary
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks-per-node=1                 # Number of task (cores/ppn) per node
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=step2_and_step3_1_llama2_7b_bsz32_remaining_4          # Name of batch job
#SBATCH --output=step2_and_step3_1_llama2_7b_bsz32_remaining_4.o%j        # Name of batch job output file
##SBATCH --error=step2_and_step3_1_llama2_7b_bsz32_remaining_4.e%j        # Name of batch job error file
#SBATCH --mail-user=qirundai@illinois.edu  # Send email notifications
#SBATCH --mail-type=BEGIN,END              # Type of email notifications to send


module purge                            # Clear all loaded modules
module load gcc/11.2.0
module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate less                     # Activate your specific Conda environment


# User scripting: any number of any Linux command (python/shell, etc.) can be executed here
# ./step2_remaining_secondary.sh
./step3_1.sh
