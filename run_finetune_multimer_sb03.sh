#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -e finetune_multimer_error.txt
#SBATCH -o finetune_multimer_out.txt
#SBATCH -p gpu-el8
#SBATCH -N 4
#SBATCH --qos=high
#SBATCH --mem-per-gpu=127GB
#SBATCH --gres="gpu:1" 
#SBATCH -C gpu="A100"
#SBATCH --exclude=gpu[25-26]
module load Anaconda3
source activate unifold_3.8
./finetune_multimer.sh /g/alphafold/unifold_multimer_train_db /scratch/gyu/finetune_unifold_multimer_sb03 /g/alphafold/unifold_parameters/uf_symmetry.pt multimer_af2
