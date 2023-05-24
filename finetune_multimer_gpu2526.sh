#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -e finetune_multimer_gpu2526_error.txt
#SBATCH -o finetune_multimer_gpu2526_out.txt
#SBATCH -p gpu-el8
#SBATCH -N 1
#SBATCH --qos=high
#SBATCH --mem=500G
#SBATCH --gres="gpu:4" 
#SBATCH -C gpu="A100"
# #SBATCH --nodelist=gpu[25-26]
module load Anaconda3
source activate unifold_3.8
./finetune_multimer.sh /g/alphafold/unifold_multimer_train_db /scratch/gyu/finetune_unifold_multimer /g/alphafold/unifold_parameters/multimer.unifold.pt multimer_af2
