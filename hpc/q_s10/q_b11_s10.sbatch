#!/bin/bash

#SBATCH --job-name=qAgentB11S10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_b11_s10_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_training.py \
	--agent q_learn \
    --lo 1 \
    --hi 1000 \
    --n_states 50 \
    --alpha .01 \
    --alpha_decay .0001 \
    --alpha_step 500000 \ 
    --q_key_fn bin_1_1 \
    --n_games 10000000 \
    --reward scalar_10_1 \
    --n_print 500000 \
    --delay 0 \
    --n_games_eval 100000 \
    --reward_eval scalar_1_1 \
    --n_print_eval 5000 \
    --delay_eval 0 \
	--file_path agent_values/q_b11_s10.pkl
