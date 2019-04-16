#!/bin/bash

#SBATCH --job-name=qb22s10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_b22_s10_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_training.py \
	--agent q_learn \
    --lo 1 \
    --hi 100 \
    --n_states 50 \
    --alpha .01 \
    --alpha_decay .0001 \
    --alpha_step 100000 \
    --q_key_fn bin_2_2 \
    --n_games 500000 \
    --reward scalar_10_1 \
    --n_print 10000 \
    --delay 0 \
    --lo_eval 1 \
    --hi_eval 100 \
    --n_states 25 \
    --n_games_eval 100000 \
    --reward_eval scalar_1_1 \
    --n_print_eval 5000 \
    --delay_eval 0 \
	--file_path agent_values/q_b22_s10.pkl
