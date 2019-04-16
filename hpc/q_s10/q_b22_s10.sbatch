#!/bin/bash

#SBATCH --job-name=qAgentB22S10
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
    --n_states 25 \
    --alpha .01 \
    --alpha_decay .0001 \
    --alpha_step 50000 \
    --q_key_fn bin_2_2 \
    --n_games 1000000 \
    --reward scalar_10_1 \
    --n_print 50000 \
    --delay 0 \
    --n_games_eval 100000 \
    --reward_eval scalar_1_1 \
    --n_print_eval 5000 \
    --delay_eval 0 \
	--file_path agent_values/q_b22_s10.pkl
