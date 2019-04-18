#!/bin/bash

#SBATCH --job-name=qs2t101010
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_s2_t101010_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_training.py \
	--agent q_learn \
    --lo 1 \
    --hi 500 \
    --n_idx 100 \
    --alpha .01 \
    --alpha_decay .0001 \
    --alpha_step 25000 \
    --q_key_fn seq_2 \
    --v_fn vSeq \
    --n_games 500000 \
    --reward topN_10_10_10 \
    --n_print 50000 \
    --delay 0 \
    --lo_eval 1 \
    --hi_eval 100 \
    --n_idx_eval 50 \
    --n_games_eval 100000 \
    --reward_eval scalar_1_1 \
    --n_print_eval 5000 \
    --delay_eval 0 \
    --file_path agent_values/q_s2_t101010.pkl
