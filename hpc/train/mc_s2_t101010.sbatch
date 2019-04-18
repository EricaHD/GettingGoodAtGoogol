#!/bin/bash

#SBATCH --job-name=mcs2t101010
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_mc_s2_t101010_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_training.py \
	--agent mcmc \
    --lo 1 \
    --hi 100 \
    --n_idx 50 \
    --reward topN_10_10_10 \
    --q_key_fn seq_2 \
    --v_fn vSeq \
    --n_episodes 25000 \
    --lo_eval 1 \
    --hi_eval 100 \
    --n_idx_eval 50 \
    --reward_eval scalar_1_1 \
    --n_games_eval 25000 \
    --n_print_eval 5000 \
    --delay_eval 0 \
    --file_path agent_values/mc_s2_t101010.pkl
