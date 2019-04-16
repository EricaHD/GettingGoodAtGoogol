#!/bin/bash

#SBATCH --job-name=qAgent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_seq2_top10_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_training.py \
	--agent q_learn \
    --lo 1 \
    --hi 100 \
    --n_states 25 \
    --n_games 10_000_000 \
	--q_key_fn seq_2 \
    --reward topN_10_10_10 \
	--file_path agent_values/q_seq2_top10.pkl
