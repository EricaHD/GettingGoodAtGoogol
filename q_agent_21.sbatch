#!/bin/bash

#SBATCH --job-name=qAgent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_qAgent%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_training.py \
	--agent q_learn \
        --lo 1 \
        --hi 100 \
        --n_states 25 \
        --n_games 1_000_000_000 \
	--q_key_fn bin21 \
	--file_path agent_values/agent_2.pkl
