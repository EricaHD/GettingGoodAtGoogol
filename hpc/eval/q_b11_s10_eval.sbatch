#!/bin/bash

#SBATCH --job-name=qb11s10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_b11_s10_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_eval.py \
	--agent q_learn \
    --agent_path agent_values/q_b11_t33.pkl \
    --lo 1 \
    --hi 1000 \
    --n_states 100 \
    --n_games 500000 \
    --reward scalar_10_1 \
    --n_print 10000 \
    --delay 0 \