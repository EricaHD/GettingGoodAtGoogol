#!/bin/bash

#SBATCH --job-name=qb21t101010
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_b21_t101010_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python agent_eval.py \
    --agent q_learn \
    --agent_path agent_values/q_b21_t101010.pkl \
    --lo 1 \
    --hi 100 \
    --n_idx 50 \
    --n_games 50000 \
    --n_print 5000 \
    --delay 0 \