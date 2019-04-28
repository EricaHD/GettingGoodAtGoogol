#!/bin/bash

#SBATCH --job-name=q_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_q_train_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python q_train.py \
    --alpha 0.001 \
    --alpha_decay .0001 \
    --alpha_step 100000000000 \
    --gamma 0.9 \
    --epsilon 0.1 \
    --eps_decay .00001 \
    --s_cost 0 \
    --q_learn False \
    --q_key_fn binV \
    --q_key_params 2_4  \
    --v_fn vMax \
    --lo 1 \
    --hi 1000 \
    --n_idx 50 \
    --replace False \
    --reward_fn topN \
    --reward 10_10_10 \
    --n_games 1000000 \
    --n_print 100000 \
    --delay 0 \
    --curr_epoch 100000 \
    --curr_params 0_0_1_minus \
    --lo_eval 1 \
    --hi_eval 1000 \
    --n_idx_eval 50 \
    --replace_eval False \
    --reward_fn_eval scalar \
    --reward_eval 1_1 \
    --n_games_eval 10000 \
    --n_print_eval 1000 \
    --delay_eval 0 \
    --file_path agents/q_train.pkl
