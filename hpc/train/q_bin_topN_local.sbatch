#!/bin/bash

#SBATCH --job-name=qbt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_qbt_%j.out

python agent_training.py \
    --agent q_learn \
    --alpha 0.01 \
    --alpha_decay 0.00001 \
    --alpha_step 100000 \
    --gamma 0.85 \
    --epsilon 0.1 \
    --eps_decay 0.00001 \
    --s_cost 0 \
    --q_learn False \
    --q_key_fn bin \
    --q_key_params 2_4  \
    --v_fn vMax \
    --lo 1 \
    --hi 10000 \
    --n_idx 50 \
    --replace False \
    --reward_fn topN \
    --reward 10_10_11 \
    --n_games 1000000 \
    --n_print 100000 \
    --delay 0 \
    --curr_epoch 100000 \
    --curr_params 0_0_1_minus \
    --lo_eval 1 \
    --hi_eval 10000 \
    --n_idx_eval 50 \
    --replace_eval False \
    --reward_fn_eval scalar \
    --reward_eval 10_1 \
    --n_games_eval 10000 \
    --n_print_eval 1000 \
    --delay_eval 0 \
    --file_path agent_values/qbt.pkl
