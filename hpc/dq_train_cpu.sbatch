#!/bin/bash

#SBATCH --job-name=dqa_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96000
#SBATCH --time=24:00:00
#SBATCH --output=slurm_dqa_train_cpu_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python dqa_train.py \
    --device cpu \
    --net basic \
    --net_params 2_128_0.0 \
    --batch_size 128 \
    --target_update 1000 \
    --optimizer rmsprop \
    --loss huber \
    --mem_size 100000 \
    --p_to_s stateMax \
    --gamma 0.9 \
    --epsilon 0.1 \
    --eps_decay .00001 \
    --s_cost 0 \
    --v_fn vIdx \
    --lo 1 \
    --hi 100000 \
    --n_idx 50 \
    --replace False \
    --reward_fn topN \
    --reward 10_10_11 \
    --n_games 10 \
    --n_print 1 \
    --delay 0 \
    --curr_epoch 10000 \
    --curr_params 0_0_1_minus \
    --lo_eval 1 \
    --hi_eval 100000 \
    --n_idx_eval 50 \
    --replace_eval False \
    --reward_fn_eval scalar \
    --reward_eval 1_1 \
    --n_games_eval 10 \
    --n_print_eval 1000 \
    --delay_eval 0 \
    --file_path agents/dqa_train_cpu
