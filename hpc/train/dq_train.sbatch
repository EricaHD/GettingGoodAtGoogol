#!/bin/bash

#SBATCH --job-name=dqa_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_dqa_train_%j.out

#Run.A
PYTHONPATH=$PYTHONPATH:. python dqa_train.py \
    --device cpu \
    --net basic \
    --net_param 2_128 \
    --batch_size 128 \
    --target_update 100 \
    --optimizer rmsprop \
    --loss huber \
    --mem_size 500000 \
    --p_to_s stateMaxV \
    --gamma 0.9 \
    --epsilon 0.1 \
    --eps_decay .00001 \
    --s_cost 0 \
    --v_fn vMax \
    --lo 1 \
    --hi 500 \
    --n_idx 100 \
    --replace False \
    --reward_fn topN \
    --reward 6_6_6 \
    --n_games 1000000 \
    --n_print 100000 \
    --delay 0 \
    --curr_epoch 100000 \
    --curr_params 0_0_1_minus \
    --lo_eval 1 \
    --hi_eval 500 \
    --n_idx_eval 100 \
    --replace_eval False \
    --reward_fn_eval scalar \
    --reward_eval 1_1 \
    --n_games_eval 10000 \
    --n_print_eval 1000 \
    --delay_eval 0 \
    --file_path agents/dqa_train.pkl
