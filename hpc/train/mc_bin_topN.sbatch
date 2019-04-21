#!/bin/bash

#SBATCH --job-name=mbt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_mbt_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ehd255@nyu.edu

module purge
module load python3/intel/3.6.3
souce /home/ehd255/pyenv/py3.6.3/bin/activate

python /home/ehd255/DeepGoogol/agent_training.py \
    --agent mcmc \
    --gamma 0.9 \
    --epsilon 0.1 \
    --eps_decay 0.00001 \
    --s_cost 0 \
    --q_key_fn bin \
    --q_key_params 2_2  \
    --v_fn vMax \
    --lo 1 \
    --hi 10000 \
    --n_idx 50 \
    --replace False \
    --reward_fn topN \
    --reward 10_10_3 \
    --n_episodes 1000000 \
    --curr_epoch 70000000 \
    --curr_params 0_0_10_minus \
    --lo_eval 1 \
    --hi_eval 10000 \
    --n_idx_eval 50 \
    --replace_eval False \
    --reward_fn_eval scalar \
    --reward_eval 10_1 \
    --n_games_eval 10000 \
    --n_print_eval 1000 \
    --delay_eval 0 \
    --file_path /home/ehd255/DeepGoogol/agent_values/mbt.pkl
