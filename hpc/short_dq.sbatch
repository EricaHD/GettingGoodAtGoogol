#!/bin/bash

#SBATCH --job-name=dq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=slurm_dq_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ehd255@nyu.edu

module purge
module load python3/intel/3.6.3
source /home/ehd255/pyenv/py3.6.3/bin/activate

python /home/ehd255/DeepGoogol/train_dq.py \
    --lo 1 \
    --hi 100000 \
    --n_idx 50 \
    --replace False \
    --reward_fn topN \
    --reward 10_10_10 \
    --lo_eval 1 \
    --hi_eval 100000 \
    --n_idx_eval 50 \
    --replace_eval False \
    --reward_fn_eval scalar \
    --reward_eval 1_1 \
    --file_path /home/ehd255/DeepGoogol/results/dq.pkl \
    --sc_file_path /home/ehd255/DeepGoogol/results/dq_sc.pkl
