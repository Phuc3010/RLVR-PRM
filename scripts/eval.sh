#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/eval-qwen2.5-3b.out
#SBATCH -e log/error-eval-qwen2.5-3b.out
#SBATCH --mem=64G
#SBATCH --time=1-00:00:0

#SBATCH --job-name=eval-qwen3b
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:4


source ~/.bashrc
conda activate prm_rlvr

python gen_vllm.py