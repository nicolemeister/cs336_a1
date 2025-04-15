#!/bin/bash
#SBATCH --job-name=openwebtext
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=owt_%j.out
#SBATCH --error=owt_%j.err

uv run /home/c-nmeist/cs336_a1/cs336_basics/bpe_works.py --dataset OWT 
