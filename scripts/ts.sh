#!/bin/bash
#SBATCH --job-name=tinystories
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH --time=05:00:00
#SBATCH --output=tinystories_%j.out
#SBATCH --error=tinystories_%j.err

uv run /home/c-nmeist/cs336_a1/cs336_basics/bpe_works.py --dataset TS 
