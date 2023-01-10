#!/bin/bash

#SBATCH -o job_%j_%N.out
#SBATCH --partition=gpu
#SBATCH -J poisson_blocks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00

nvprof ./cuda_gs_rb_float_exec
