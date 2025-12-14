#!/bin/bash

# Set job name
#SBATCH --job-name=multilabel_classification-dinov3
# Specify the number of nodes and processors per nodes
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=17

# For ascend cluster, we have nextgen and quad nodes
#SBATCH --partition=nextgen


# Specify the amount of time for this job
#SBATCH --time=12:00:00

# Specify the maximum amount of physical memory required
#SBATCH --mem=128gb

# Specify an account when more than one available
#SBATCH --account=PCON0023

#SBATCH --output=multilabel_classification-dinov3/%j_0_log.out

#SBATCH --error=multilabel_classification-dinov3/%j_0_log.err

module load miniconda3/24.1.2-py310

source activate raptor

cd /fs/ess/PCON0023/shileicao/code/raptor

python multilabel_classification.py \
    --split_json /fs/ess/PCON0023/eye3d/data/ukbiobank/train_val_test2.json \
    --data_path dinov3_processed_data/proj_normal_d1024_k100_run2
