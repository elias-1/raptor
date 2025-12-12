#!/bin/bash

# Set job name
#SBATCH --job-name=raptor
# Specify the number of nodes and processors and gpus per nodes
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --cpus-per-task=17

#SBATCH --gres=gpu:1

# For ascend cluster, we have nextgen and quad nodes
#SBATCH --partition=nextgen


# Specify the amount of time for this job
#SBATCH --time=36:00:00

# Specify the maximum amount of physical memory required
#SBATCH --mem=128gb

# Specify an account when more than one available
#SBATCH --account=PCON0023

#SBATCH --output=dinov2_processed_data/%j_0_log.out

#SBATCH --error=dinov2_processed_data/%j_0_log.err


# Load modules:
module load cuda/11.8.0

module load miniconda3/24.1.2-py310

source activate slivit

cd /fs/ess/PCON0023/shileicao/code/raptor


python create_projector.py --seed 0 --d 1024 --k 100 --saveas data/proj_normal_d1024_k100_run1

python -u new_embed.py --folder /fs/ess/PCON0023/eye3d/data/ukbiobank/oct \
    --encoder DINOv2 \
    --batch_size 1024 \
    --saveto dinov2_processed_data \
    --k data/proj_normal_d1024_k100_run1.npy