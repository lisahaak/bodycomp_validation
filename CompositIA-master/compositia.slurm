#!/bin/bash

#SBATCH --job-name=compositia_test
#SBATCH --output=/home/jjkool/bep/output_%j.log
#SBATCH --error=/home/jjkool/bep/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mail-user=j.j.kool@lumc.nl
#SBATCH --mail-type=ALL

# Environment setup
module purge
module load tools/miniconda/python3.9

# Activate the correct conda environment
source activate env_compositia_py39

# Change to working directory
cd /home/jjkool/bep/compositIA/

# Run model on three input sets
python CompositIA_l3_seg_folders.py \
--input_path /home/jjkool/bep/data/inputs/segmentation_compo_selected_a/ \
--output_folder /home/jjkool/bep/data/outputs/segmentation_compo_selected_a/ \
--weights_L3 /home/jjkool/bep/models_bodycompia/unet_L3.hdf5

python CompositIA_l3_seg_folders.py \
--input_path /home/jjkool/bep/data/inputs/segmentation_compo_selected_b1/ \
--output_folder /home/jjkool/bep/data/outputs/segmentation_compo_selected_b1/ \
--weights_L3 /home/jjkool/bep/models_bodycompia/unet_L3.hdf5

python CompositIA_l3_seg_folders.py \
--input_path /home/jjkool/bep/data/inputs/segmentation_compo_selected_b2/ \
--output_folder /home/jjkool/bep/data/outputs/segmentation_compo_selected_b2/ \
--weights_L3 /home/jjkool/bep/models_bodycompia/unet_L3.hdf5
