#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=thesisProjectTraining
#SBATCH --mail-type=END
#SBATCH --mail-user=smg8196@nyu.edu
##SBATCH --output=slurm-%j.out
#SBATCH --output=slurmMostRecent.out

module purge

singularity \
    exec --nv --overlay /scratch/smg8196/tensorflow_singularity/tensorflow-overlay.ext3:ro \
    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/smg8196/Experiment_2/Greene; python build_and_train_model.py"