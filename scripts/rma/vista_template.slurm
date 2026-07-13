#!/bin/bash
#SBATCH --job-name=air_hockey_rma
#SBATCH --output=slurm_jobs/job_%j/job_%j.out
#SBATCH --error=slurm_jobs/job_%j/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=fail
#SBATCH --mail-user=rpp689@eid.utexas.edu
#SBATCH --time=00:20:00
#SBATCH --partition=gh
#SBATCH --account=ASC24079

echo $SHELL
echo $SLURM_JOB_NODELIST
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/.zshrc

# RMA lives in air-hockey-rl-rma; reuse the same venv as the TD3 vista jobs.
cd /work/10993/rohanpatel01/vista/air-hockey-rl-rma
source /work/10993/rohanpatel01/vista/air-hockey-rl/.venv/bin/activate
