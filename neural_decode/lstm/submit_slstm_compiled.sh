#!/bin/bash
#SBATCH --job-name=slstm_compiled
#SBATCH --account=singhlab
#SBATCH --partition=singhlab-gpu
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00


echo "Running on node: $HOSTNAME"
echo "Starting slstm training (torch.compile)..."

# Ensure we're in the right directory
cd /hpc/home/zy231/gersbachlab/zy231/neural_decode

# Execute the orchestrator
python lstm/orchestrator_run_compiled.py --model slstm

echo "Job finished."
