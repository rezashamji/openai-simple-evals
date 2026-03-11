#!/bin/bash
#SBATCH --job-name=eval-5000q-sweeps-all
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10/slurm_%j.out
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10/slurm_%j.err
#SBATCH --gres=gpu:1

set -e

SCRIPT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/orchestration_scripts"
BASE_OUTPUT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/simple-evals/results/5000q_sweeps_mar10"

mkdir -p "$BASE_OUTPUT_DIR"

bash "$SCRIPT_DIR/orchestrate_5000q_sweeps.sh" "$BASE_OUTPUT_DIR"
