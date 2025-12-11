#!/bin/bash
#SBATCH --job-name=arxiv-dl-003
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm04
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/m000152-pm04/arxiv-preprint-parsing/utils/download_arxiv_pdfs/download_latest_arxiv_pdfs/slurm_jobs/logs/download_batch_003_%j.out
#SBATCH --error=/scratch/m000152-pm04/arxiv-preprint-parsing/utils/download_arxiv_pdfs/download_latest_arxiv_pdfs/slurm_jobs/logs/download_batch_003_%j.err

# Print job information
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: arxiv-dl-003"
echo "Batch Index: 3"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Change to working directory
cd /scratch/m000152-pm04/arxiv-preprint-parsing/utils/download_arxiv_pdfs/download_latest_arxiv_pdfs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate comet-inference

# Print environment info
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Run download
echo "Starting download for batch 3..."
python /scratch/m000152-pm04/arxiv-preprint-parsing/utils/download_arxiv_pdfs/download_latest_arxiv_pdfs/download_batch.py 3 \
    --slurm-dir /scratch/m000152-pm04/arxiv-preprint-parsing/utils/download_arxiv_pdfs/download_latest_arxiv_pdfs/slurm_jobs \
    --output-dir /scratch/m000152-pm04/arxiv-preprint-parsing/utils/download_arxiv_pdfs/download_latest_arxiv_pdfs/pdfs \
    --parallel 8

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Batch 3 completed successfully!"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Batch 3 failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
