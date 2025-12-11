#!/bin/bash
# Submit all batch jobs to SLURM queue
# Generated for 10 batches

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/jobs/download_batch_001.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_002.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_003.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_004.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_005.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_006.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_007.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_008.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_009.sbatch"
sbatch "$SCRIPT_DIR/jobs/download_batch_010.sbatch"

echo "Submitted all batch jobs"
