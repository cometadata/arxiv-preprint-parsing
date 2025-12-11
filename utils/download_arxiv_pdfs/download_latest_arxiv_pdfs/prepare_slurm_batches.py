import os
import sys
import argparse
import math
from pathlib import Path
from tqdm import tqdm
from glob import glob


SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=arxiv-dl-{batch_idx:03d}
#SBATCH -p batch
#SBATCH -A marlowe-m000152-pm04
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task={cpus}
#SBATCH --time=24:00:00
#SBATCH --output={logs_dir}/download_batch_{batch_idx:03d}_%j.out
#SBATCH --error={logs_dir}/download_batch_{batch_idx:03d}_%j.err

# Print job information
echo "========================================="
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Job Name: arxiv-dl-{batch_idx:03d}"
echo "Batch Index: {batch_idx}"
echo "Node: ${{SLURMD_NODENAME}}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Change to working directory
cd {work_dir}

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {conda_env}

# Print environment info
echo "Conda environment: ${{CONDA_DEFAULT_ENV}}"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Run download
echo "Starting download for batch {batch_idx}..."
python {script_dir}/download_batch.py {batch_idx} \\
    --slurm-dir {slurm_dir} \\
    --output-dir {output_dir} \\
    --parallel {cpus}

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Batch {batch_idx} completed successfully!"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Batch {batch_idx} failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi
'''


def load_manifest(manifest_path: str) -> list[tuple[str, str]]:
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
    return items

def filter_existing(
    items: list[tuple[str, str]],
    output_dir: str
) -> list[tuple[str, str]]:
    existing = set()
    for root, _, files in os.walk(output_dir):
        for fname in tqdm(files, desc="walking output"):
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, output_dir)
            existing.add(rel)

    # filter remaining
    remaining = []
    for gcs_path, local_path in tqdm(items, desc="filtering"):
        if local_path not in existing:
            remaining.append((gcs_path, local_path))

    return remaining



def split_into_batches(
    items: list[tuple[str, str]],
    batch_size: int | None = None,
    num_batches: int | None = None
) -> list[list[tuple[str, str]]]:
    total = len(items)

    if batch_size:
        num_batches = math.ceil(total / batch_size)
    elif num_batches:
        batch_size = math.ceil(total / num_batches)
    else:
        raise ValueError("Either batch_size or num_batches must be specified")

    batches = []
    for i in range(0, total, batch_size):
        batches.append(items[i:i + batch_size])

    return batches


def write_batch_manifests(
    batches: list[list[tuple[str, str]]],
    batches_dir: str
) -> None:
    os.makedirs(batches_dir, exist_ok=True)

    for idx, batch in enumerate(batches, start=1):
        batch_file = os.path.join(batches_dir, f"batch_{idx:03d}.txt")
        with open(batch_file, 'w') as f:
            for gcs_path, local_path in batch:
                f.write(f"{gcs_path}\t{local_path}\n")


def write_slurm_jobs(
    num_batches: int,
    slurm_dir: str,
    work_dir: str,
    output_dir: str,
    cpus: int,
    conda_env: str
) -> None:
    jobs_dir = os.path.join(slurm_dir, "jobs")
    logs_dir = os.path.join(slurm_dir, "logs")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for idx in range(1, num_batches + 1):
        job_content = SLURM_TEMPLATE.format(
            batch_idx=idx,
            work_dir=work_dir,
            output_dir=output_dir,
            slurm_dir=slurm_dir,
            script_dir=script_dir,
            logs_dir=logs_dir,
            cpus=cpus,
            conda_env=conda_env
        )
        job_file = os.path.join(jobs_dir, f"download_batch_{idx:03d}.sbatch")
        with open(job_file, 'w') as f:
            f.write(job_content)


def write_submit_script(
    num_batches: int,
    slurm_dir: str
) -> None:
    jobs_dir = os.path.join(slurm_dir, "jobs")
    submit_script = os.path.join(slurm_dir, "submit_all.sh")

    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all batch jobs to SLURM queue\n")
        f.write(f"# Generated for {num_batches} batches\n\n")

        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')

        for idx in range(1, num_batches + 1):
            f.write(f'sbatch "$SCRIPT_DIR/jobs/download_batch_{idx:03d}.sbatch"\n')

        f.write("\necho \"Submitted all batch jobs\"\n")

    os.chmod(submit_script, 0o755)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SLURM batch jobs for downloading arXiv PDFs"
    )
    parser.add_argument(
        'manifest',
        help='Path to master manifest.tsv'
    )
    parser.add_argument(
        'output_dir',
        help='Directory where PDFs are/will be downloaded'
    )
    parser.add_argument(
        '--batches',
        type=int,
        metavar='N',
        help='Number of batches to create'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        metavar='N',
        help='Files per batch (alternative to --batches)'
    )
    parser.add_argument(
        '--slurm-dir',
        default='./slurm_jobs',
        help='Directory for SLURM job files (default: ./slurm_jobs/)'
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=8,
        help='CPUs per task / parallel downloads (default: 8)'
    )
    parser.add_argument(
        '--conda-env',
        default='comet-inference',
        help='Conda environment to activate (default: comet-inference)'
    )
    parser.add_argument(
        '--work-dir',
        help='Working directory for jobs (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without writing files'
    )

    args = parser.parse_args()

    if not args.batches and not args.batch_size:
        parser.error("Either --batches or --batch-size must be specified")

    if args.batches and args.batch_size:
        parser.error("Cannot specify both --batches and --batch-size")

    work_dir = args.work_dir or os.getcwd()
    output_dir = os.path.abspath(args.output_dir)
    slurm_dir = os.path.abspath(args.slurm_dir)

    print(f"Loading manifest from {args.manifest}...", file=sys.stderr)
    items = load_manifest(args.manifest)
    print(f"Manifest contains {len(items)} files", file=sys.stderr)

    print(f"Checking for existing files in {output_dir}...", file=sys.stderr)
    os.makedirs(output_dir, exist_ok=True)
    remaining = filter_existing(items, output_dir)
    skipped = len(items) - len(remaining)
    print(f"Skipping {skipped} already downloaded files", file=sys.stderr)
    print(f"Remaining files to download: {len(remaining)}", file=sys.stderr)

    if len(remaining) == 0:
        print("Nothing to download!", file=sys.stderr)
        return

    batches = split_into_batches(
        remaining,
        batch_size=args.batch_size,
        num_batches=args.batches
    )
    num_batches = len(batches)

    print(f"\nBatch configuration:", file=sys.stderr)
    print(f"  Total files: {len(remaining)}", file=sys.stderr)
    print(f"  Number of batches: {num_batches}", file=sys.stderr)
    print(f"  Files per batch: ~{len(remaining) // num_batches}", file=sys.stderr)
    print(f"  CPUs per job: {args.cpus}", file=sys.stderr)

    if args.dry_run:
        print("\n[DRY RUN] Would create:", file=sys.stderr)
        print(f"  {slurm_dir}/batches/batch_001.txt ... batch_{num_batches:03d}.txt", file=sys.stderr)
        print(f"  {slurm_dir}/jobs/download_batch_001.sbatch ... download_batch_{num_batches:03d}.sbatch", file=sys.stderr)
        print(f"  {slurm_dir}/submit_all.sh", file=sys.stderr)
        return

    batches_dir = os.path.join(slurm_dir, "batches")
    print(f"\nWriting batch manifests to {batches_dir}/...", file=sys.stderr)
    write_batch_manifests(batches, batches_dir)

    print(f"Writing SLURM job scripts to {slurm_dir}/jobs/...", file=sys.stderr)
    write_slurm_jobs(
        num_batches,
        slurm_dir,
        work_dir,
        output_dir,
        args.cpus,
        args.conda_env
    )

    write_submit_script(num_batches, slurm_dir)

    print(f"\nCreated:", file=sys.stderr)
    print(f"  {num_batches} batch manifests in {batches_dir}/", file=sys.stderr)
    print(f"  {num_batches} SLURM job scripts in {slurm_dir}/jobs/", file=sys.stderr)
    print(f"  {slurm_dir}/submit_all.sh", file=sys.stderr)

    print(f"\nTo submit all jobs:", file=sys.stderr)
    print(f"  {slurm_dir}/submit_all.sh", file=sys.stderr)


if __name__ == '__main__':
    main()
