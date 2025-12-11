import os
import sys
import argparse
import subprocess
import concurrent.futures


def load_batch_manifest(batch_file: str) -> list[tuple[str, str]]:
    items = []
    with open(batch_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
    return items


def download_one(
    gcs_path: str,
    local_path: str,
    output_dir: str
) -> tuple[str, bool, str]:
    dest_path = os.path.join(output_dir, local_path)
    dest_dir = os.path.dirname(dest_path)

    try:
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(dest_path):
            return gcs_path, True, "already exists"

        result = subprocess.run(
            ['gsutil', 'cp', gcs_path, dest_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return gcs_path, False, result.stderr.strip()
        return gcs_path, True, ""
    except Exception as e:
        return gcs_path, False, str(e)


def download_batch(
    items: list[tuple[str, str]],
    output_dir: str,
    parallel: int
) -> tuple[int, int]:
    total = len(items)
    completed = 0
    failed = 0
    skipped = 0

    print(f"Downloading {total} files with {parallel} parallel workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(download_one, gcs_path, local_path, output_dir): (gcs_path, local_path)
            for gcs_path, local_path in items
        }

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            gcs_path, success, message = future.result()

            if not success:
                failed += 1
                print(f"FAILED: {gcs_path}: {message}", file=sys.stderr)
            elif message == "already exists":
                skipped += 1

            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({failed} failed, {skipped} skipped)")

    succeeded = completed - failed
    print(f"\nBatch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    return succeeded, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download a specific batch of arXiv PDFs"
    )
    parser.add_argument(
        'batch_idx',
        type=int,
        help='Batch index (1-based)'
    )
    parser.add_argument(
        '--slurm-dir',
        default='./slurm_jobs',
        help='Directory containing batch manifests (default: ./slurm_jobs)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for downloaded PDFs'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=8,
        help='Number of parallel downloads (default: 8)'
    )

    args = parser.parse_args()

    batch_file = os.path.join(
        args.slurm_dir,
        "batches",
        f"batch_{args.batch_idx:03d}.txt"
    )

    if not os.path.exists(batch_file):
        print(f"ERROR: Batch file not found: {batch_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading batch {args.batch_idx} from {batch_file}...")
    items = load_batch_manifest(batch_file)
    print(f"Batch contains {len(items)} files")

    if len(items) == 0:
        print("Nothing to download")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    succeeded, failed = download_batch(items, args.output_dir, args.parallel)

    if failed > 0:
        print(f"\nWARNING: {failed} files failed to download", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
