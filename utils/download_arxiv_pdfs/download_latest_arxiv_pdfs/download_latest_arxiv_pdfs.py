import os
import re
import sys
import subprocess
import argparse
import concurrent.futures
from typing import Iterator

VERSION_PATTERN = re.compile(r'v(\d+)\.pdf$')
NEW_FORMAT_PATTERN = re.compile(r'/(\d{4}\.\d{4,5})v\d+\.pdf$')
LEGACY_FORMAT_PATTERN = re.compile(
    r'/arxiv/([^/]+)/pdf/\d{4}/(\d{7})v\d+\.pdf$')
BUCKET = 'gs://arxiv-dataset/arxiv'
MAX_WORKERS = 16


def extract_paper_info(gcs_path: str) -> tuple[str, int, str] | None:
    if not gcs_path.endswith('.pdf'):
        return None

    version_match = VERSION_PATTERN.search(gcs_path)
    if not version_match:
        return None

    version = int(version_match.group(1))
    filename = gcs_path.split('/')[-1]

    new_match = NEW_FORMAT_PATTERN.search(gcs_path)
    if new_match:
        paper_id = new_match.group(1)
        local_path = filename
        return paper_id, version, local_path

    legacy_match = LEGACY_FORMAT_PATTERN.search(gcs_path)
    if legacy_match:
        subject = legacy_match.group(1)
        paper_num = legacy_match.group(2)
        paper_id = f"{subject}/{paper_num}"
        local_path = f"{subject}_{filename}"
        return paper_id, version, local_path

    return None


def list_pdf_directories() -> list[str]:
    directories = []

    result = subprocess.run(
        ['gsutil', 'ls', f'{BUCKET}/'],
        capture_output=True,
        text=True,
        check=True
    )
    categories = [line.strip()
                  for line in result.stdout.strip().split('\n') if line.strip()]

    for category in categories:
        category_name = category.rstrip('/').split('/')[-1]

        if category_name == 'pdf':
            result = subprocess.run(
                ['gsutil', 'ls', category],
                capture_output=True,
                text=True,
                check=True
            )
            directories.extend([
                d.strip() for d in result.stdout.strip().split('\n')
                if d.strip() and d.strip().endswith('/')
            ])
        elif category_name == 'arxiv':
            pdf_path = category + 'pdf/'
            result = subprocess.run(
                ['gsutil', 'ls', pdf_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                directories.extend([
                    d.strip() for d in result.stdout.strip().split('\n')
                    if d.strip() and d.strip().endswith('/')
                ])
        else:
            pdf_path = category + 'pdf/'
            result = subprocess.run(
                ['gsutil', 'ls', pdf_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                directories.extend([
                    d.strip() for d in result.stdout.strip().split('\n')
                    if d.strip() and d.strip().endswith('/')
                ])

    return directories


def list_pdfs_in_directory(directory: str) -> list[str]:
    result = subprocess.run(
        ['gsutil', 'ls', directory],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return []

    return [
        line.strip() for line in result.stdout.strip().split('\n')
        if line.strip() and line.strip().endswith('.pdf')
    ]


def enumerate_all_pdfs(directories: list[str]) -> Iterator[str]:
    total_dirs = len(directories)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(
            list_pdfs_in_directory, d): d for d in directories}

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            directory = futures[future]
            if completed % 50 == 0 or completed == total_dirs:
                print(f"  Processed {completed}/{total_dirs} directories...", file=sys.stderr)

            try:
                pdfs = future.result()
                yield from pdfs
            except Exception as e:
                print(f"  Error listing {directory}: {e}", file=sys.stderr)


def find_latest_versions(pdf_paths: Iterator[str]) -> dict[str, tuple[str, int, str]]:
    latest_versions: dict[str, tuple[str, int, str]] = {}

    for path in pdf_paths:
        result = extract_paper_info(path)
        if result is None:
            continue

        paper_id, version, local_path = result

        if paper_id not in latest_versions or version > latest_versions[paper_id][1]:
            latest_versions[paper_id] = (path, version, local_path)

    return latest_versions


def load_manifest(manifest_path: str) -> dict[str, tuple[str, str]]:
    manifest = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                gcs_path, local_path = parts
                manifest[gcs_path] = local_path
    return manifest


def download_from_manifest(
    manifest: dict[str, str],
    output_dir: str,
    parallel: int = 8,
    resume: bool = False,
    batch_size: int | None = None
) -> None:

    items = list(manifest.items())

    if resume:
        items = [
            (gcs_path, local_path) for gcs_path, local_path in items
            if not os.path.exists(os.path.join(output_dir, local_path))
        ]
        skipped = len(manifest) - len(items)
        if skipped > 0:
            print(f"Skipping {skipped} already downloaded", file=sys.stderr)

    if batch_size and len(items) > batch_size:
        print(f"Batch: downloading {batch_size} of {len(items)} remaining", file=sys.stderr)
        items = items[:batch_size]

    total = len(items)
    if total == 0:
        print("Nothing to download", file=sys.stderr)
        return

    def download_one(item: tuple[str, str]) -> tuple[str, bool, str]:
        gcs_path, local_path = item
        dest_path = os.path.join(output_dir, local_path)
        dest_dir = os.path.dirname(dest_path)

        try:
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            result = subprocess.run(
                ['gsutil', 'cp', gcs_path, dest_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return gcs_path, False, result.stderr
            return gcs_path, True, ""
        except Exception as e:
            return gcs_path, False, str(e)

    completed = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(download_one, item): item for item in items}

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            gcs_path, success, error = future.result()

            if not success:
                failed += 1
                print(f"Failed: {gcs_path}: {error}", file=sys.stderr)

            if completed % 100 == 0 or completed == total:
                print(f"Downloaded {completed}/{total} ({failed} failed)...", file=sys.stderr)

    print(f"Download complete: {completed - failed} succeeded, {failed} failed", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest of latest arXiv PDFs or download them"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    manifest_parser = subparsers.add_parser(
        'manifest',
        help='Generate TSV manifest file (discovers all PDFs from server)'
    )
    manifest_parser.add_argument(
        'output',
        metavar='FILE',
        help='Output manifest file (use - for stdout)'
    )

    download_parser = subparsers.add_parser(
        'download',
        help='Download PDFs using a manifest file'
    )
    download_parser.add_argument(
        'manifest',
        metavar='MANIFEST',
        help='Input manifest file (TSV: gcs_path, local_path)'
    )
    download_parser.add_argument(
        'output_dir',
        metavar='DIR',
        help='Output directory for downloaded PDFs'
    )
    download_parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=8,
        help='Number of parallel downloads (default: 8)'
    )
    download_parser.add_argument(
        '-r', '--resume',
        action='store_true',
        help='Skip files that already exist in output directory'
    )
    download_parser.add_argument(
        '-b', '--batch',
        type=int,
        metavar='N',
        help='Download only N files per run (use with --resume for incremental downloads)'
    )

    args = parser.parse_args()

    if args.command == 'manifest':
        print("Discovering PDF directories...", file=sys.stderr)
        directories = list_pdf_directories()
        print(f"Found {len(directories)} directories to scan", file=sys.stderr)

        print("Enumerating PDFs (this may take 10-30 minutes)...", file=sys.stderr)
        pdf_paths = enumerate_all_pdfs(directories)

        print("Finding latest versions...", file=sys.stderr)
        latest_versions = find_latest_versions(pdf_paths)

        print(f"Found {len(latest_versions)} unique PDFs (latest versions only)", file=sys.stderr)

        output = sys.stdout if args.output == '-' else open(args.output, 'w')
        try:
            for paper_id in sorted(latest_versions.keys()):
                gcs_path, version, local_path = latest_versions[paper_id]
                output.write(f"{gcs_path}\t{local_path}\n")
        finally:
            if output is not sys.stdout:
                output.close()

        if args.output != '-':
            print(f"Manifest written to {args.output}", file=sys.stderr)

    elif args.command == 'download':
        print(f"Loading manifest from {args.manifest}...", file=sys.stderr)
        manifest = load_manifest(args.manifest)
        print(f"Manifest contains {len(manifest)} files", file=sys.stderr)

        os.makedirs(args.output_dir, exist_ok=True)
        download_from_manifest(
            manifest,
            args.output_dir,
            args.parallel,
            args.resume,
            args.batch
        )


if __name__ == '__main__':
    main()
