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


def download_files(
    latest_versions: dict[str, tuple[str, int, str]],
    output_dir: str,
    parallel: int = 8
) -> None:
    items = list(latest_versions.items())
    total = len(items)

    def download_one(item: tuple[str, tuple[str, int, str]]) -> tuple[str, bool, str]:
        paper_id, (gcs_path, version, local_path) = item
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
                return paper_id, False, result.stderr
            return paper_id, True, ""
        except Exception as e:
            return paper_id, False, str(e)

    completed = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(download_one, item): item for item in items}

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            paper_id, success, error = future.result()

            if not success:
                failed += 1
                print(f"  Failed: {paper_id}: {error}", file=sys.stderr)

            if completed % 100 == 0 or completed == total:
                print(f"  Downloaded {completed}/{total} ({failed} failed)...", file=sys.stderr)

    print(f"Download complete: {completed - failed} succeeded, {failed} failed", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest of latest arXiv PDFs or download them"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-m', '--manifest',
        metavar='FILE',
        help='Generate TSV manifest file (use - for stdout)'
    )
    group.add_argument(
        '-d', '--download',
        metavar='DIR',
        help='Download PDFs to specified directory'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=8,
        help='Number of parallel downloads (default: 8)'
    )
    args = parser.parse_args()

    print("Discovering PDF directories...", file=sys.stderr)
    directories = list_pdf_directories()
    print(f"Found {len(directories)} directories to scan", file=sys.stderr)

    print("Enumerating PDFs (this may take 10-30 minutes)...", file=sys.stderr)
    pdf_paths = enumerate_all_pdfs(directories)

    print("Finding latest versions...", file=sys.stderr)
    latest_versions = find_latest_versions(pdf_paths)

    print(f"Found {len(latest_versions)} unique PDFs (latest versions only)", file=sys.stderr)

    if args.download:
        print(f"Downloading to {args.download}...", file=sys.stderr)
        os.makedirs(args.download, exist_ok=True)
        download_files(latest_versions, args.download, args.parallel)
    else:
        output = sys.stdout if args.manifest == '-' else open(args.manifest, 'w')
        try:
            for paper_id in sorted(latest_versions.keys()):
                gcs_path, version, local_path = latest_versions[paper_id]
                output.write(f"{gcs_path}\t{local_path}\n")
        finally:
            if output is not sys.stdout:
                output.close()


if __name__ == '__main__':
    main()
