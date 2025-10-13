import os
import re
import sys
import json
import gzip
import argparse
import subprocess
from tqdm import tqdm
from typing import Tuple, Dict, List

MANIFEST_FILE = 'arxiv_pdf_manifest.json'
MANIFEST_URL = 'gs://arxiv-dataset/arxiv-dataset_list-of-files.txt.gz'
MANIFEST_GZ = 'arxiv-dataset_list-of-files.txt.gz'


def detect_arxiv_format(arxiv_id: str) -> Tuple[str, str]:
    if re.match(r'^\d{4}\.\d{4,5}$', arxiv_id):
        return 'modern', arxiv_id

    if '/' in arxiv_id:
        return 'legacy', arxiv_id

    if re.match(r'^\d', arxiv_id):
        return 'modern', arxiv_id

    return 'unknown', arxiv_id


def construct_pdf_url(arxiv_id: str, version: int = 1) -> str:
    format_type, normalized_id = detect_arxiv_format(arxiv_id)

    if format_type == 'modern':
        yymm = normalized_id[:4]
        return f"gs://arxiv-dataset/arxiv/arxiv/pdf/{yymm}/{normalized_id}v{version}.pdf"

    if format_type == 'legacy':
        category, number = normalized_id.split('/', 1)
        yymm = number[:4]
        return f"gs://arxiv-dataset/arxiv/{category}/pdf/{yymm}/{number}v{version}.pdf"

    return ""


def batch_verify_urls(urls_to_verify: List[str]) -> Dict[str, List[str]]:
    if not urls_to_verify:
        return {}

    results: Dict[str, List[str]] = {}
    wildcard_patterns = [
        re.sub(r'v\d+\.pdf$', 'v*.pdf', url) for url in urls_to_verify
    ]

    print(f"Verifying {len(wildcard_patterns)} URLs with gsutil...")
    chunk_size = 64

    for start in range(0, len(wildcard_patterns), chunk_size):
        chunk = wildcard_patterns[start:start + chunk_size]
        command = ['gsutil', '-m', 'ls'] + chunk
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True
            )
        except FileNotFoundError:
            print("\nError: 'gsutil' command not found.", file=sys.stderr)
            print("Please ensure the Google Cloud SDK is installed and 'gsutil' is in your system's PATH.", file=sys.stderr)
            sys.exit(1)

        stdout_lines = completed.stdout.splitlines()
        stderr_lines = completed.stderr.splitlines()

        for line in stdout_lines:
            line = line.strip()
            if line.startswith('gs://') and line.endswith('.pdf'):
                base_url = re.sub(r'v\d+\.pdf$', '', line)
                results.setdefault(base_url, []).append(line)

        for line in stderr_lines:
            if 'No URLs matched' in line:
                match = re.search(r'No URLs matched: (.+)', line)
                if match:
                    failed_pattern = match.group(1).strip()
                    base_url = re.sub(r'v\*\.pdf$', '', failed_pattern)
                    results.setdefault(base_url, [])

        if completed.returncode not in (0, 1):
            print(f"gsutil ls returned {completed.returncode} for chunk starting at index {start}.", file=sys.stderr)

    return results


def build_or_load_manifest(force_rebuild: bool) -> dict:
    if not force_rebuild and os.path.exists(MANIFEST_FILE):
        print(f"Loading cached manifest from '{MANIFEST_FILE}'...")
        with open(MANIFEST_FILE, 'r') as f:
            return json.load(f)

    print(f"Building new manifest. This may take a few minutes.")
    print(f"Downloading master file list from {MANIFEST_URL}...")
    try:
        subprocess.run(['gsutil', 'cp', MANIFEST_URL, '.'],
                       check=True, capture_output=True)
    except FileNotFoundError:
        print("\nError: 'gsutil' command not found.", file=sys.stderr)
        print("Please ensure the Google Cloud SDK is installed and 'gsutil' is in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading manifest with gsutil: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    manifest = {}
    print(f"Processing '{MANIFEST_GZ}' to build manifest...")
    with gzip.open(MANIFEST_GZ, 'rt') as f:
        for line in tqdm(f, desc="Parsing file list"):
            if '/pdf/' not in line or not line.endswith('.pdf\n'):
                continue
            clean_line = line.strip()
            full_gcs_url = clean_line
            parts = clean_line.split('/')
            if len(parts) < 6:
                continue
            filename = parts[-1]
            base_id_with_version = os.path.splitext(filename)[0]
            base_id = re.sub(r'v\d+$', '', base_id_with_version)
            category_part = parts[4]
            if category_part == 'arxiv':
                arxiv_id = base_id
            else:
                category = category_part
                arxiv_id = f"{category}/{base_id}"

            if arxiv_id not in manifest:
                manifest[arxiv_id] = []
            manifest[arxiv_id].append(full_gcs_url)

    print(f"Saving manifest to '{MANIFEST_FILE}'...")
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

    os.remove(MANIFEST_GZ)
    print("Manifest build complete.")
    return manifest


def url_generator(arxiv_ids: List[str], manifest: dict) -> List[str]:
    urls: List[str] = []
    urls_to_verify: List[str] = []
    id_to_tentative_url: Dict[str, str] = {}

    print(f"Generating URLs for {len(arxiv_ids)} papers...")

    for arxiv_id in tqdm(arxiv_ids, desc="Processing IDs"):
        if arxiv_id in manifest:
            available_versions = manifest[arxiv_id]
            if available_versions:
                latest_version_url = sorted(
                    available_versions,
                    key=lambda url: int(re.search(r'v(\d+)\.pdf$', url).group(1))
                )[-1]
                urls.append(latest_version_url)
                continue

        constructed_url = construct_pdf_url(arxiv_id, version=1)
        if constructed_url:
            urls_to_verify.append(constructed_url)
            id_to_tentative_url[arxiv_id] = constructed_url
        else:
            print(f"Warning: Could not construct URL for '{arxiv_id}'.", file=sys.stderr)

    if urls_to_verify:
        print(f"\n{len(urls_to_verify)} papers not in manifest. Verifying availability...")
        verification_results = batch_verify_urls(urls_to_verify)

        found_count = 0
        not_found_ids: List[str] = []

        for arxiv_id, tentative_url in id_to_tentative_url.items():
            base_url = re.sub(r'v\d+\.pdf$', '', tentative_url)
            available_versions = verification_results.get(base_url, [])

            if available_versions:
                latest_version = sorted(
                    available_versions,
                    key=lambda url: int(re.search(r'v(\d+)\.pdf$', url).group(1))
                )[-1]
                urls.append(latest_version)
                found_count += 1
            else:
                not_found_ids.append(arxiv_id)

        print(f"Verification complete: {found_count} found, {len(not_found_ids)} not found.")

        if not_found_ids:
            print("\nPapers not found in dataset:")
            for missing_id in not_found_ids[:10]:
                print(f"  - {missing_id}")
            if len(not_found_ids) > 10:
                remainder = len(not_found_ids) - 10
                print(f"  ... and {remainder} more")

    print(f"\nTotal URLs ready for download: {len(urls)}")
    return urls


def generate_gcs_urls(json_path: str, manifest: dict) -> list[str]:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: JSON file '{json_path}' must contain a list of objects.", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{json_path}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{json_path}'. Check for syntax errors.", file=sys.stderr)
        sys.exit(1)

    arxiv_ids: List[str] = []
    for item in tqdm(data, desc="Collecting arXiv IDs"):
        arxiv_id = item.get('arxiv_id')
        if arxiv_id:
            arxiv_ids.append(arxiv_id)

    if not arxiv_ids:
        print("No arxiv_id fields found in input JSON.", file=sys.stderr)
        return []

    return url_generator(arxiv_ids, manifest)


def download_pdfs_with_gsutil(urls: list[str], output_dir: str):
    if not urls:
        print("No valid URLs were generated. Nothing to download.")
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nAttempting to download {len(urls)} PDFs to '{output_dir}' using gsutil...")
    print("gsutil will display its own progress and error messages below.")
    urls_stdin = "\n".join(urls)
    command = [
        'gsutil',
        '-m',
        'cp',
        '-n',
        '-I',
        output_dir
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
        process.communicate(input=urls_stdin)

        if process.returncode == 0:
            print("\nDownload command completed successfully.")
        else:
            print(f"\ngsutil command finished with errors (exit code {process.returncode}).", file=sys.stderr)
            print(
                "Please check the gsutil output above for specific file errors.", file=sys.stderr)
    except FileNotFoundError:
        print("\nError: 'gsutil' command not found.", file=sys.stderr)
        print("Please ensure the Google Cloud SDK is installed and 'gsutil' is in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def filter_already_downloaded(urls: list[str], output_dir: str) -> list[str]:
    # Look at filenames that would be created by gsutil (the URL basename)
    existing = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    missing = []
    skipped = 0
    for url in urls:
        fname = os.path.basename(url)
        if fname in existing:
            skipped += 1
            continue
        missing.append(url)
    if skipped:
        print(f"Skipping {skipped} PDFs already present in '{output_dir}'.")
    return missing



def main():
    parser = argparse.ArgumentParser(
        description="""
        Script to download arXiv PDFs from the Kaggle dataset of arXiv PDFs.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input-json', required=True,
        help='Path to the input JSON file. Must be a list of objects, each with an "arxiv_id" key.'
    )
    parser.add_argument(
        '-o', '--output-dir', default="output",
        help='Path to the directory where PDFs will be saved.'
    )
    parser.add_argument(
        '-f', '--force-rebuild-manifest', action='store_true',
        help='Force the download and recreation of the manifest file, even if a cached one exists.'
    )
    args = parser.parse_args()

    manifest = build_or_load_manifest(args.force_rebuild_manifest)
    urls_to_download = generate_gcs_urls(args.input_json, manifest)

    os.makedirs(args.output_dir, exist_ok=True)
    urls_to_download = filter_already_downloaded(urls_to_download, args.output_dir)

    download_pdfs_with_gsutil(urls_to_download, args.output_dir)


if __name__ == '__main__':
    main()
