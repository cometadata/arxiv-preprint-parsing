import os
import re
import sys
import json
import gzip
import argparse
import subprocess
from tqdm import tqdm

MANIFEST_FILE = 'arxiv_pdf_manifest.json'
MANIFEST_URL = 'gs://arxiv-dataset/arxiv-dataset_list-of-files.txt.gz'
MANIFEST_GZ = 'arxiv-dataset_list-of-files.txt.gz'


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

    urls = []
    found_count = 0
    for item in tqdm(data, desc="Generating URLs from manifest"):
        arxiv_id = item.get('arxiv_id')
        if not arxiv_id:
            continue

        if arxiv_id in manifest:
            available_versions = manifest[arxiv_id]
            if not available_versions:
                continue
            latest_version_url = sorted(
                available_versions,
                key=lambda url: int(re.search(r'v(\d+)\.pdf$', url).group(1))
            )[-1]
            urls.append(latest_version_url)
            found_count += 1
        else:
            print(f"Warning: arXiv ID '{arxiv_id}' not found in manifest. Skipping.", file=sys.stderr)

    print(f"\nFound {found_count} matching PDFs in manifest out of {len(data)} total entries.")
    return urls


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
    download_pdfs_with_gsutil(urls_to_download, args.output_dir)


if __name__ == '__main__':
    main()
