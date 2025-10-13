import os
import re
import sys
import json
import gzip
import argparse
import subprocess
from tqdm import tqdm
from typing import Tuple, Dict, List, NamedTuple

MANIFEST_FILE = 'arxiv_pdf_manifest.json'
MANIFEST_URL = 'gs://arxiv-dataset/arxiv-dataset_list-of-files.txt.gz'
MANIFEST_GZ = 'arxiv-dataset_list-of-files.txt.gz'

MODERN_ID_RE = re.compile(r'^\d{4}\.\d{4,5}$')
DIGIT_START_RE = re.compile(r'^\d')
VERSION_SUFFIX_RE = re.compile(r'v(\d+)\.pdf$')
VERSION_STRIP_RE = re.compile(r'v\d+\.pdf$')
PDF_SUFFIX_RE = re.compile(r'(v\d+)?\.pdf$')
WILDCARD_CHUNK_SIZE = 64


class ResolutionResult(NamedTuple):
    urls: List[str]
    id_to_url: Dict[str, str]
    not_found_ids: List[str]


def write_json_output(path: str, entries: List[dict], description: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} {description} entries to '{path}'.")


def extract_version(url: str) -> int:
    match = VERSION_SUFFIX_RE.search(url)
    if match:
        return int(match.group(1))
    return 0


def normalize_base_url(url: str) -> str:
    return PDF_SUFFIX_RE.sub('', url)


def detect_arxiv_format(arxiv_id: str) -> Tuple[str, str]:
    if MODERN_ID_RE.match(arxiv_id):
        return 'modern', arxiv_id

    if '/' in arxiv_id:
        return 'legacy', arxiv_id

    if DIGIT_START_RE.match(arxiv_id):
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

    base_urls = sorted({VERSION_STRIP_RE.sub('', url) for url in urls_to_verify})
    results: Dict[str, List[str]] = {base_url: [] for base_url in base_urls}
    dir_to_basenames: Dict[str, List[str]] = {}

    for base_url in base_urls:
        dir_part, _, base_name = base_url.rpartition('/')
        if not dir_part or not base_name:
            continue
        dir_to_basenames.setdefault(dir_part, []).append(base_name)

    total_patterns = sum(len(set(names)) for names in dir_to_basenames.values())
    print(f"Verifying {total_patterns} URLs with gsutil...")

    for dir_part, base_names in dir_to_basenames.items():
        unique_names = sorted(set(base_names))
        for start in range(0, len(unique_names), WILDCARD_CHUNK_SIZE):
            chunk = unique_names[start:start + WILDCARD_CHUNK_SIZE]
            patterns = [f"{dir_part}/{name}*.pdf" for name in chunk]
            command = ['gsutil', '-m', 'ls', *patterns]
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
                    base_url = normalize_base_url(line)
                    if base_url in results:
                        results[base_url].append(line)

        for line in stderr_lines:
            if 'No URLs matched' in line:
                match = re.search(r'No URLs matched: (.+)', line)
                if match:
                    failed_pattern = match.group(1).strip()
                    base_url = normalize_base_url(failed_pattern)
                    results.setdefault(base_url, [])

            if completed.returncode not in (0, 1):
                print(f"gsutil ls returned {completed.returncode} for directory '{dir_part}' chunk starting at index {start}.", file=sys.stderr)

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


def url_generator(arxiv_ids: List[str], manifest: dict) -> ResolutionResult:
    resolved_urls: Dict[str, str] = {}
    pending_verification: Dict[str, str] = {}
    not_found_ids: List[str] = []
    not_found_seen: set[str] = set()

    print(f"Generating URLs for {len(arxiv_ids)} papers...")

    def mark_not_found(arxiv_id: str):
        if arxiv_id not in not_found_seen:
            not_found_seen.add(arxiv_id)
            not_found_ids.append(arxiv_id)

    for arxiv_id in tqdm(arxiv_ids, desc="Processing IDs"):
        if arxiv_id in manifest:
            available_versions = manifest[arxiv_id]
            if available_versions:
                latest_version_url = max(available_versions, key=extract_version)
                resolved_urls[arxiv_id] = latest_version_url
                continue

        constructed_url = construct_pdf_url(arxiv_id, version=1)
        if constructed_url:
            pending_verification[arxiv_id] = constructed_url
        else:
            print(f"Warning: Could not construct URL for '{arxiv_id}'.", file=sys.stderr)
            mark_not_found(arxiv_id)

    verification_not_found_ids: List[str] = []

    if pending_verification:
        print(f"\n{len(pending_verification)} papers not in manifest. Verifying availability...")
        verification_results = batch_verify_urls(list(pending_verification.values()))

        found_count = 0

        for arxiv_id, tentative_url in pending_verification.items():
            base_url = normalize_base_url(tentative_url)
            available_versions = verification_results.get(base_url, [])

            if available_versions:
                latest_version = max(available_versions, key=extract_version)
                resolved_urls[arxiv_id] = latest_version
                found_count += 1
            else:
                mark_not_found(arxiv_id)
                verification_not_found_ids.append(arxiv_id)

        print(f"Verification complete: {found_count} found, {len(verification_not_found_ids)} not found.")

        if verification_not_found_ids:
            print("\nPapers not found in dataset:")
            for missing_id in verification_not_found_ids[:10]:
                print(f"  - {missing_id}")
            if len(verification_not_found_ids) > 10:
                remainder = len(verification_not_found_ids) - 10
                print(f"  ... and {remainder} more")

    urls = list(resolved_urls.values())

    print(f"\nTotal URLs ready for download: {len(urls)}")
    return ResolutionResult(urls=urls, id_to_url=resolved_urls, not_found_ids=not_found_ids)


def generate_gcs_urls(json_path: str, manifest: dict) -> ResolutionResult:
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
        return ResolutionResult(urls=[], id_to_url={}, not_found_ids=[])

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
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Resolve URLs without downloading. Useful for testing manifest and lookup logic.'
    )
    parser.add_argument(
        '--write-resolved-json',
        help='Optional path to write all resolved arXiv IDs (includes gcs_url field).'
    )
    parser.add_argument(
        '--write-not-found-json',
        help='Optional path to write IDs that were missing from manifest and unavailable via gsutil.'
    )
    parser.add_argument(
        '--write-pending-json',
        help='Optional path to write resolved IDs that still need downloading after skipping existing files.'
    )
    args = parser.parse_args()

    manifest = build_or_load_manifest(args.force_rebuild_manifest)
    resolution = generate_gcs_urls(args.input_json, manifest)

    if args.write_resolved_json:
        resolved_entries = [
            {'arxiv_id': arxiv_id, 'gcs_url': url}
            for arxiv_id, url in resolution.id_to_url.items()
        ]
        write_json_output(args.write_resolved_json, resolved_entries, 'resolved')

    if args.write_not_found_json and resolution.not_found_ids:
        not_found_entries = [{'arxiv_id': arxiv_id}
                             for arxiv_id in resolution.not_found_ids]
        write_json_output(args.write_not_found_json,
                          not_found_entries, 'not-found')
    elif args.write_not_found_json:
        write_json_output(args.write_not_found_json, [], 'not-found')

    os.makedirs(args.output_dir, exist_ok=True)
    urls_to_download = filter_already_downloaded(resolution.urls, args.output_dir)

    if args.write_pending_json:
        pending_set = set(urls_to_download)
        pending_entries = [
            {'arxiv_id': arxiv_id, 'gcs_url': url}
            for arxiv_id, url in resolution.id_to_url.items()
            if url in pending_set
        ]
        write_json_output(args.write_pending_json, pending_entries, 'pending')

    if args.dry_run:
        print("\nDry run mode enabled. Skipping download.")
        print(f"Resolved {len(urls_to_download)} URLs that are not already present in '{args.output_dir}'.")
        if urls_to_download:
            preview_count = min(10, len(urls_to_download))
            print(f"Previewing first {preview_count} URLs:")
            for url in urls_to_download[:preview_count]:
                print(f"  {url}")
        return

    download_pdfs_with_gsutil(urls_to_download, args.output_dir)


if __name__ == '__main__':
    main()
