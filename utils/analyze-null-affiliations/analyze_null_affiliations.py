#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
# ]
# ///
"""Sample COMET arxiv author affiliation dataset and analyze null affiliation rates."""

import csv
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from typing import Literal


DEFAULT_REPO = "cometadata/arxiv-author-affiliations-matched-ror-ids"
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_DATA_DIR = Path("data")
DEFAULT_SAMPLES_DIR = Path("samples")

AffiliationStatus = Literal["all_null", "some_null", "none_null", "no_authors"]


def _repo_to_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _is_author_affiliation_null(author: dict) -> bool:
    affiliations = author.get("affiliations")
    if affiliations is None or affiliations == []:
        return True
    return all(
        aff is None or aff.get("affiliation") is None or aff.get("affiliation") == ""
        for aff in affiliations
    )


def _count_valid_affiliations(author: dict) -> int:
    affiliations = author.get("affiliations")
    if affiliations is None:
        return 0
    return sum(
        1 for aff in affiliations
        if aff is not None and aff.get("affiliation") not in (None, "")
    )


def _find_jsonl_file(directory: Path) -> Path:
    jsonl_files = list(directory.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in: {directory}")
    if len(jsonl_files) > 1:
        # Prefer data.jsonl, as named in the HuggingFace repo, if this exists
        data_file = directory / "data.jsonl"
        if data_file in jsonl_files:
            return data_file
        print(f"Warning: Multiple JSONL files found, using: {jsonl_files[0]}", file=sys.stderr)
    return jsonl_files[0]


def download_dataset(repo_id: str, data_dir: Path, force: bool = False) -> Path:
    output_dir = data_dir / _repo_to_dirname(repo_id)

    if not force and output_dir.exists():
        try:
            existing_file = _find_jsonl_file(output_dir)
            print(f"Data already exists: {existing_file}")
            print("Use --force to re-download.")
            return existing_file
        except FileNotFoundError:
            pass 

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {repo_id}")
    print(f"Destination: {output_dir}")

    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(output_dir),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_output = f"{e.stdout or ''}{e.stderr or ''}"

        if "No space left on device" in error_output or "os error 28" in error_output:
            print("Download failed: Not enough disk space.", file=sys.stderr)
            print(f"  Free up space or use a different --data-dir location.", file=sys.stderr)
        elif "rate limit" in error_output.lower() or "429" in error_output:
            print("Download failed: Rate limited by HuggingFace.", file=sys.stderr)
            print("  Wait a while before retrying, or use --skip-download with existing data.", file=sys.stderr)
        elif "401" in error_output or "unauthorized" in error_output.lower():
            print("Download failed: Authentication required.", file=sys.stderr)
            print("  Run 'huggingface-cli login' to authenticate.", file=sys.stderr)
        elif "404" in error_output or "not found" in error_output.lower():
            print(f"Download failed: Dataset '{repo_id}' not found.", file=sys.stderr)
        else:
            print(f"Download failed: {error_output or 'Unknown error'}", file=sys.stderr)

        # Check if partial data exists that could be used
        try:
            partial_file = _find_jsonl_file(output_dir)
            print(f"\nPartial data may exist: {partial_file}", file=sys.stderr)
            print("You can try: --skip-download to use existing data", file=sys.stderr)
        except FileNotFoundError:
            pass

        raise

    data_file = _find_jsonl_file(output_dir)
    print(f"Downloaded: {data_file}")
    return data_file


def create_sample(
    input_file: Path,
    output_file: Path,
    sample_size: int,
    seed: int | None = None,
) -> int:
    """Single-pass reservoir sampling."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if seed is not None:
        random.seed(seed)

    print(f"Sampling from {input_file} (reservoir sampling)...")
    reservoir: list[str] = []
    total_lines = 0

    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            total_lines = i + 1
            if i < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line
            if total_lines % 1_000_000 == 0:
                print(f"  Processed {total_lines:,} lines...")

    print(f"Total lines: {total_lines:,}")
    actual_sample_size = len(reservoir)
    if actual_sample_size < sample_size:
        print(f"Warning: Only {actual_sample_size:,} lines available (requested {sample_size:,})")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing sample to {output_file}...")

    with open(output_file, "w") as f:
        f.writelines(reservoir)

    print(f"Extracted {actual_sample_size:,} lines")
    return actual_sample_size


def _check_affiliations(record: dict) -> AffiliationStatus:
    authors = record.get("prediction", [])
    if not authors:
        return "no_authors"

    null_count = sum(1 for author in authors if _is_author_affiliation_null(author))

    if null_count == len(authors):
        return "all_null"
    elif null_count > 0:
        return "some_null"
    else:
        return "none_null"


def analyze_sample(sample_file: Path, output_dir: Path) -> dict:
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")

    print(f"Analyzing: {sample_file}")

    counts: dict[str, int] = {"all_null": 0, "some_null": 0, "none_null": 0, "no_authors": 0}
    author_stats = {
        "total_authors": 0,
        "authors_with_affiliations": 0,
        "authors_without_affiliations": 0,
        "total_affiliations": 0,
        "valid_affiliations": 0,
        "null_affiliations": 0,
    }
    parse_errors = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    works_all_missing_file = output_dir / "works_all_missing.jsonl"
    works_some_missing_file = output_dir / "works_some_missing.jsonl"
    works_complete_file = output_dir / "works_complete.jsonl"
    works_no_authors_file = output_dir / "works_no_authors.jsonl"

    with (
        open(sample_file, "r") as infile,
        open(works_all_missing_file, "w") as works_all_missing_out,
        open(works_some_missing_file, "w") as works_some_missing_out,
        open(works_complete_file, "w") as works_complete_out,
        open(works_no_authors_file, "w") as works_no_authors_out,
    ):
        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                parse_errors += 1
                continue

            status = _check_affiliations(record)
            counts[status] += 1

            for author in record.get("prediction", []):
                author_stats["total_authors"] += 1
                affiliations = author.get("affiliations") or []
                num_total = len(affiliations)
                num_valid = _count_valid_affiliations(author)
                num_null = num_total - num_valid

                author_stats["total_affiliations"] += num_total
                author_stats["valid_affiliations"] += num_valid
                author_stats["null_affiliations"] += num_null

                if _is_author_affiliation_null(author):
                    author_stats["authors_without_affiliations"] += 1
                else:
                    author_stats["authors_with_affiliations"] += 1

            if status == "all_null":
                works_all_missing_out.write(line)
            elif status == "some_null":
                works_some_missing_out.write(line)
            elif status == "no_authors":
                works_no_authors_out.write(line)
            else:
                works_complete_out.write(line)

    if parse_errors > 0:
        print(f"Warning: {parse_errors} lines skipped due to JSON parse errors", file=sys.stderr)

    return {"record_counts": counts, "author_stats": author_stats}


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.2f}%"


def write_stats_csv(stats: dict, output_file: Path) -> None:
    record_counts = stats["record_counts"]
    author_stats = stats["author_stats"]

    total_works = sum(record_counts.values())
    total_authors = author_stats["total_authors"]
    authors_with_affiliations = author_stats["authors_with_affiliations"]
    authors_without_affiliations = author_stats["authors_without_affiliations"]
    total_affiliations = author_stats["total_affiliations"]
    valid_affiliations = author_stats["valid_affiliations"]

    rows = [
        # Work-level stats
        {"category": "works_all_authors_missing_affiliations", "count": record_counts["all_null"],
         "percentage": _pct(record_counts["all_null"], total_works)},
        {"category": "works_some_authors_missing_affiliations", "count": record_counts["some_null"],
         "percentage": _pct(record_counts["some_null"], total_works)},
        {"category": "works_all_authors_have_affiliations", "count": record_counts["none_null"],
         "percentage": _pct(record_counts["none_null"], total_works)},
        {"category": "works_no_authors", "count": record_counts["no_authors"],
         "percentage": _pct(record_counts["no_authors"], total_works)},
        {"category": "total_works", "count": total_works,
         "percentage": "100.00%" if total_works > 0 else "N/A"},
        # Author-level stats
        {"category": "total_authors", "count": total_authors, "percentage": ""},
        {"category": "authors_with_affiliations", "count": authors_with_affiliations,
         "percentage": _pct(authors_with_affiliations, total_authors)},
        {"category": "authors_without_affiliations", "count": authors_without_affiliations,
         "percentage": _pct(authors_without_affiliations, total_authors)},
        # Affiliation-level stats
        {"category": "total_affiliations", "count": total_affiliations, "percentage": ""},
        {"category": "valid_affiliations", "count": valid_affiliations,
         "percentage": _pct(valid_affiliations, total_affiliations)},
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "count", "percentage"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Stats written to: {output_file}")


def print_stats(stats: dict) -> None:
    record_counts = stats["record_counts"]
    author_stats = stats["author_stats"]

    total_works = sum(record_counts.values())
    total_authors = author_stats["total_authors"]
    authors_with = author_stats["authors_with_affiliations"]
    authors_without = author_stats["authors_without_affiliations"]
    total_affiliations = author_stats["total_affiliations"]
    valid_affiliations = author_stats["valid_affiliations"]

    print("\n" + "=" * 60)
    print("WORK STATISTICS")
    print("=" * 60)

    if total_works == 0:
        print("No works to analyze.")
    else:
        print(f"All authors missing affiliations:  {record_counts['all_null']:>8,} ({record_counts['all_null'] / total_works * 100:5.2f}%)")
        print(f"Some authors missing affiliations: {record_counts['some_null']:>8,} ({record_counts['some_null'] / total_works * 100:5.2f}%)")
        print(f"All authors have affiliations:     {record_counts['none_null']:>8,} ({record_counts['none_null'] / total_works * 100:5.2f}%)")
        print(f"No authors in work:                {record_counts['no_authors']:>8,} ({record_counts['no_authors'] / total_works * 100:5.2f}%)")
        print("-" * 60)
        print(f"Total works:                       {total_works:>8,}")

    print("\n" + "=" * 60)
    print("AUTHOR STATISTICS")
    print("=" * 60)
    print(f"Total authors:                     {total_authors:>8,}")
    if total_authors > 0:
        print(f"Authors with affiliations:         {authors_with:>8,} ({authors_with / total_authors * 100:5.2f}%)")
        print(f"Authors without affiliations:      {authors_without:>8,} ({authors_without / total_authors * 100:5.2f}%)")
    else:
        print("Authors with affiliations:         N/A")
        print("Authors without affiliations:      N/A")

    print("\n" + "=" * 60)
    print("AFFILIATION STATISTICS")
    print("=" * 60)
    print(f"Total affiliations:                {total_affiliations:>8,}")
    if total_affiliations > 0:
        print(f"Valid affiliations:                {valid_affiliations:>8,} ({valid_affiliations / total_affiliations * 100:5.2f}%)")
    else:
        print("Valid affiliations:                N/A")
    print("=" * 60 + "\n")


def cmd_download(args: argparse.Namespace) -> None:
    download_dataset(args.repo_id, args.data_dir, force=args.force)


def cmd_sample(args: argparse.Namespace) -> None:
    create_sample(args.data_file, args.output, args.num, args.seed)


def cmd_analyze(args: argparse.Namespace) -> None:
    stats = analyze_sample(args.sample_file, args.output_dir)
    print_stats(stats)
    write_stats_csv(stats, args.output_dir / "stats.csv")

    print("Output files:")
    print(f"  {args.output_dir / 'works_all_missing.jsonl'}")
    print(f"  {args.output_dir / 'works_some_missing.jsonl'}")
    print(f"  {args.output_dir / 'works_complete.jsonl'}")
    print(f"  {args.output_dir / 'works_no_authors.jsonl'}")
    print(f"  {args.output_dir / 'stats.csv'}")


def cmd_run(args: argparse.Namespace) -> None:
    repo_dir = args.data_dir / _repo_to_dirname(args.repo_id)

    if args.skip_download:
        # Require existing data when --skip-download is used
        if not repo_dir.exists():
            print(f"Error: --skip-download specified but no data found at {repo_dir}", file=sys.stderr)
            sys.exit(1)
        try:
            data_file = _find_jsonl_file(repo_dir)
            print(f"Using existing: {data_file}")
        except FileNotFoundError:
            print(f"Error: --skip-download specified but no JSONL file found in {repo_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        # download_dataset will check for existing data automatically
        data_file = download_dataset(args.repo_id, args.data_dir, force=args.force)

    sample_dir = args.samples_dir / _repo_to_dirname(args.repo_id)
    sample_file = sample_dir / f"sample_{args.num}.jsonl"
    create_sample(data_file, sample_file, args.num, args.seed)

    stats = analyze_sample(sample_file, sample_dir)
    print_stats(stats)
    write_stats_csv(stats, sample_dir / "stats.csv")

    print("Output files:")
    print(f"  {sample_file}")
    print(f"  {sample_dir / 'works_all_missing.jsonl'}")
    print(f"  {sample_dir / 'works_some_missing.jsonl'}")
    print(f"  {sample_dir / 'works_complete.jsonl'}")
    print(f"  {sample_dir / 'works_no_authors.jsonl'}")
    print(f"  {sample_dir / 'stats.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample HuggingFace datasets and analyze null affiliation rates",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download a HuggingFace dataset")
    p_download.add_argument("repo_id", help="HuggingFace dataset repo (e.g., org/dataset)")
    p_download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                            help=f"Base directory for downloads (default: {DEFAULT_DATA_DIR})")
    p_download.add_argument("--force", action="store_true",
                            help="Force re-download even if data exists")
    p_download.set_defaults(func=cmd_download)

    p_sample = subparsers.add_parser("sample", help="Create a random sample from a JSONL file")
    p_sample.add_argument("data_file", type=Path, help="Input JSONL file")
    p_sample.add_argument("-n", "--num", type=int, default=DEFAULT_SAMPLE_SIZE,
                          help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    p_sample.add_argument("-o", "--output", type=Path, default=DEFAULT_SAMPLES_DIR / "sample.jsonl",
                          help=f"Output file (default: {DEFAULT_SAMPLES_DIR}/sample.jsonl)")
    p_sample.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p_sample.set_defaults(func=cmd_sample)

    p_analyze = subparsers.add_parser("analyze", help="Analyze a sample file for null affiliations")
    p_analyze.add_argument("sample_file", type=Path, help="Sample JSONL file to analyze")
    p_analyze.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_SAMPLES_DIR,
                           help=f"Output directory for results (default: {DEFAULT_SAMPLES_DIR})")
    p_analyze.set_defaults(func=cmd_analyze)

    p_run = subparsers.add_parser("run", help="Download, sample, and analyze (all-in-one)")
    p_run.add_argument("repo_id", nargs="?", default=DEFAULT_REPO,
                       help=f"HuggingFace dataset repo (default: {DEFAULT_REPO})")
    p_run.add_argument("-n", "--num", type=int, default=DEFAULT_SAMPLE_SIZE,
                       help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE})")
    p_run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                       help=f"Base directory for downloads (default: {DEFAULT_DATA_DIR})")
    p_run.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR,
                       help=f"Base directory for samples (default: {DEFAULT_SAMPLES_DIR})")
    p_run.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p_run.add_argument("--skip-download", action="store_true",
                       help="Require existing data (error if not found)")
    p_run.add_argument("--force", action="store_true",
                       help="Force re-download even if data exists")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
