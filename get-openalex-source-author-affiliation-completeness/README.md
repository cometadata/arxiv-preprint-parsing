# OpenAlex Source Affiliation Analyzer

Analyzes author affiliation completeness for works from a specific OpenAlex source ID by parsing OpenAlex snapshot data.

## Installation

```bash
cargo build --release
````

## Usage

```bash
./target/release/openalex-source-analyzer --input-dir <PATH_TO_INPUT_JSONL_GZ_FILES> --output-dir <PATH_TO_OUTPUT_DIRECTORY> --source-id <OPENALEX_SOURCE_ID> [OPTIONS]
```

### Required Arguments:

  * `-i, --input-dir <INPUT_DIR>`: Directory containing input JSONL.gz files. Recursively searches for `.gz` files, excluding those ending in `.csv.gz`.
  * `-o, --output-dir <OUTPUT_DIR>`: Output directory for the summary text file and categorized CSV files.
  * `-s, --source-id <SOURCE_ID>`: OpenAlex Source ID to analyze (e.g., S4306400194).

### Optional Arguments:

  * `--log-level <LOG_LEVEL>`: Logging level (DEBUG, INFO, WARN, ERROR). Default: `INFO`.
  * `-t, --threads <THREADS>`: Number of threads to use (0 for auto-detection). Default: `0`.
  * `--stats-interval <STATS_INTERVAL>`: Interval in seconds to log statistics during processing. Default: `60`.

## Example 

```bash
./target/release/get-openalex-source-author-affiliation-completeness \
    --input-dir /path/to/openalex-snapshot/data/works \
    --output-dir /path/to/analysis_results \
    --source-id S123456789 
```

## Input

The script expects as input gzipped JSONL files (`.jsonl.gz`) from the OpenAlex snapshot file, where each line is a JSON object representing an OpenAlex work record.

## Output

1.  In the specified output directory, for each category of affiliation status:

      * `source_<SOURCE_ID>_all_affiliations.csv`: Works where all authors have at least one institution listed.
      * `source_<SOURCE_ID>_partial_affiliations.csv`: Works where some, but not all, authors have institutions.
      * `source_<SOURCE_ID>_no_affiliations.csv`: Works where authors are listed, but none have any institutions.
      * `source_<SOURCE_ID>_no_authors.csv`: Works where no authorships are listed or the authorships array is empty.

2.  A summary stats file is also produced:

      * `summary_text_<SOURCE_ID>.txt`: A text file summarizing the processing run, including total works found, counts for each affiliation category, percentages, median affiliation percentage, and runtime statistics.