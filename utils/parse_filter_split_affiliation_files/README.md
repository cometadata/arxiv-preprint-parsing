# Parse Filter Split Affiliation Files

## Overview

Filters rows from an input CSV file based on DOI content into three separate output files:
1.  DOIs containing "arxiv".
2.  DOIs not containing "arxiv".
3.  Entries without DOIs.

Assumes a CSV with acolumn named 'doi' that contains DOI values.

## Usage

```bash
python parse_filter_split_affiliation_files.py -i <input_csv> [-a <arxiv_output_csv>] [-n <non_arxiv_output_csv>] [-x <no_doi_output_csv>]
````

  - `-i, --input <input_csv>`: **Required.** Path to the input CSV file.
  - `-a, --arxiv_output <arxiv_output_csv>`: Output for "arxiv" DOIs.
    (Default: `<input_base>_arxiv.csv`)
  - `-n, --non_arxiv_output <non_arxiv_output_csv>`: Output for other DOIs.
    (Default: `<input_base>_non_arxiv.csv`)
  - `-x, --no_doi_output <no_doi_output_csv>`: Output for entries without DOIs.
    (Default: `<input_base>_no_doi.csv`)

## Example

```bash
python parse_filter_split_affiliation_files.py -i input_data.csv
```

This will output:

  - `input_data_arxiv.csv`
  - `input_data_non_arxiv.csv`
  - `input_data_no_doi.csv`
