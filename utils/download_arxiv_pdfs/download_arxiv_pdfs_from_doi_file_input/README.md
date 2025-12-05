# Download arXiv PDFs

Downloads PDF articles from arXiv based on DOIs provided in input CSV files.


## Installation
```bash
pip install arxiv
````

## Usage

The script accepts either a single CSV file or a directory containing multiple CSV files as input.

**To process a single CSV file:**

```bash
python download_arxiv_pdfs.py -i path/to/your_file.csv
```

**To process all CSV files in a directory:**

```bash
python download_arxiv_pdfs.py -i path/to/your_csv_directory/
```

### Arguments

  * `-i`, `--input_path`: (Required) Path to a single CSV file or a directory containing CSV files.


## Input Format

The script expects input CSV files to contain a header row with a column named `doi`. This column should list the arXiv DOIs (e.g., `https://doi.org/10.48550/arxiv.2103.17074`).

Example `input.csv`:

```
openalex_id,doi
[https://openalex.org/W3143928891](https://openalex.org/W3143928891),[https://doi.org/10.48550/arxiv.2103.17074](https://doi.org/10.48550/arxiv.2103.17074)
[https://openalex.org/W2231573064](https://openalex.org/W2231573064),[https://doi.org/10.48550/arxiv.1509.02391](https://doi.org/10.48550/arxiv.1509.02391)
```

## Output

For each input CSV file (e.g., `my_papers.csv`), a corresponding directory named `my_papers_pdfs` will be created in the same location as the CSV file (or within the input directory if a directory was specified). Downloaded PDF files, named after their arXiv ID (e.g., `2103.17074.pdf`), will be saved into these respective output directories.