# kaggle_arxiv_dataset_dl

Download arXiv PDFs from the Kaggle arXiv dataset using arXiv IDs.

## Installation

1. Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install):


2. Install Python dependencies:
   ```bash
   pip install tqdm
   ```

## Usage

```bash
python kaggle_arxiv_dataset_dl.py -i input.json -o output_dir
```

### Arguments

- `-i, --input-json`: Path to JSON file containing list of objects with `arxiv_id` field (required)
- `-o, --output-dir`: Output directory for PDFs (default: `output`)
- `-f, --force-rebuild-manifest`: Force rebuild of manifest cache

### Input Format

```json
[
  {"arxiv_id": "2301.12345"},
  {"arxiv_id": "cs/0001001"}
]
```

See also [cometadata/202508-arxiv-cs-ai-cv-ro-lg-ma-cl-dois](https://huggingface.co/datasets/cometadata/202508-arxiv-cs-ai-cv-ro-lg-ma-cl-dois) for an example input.

## How it works

1. Downloads and caches the arXiv dataset file manifest
2. Maps the arXiv IDs in the JSON to the corresponding GCS URLs, mapping to the latest version of the PDF
3. Downloads the PDFs in parallel using `gsutil -m`

The manifest is then saved to a JSON output (`arxiv_pdf_manifest.json`) for use in/to speed up subsequent runs.
