# kaggle_arxiv_dataset_dl

Download arXiv PDFs from the Kaggle arXiv dataset using arXiv IDs. The tool now supports manifest caching, automatic fallback searches on Google Cloud Storage, dry-run verification, and optional JSON outputs that summarize what was found or missing.

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
- `--dry-run`: Resolve URLs and report status without downloading
- `--write-resolved-json PATH`: Write all resolved IDs with `gcs_url` to `PATH`
- `--write-not-found-json PATH`: Write IDs that could not be resolved to `PATH`
- `--write-pending-json PATH`: Write the subset of resolved IDs not already present in the output directory

### Input Format

```json
[
  {"arxiv_id": "2301.12345"},
  {"arxiv_id": "cs/0001001"}
]
```

See also [cometadata/202508-arxiv-cs-ai-cv-ro-lg-ma-cl-dois](https://huggingface.co/datasets/cometadata/202508-arxiv-cs-ai-cv-ro-lg-ma-cl-dois) for an example input.

## How it works

1. Downloads and caches the arXiv dataset file manifest.
2. Maps the arXiv IDs in the JSON to the corresponding GCS URLs, mapping to the latest available version of each PDF when the manifest contains an entry.
3. For IDs missing from the manifest, constructs the expected GCS path and issues a wildcard `gsutil ls` to discover the available versions.
4. Optionally writes summary JSON files for downstream processing.
5. Downloads the PDFs in parallel using `gsutil -m`, skipping any files that already exist in the output directory.

The manifest is then saved to a JSON output (`arxiv_pdf_manifest.json`) for use in/to speed up subsequent runs.
