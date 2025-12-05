# arXiv PDF Downloader

Downloads the most recent version of every PDF from the [arXiv Kaggle dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).


## Installation

```bash
pip install gsutil
```

or

```bash
uv sync
```

## Usage

### Generate manifest

Output a TSV file with the Kaggle GCS source paths and local filenames:

```bash
uv run python download_latest_arxiv_pdfs.py -m manifest.tsv
uv run python download_latest_arxiv_pdfs.py -m -  # stdout
```

Manifest format:
```
gs://arxiv-dataset/arxiv/pdf/2411/2411.00003v4.pdf	2411.00003v4.pdf
gs://arxiv-dataset/arxiv/hep-th/pdf/9201/9201001v1.pdf	hep-th_9201001v1.pdf
```

### Download directly

Download all PDFs to a directory:

```bash
uv run python download_latest_arxiv_pdfs.py -d ./pdfs/
uv run python download_latest_arxiv_pdfs.py -d ./pdfs/ -p 16  # custom parallelism
```

### Options

| Option | Description |
|--------|-------------|
| `-m, --manifest FILE` | Generate TSV manifest (use `-` for stdout) |
| `-d, --download DIR` | Download PDFs to directory |
| `-p, --parallel N` | Number of parallel downloads (default: 8) |

### Using the manifest manually

```bash
while IFS=$'\t' read -r src dest; do
    gsutil cp "$src" "./pdfs/$dest"
done < manifest.tsv
```

## Filename Format

| Format | arXiv ID | GCS Path | Local Filename |
|--------|----------|----------|----------------|
| New (2007+) | `2411.00003` | `.../pdf/2411/2411.00003v4.pdf` | `2411.00003v4.pdf` |
| Legacy | `hep-th/9201001` | `.../hep-th/pdf/9201/9201001v1.pdf` | `hep-th_9201001v1.pdf` |

Legacy papers include the subject prefix in the filename to avoid collisions (e.g., `hep-th_9201002v1.pdf` and `hep-lat_9201002v1.pdf` are different papers).