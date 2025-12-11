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

### 1. Generate manifest

First, generate a TSV manifest of all PDFs to download:

```bash
python download_latest_arxiv_pdfs.py manifest output.tsv
python download_latest_arxiv_pdfs.py manifest -  # stdout
```

Manifest format:
```
gs://arxiv-dataset/arxiv/pdf/2411/2411.00003v4.pdf	2411.00003v4.pdf
gs://arxiv-dataset/arxiv/hep-th/pdf/9201/9201001v1.pdf	hep-th_9201001v1.pdf
```

### 2. Download using manifest

Download PDFs from a manifest file:

```bash
python download_latest_arxiv_pdfs.py download manifest.tsv ./pdfs/
python download_latest_arxiv_pdfs.py download manifest.tsv ./pdfs/ -p 16  # 16 parallel downloads
```

#### Resume and batch downloads

For large downloads, use batching with resume:

```bash
python download_latest_arxiv_pdfs.py download manifest.tsv ./pdfs/ -r -b 10000  # download 10k
python download_latest_arxiv_pdfs.py download manifest.tsv ./pdfs/ -r -b 10000  # next 10k
python download_latest_arxiv_pdfs.py download manifest.tsv ./pdfs/ -r           # until complete
```

### Options

**manifest command:**
| Argument | Description |
|----------|-------------|
| `FILE` | Output manifest file (use `-` for stdout) |

**download command:**
| Argument | Description |
|----------|-------------|
| `MANIFEST` | Input manifest file |
| `DIR` | Output directory for PDFs |
| `-p, --parallel N` | Number of parallel downloads (default: 8) |
| `-r, --resume` | Skip files that already exist |
| `-b, --batch N` | Download only N files per run |


## SLURM Batch Downloads

For cluster environments, use the SLURM batch scripts to parallelize downloads across jobs.

### 1. Prepare batches

```bash
# Split into batches of 10,000 files each
python prepare_slurm_batches.py manifest.tsv ./pdfs/ --batch-size 10000

# Or specify number of batches
python prepare_slurm_batches.py manifest.tsv ./pdfs/ --batches 25

# Preview without writing files
python prepare_slurm_batches.py manifest.tsv ./pdfs/ --batch-size 10000 --dry-run
```

This creates:
```
slurm_jobs/
├── batches/
│   ├── batch_001.txt
│   ├── batch_002.txt
│   └── ...
├── jobs/
│   ├── download_batch_001.sbatch
│   ├── download_batch_002.sbatch
│   └── ...
├── logs/
└── submit_all.sh
```

### 2. Submit jobs

```bash
# Submit all jobs to queue
./slurm_jobs/submit_all.sh

# Or submit individually
sbatch slurm_jobs/jobs/download_batch_001.sbatch
```

### SLURM options

| Option | Description |
|--------|-------------|
| `--batch-size N` | Files per batch |
| `--batches N` | Number of batches (alternative to --batch-size) |
| `--slurm-dir DIR` | Output directory for SLURM files (default: ./slurm_jobs) |
| `--cpus N` | CPUs per task / parallel downloads (default: 8) |
| `--conda-env NAME` | Conda environment to activate (default: comet-inference) |
| `--work-dir DIR` | Working directory for jobs (default: current directory) |
| `--dry-run` | Preview without writing files |


## Filename Format

| Format | arXiv ID | GCS Path | Local Filename |
|--------|----------|----------|----------------|
| New (2007+) | `2411.00003` | `.../pdf/2411/2411.00003v4.pdf` | `2411.00003v4.pdf` |
| Legacy | `hep-th/9201001` | `.../hep-th/pdf/9201/9201001v1.pdf` | `hep-th_9201001v1.pdf` |

Legacy papers include the subject prefix in the filename to avoid collisions.
