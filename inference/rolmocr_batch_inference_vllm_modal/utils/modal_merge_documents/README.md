# Modal Merge Documents

Downloads page-level predictions from Modal and merges into document-level JSONL.


## Installation

Set up a [Modal](https://modal.com/) account, then:

```bash
pip install modal
```

## Usage

```bash
python modal_merge_documents.py --output merged_docs.jsonl
```

## Options

- `--volume`: Modal volume name (default: `rolmocr-data`)
- `--remote-predictions`: Page predictions file on volume (default: `rolmocr_predictions.jsonl`)
- `--remote-errors`: Optional errors file to download
- `--output`: Output path for merged documents (required)
- `--download-dir`: Local cache directory (default: `.modal_downloads`)
- `--modal-cmd`: Modal CLI executable (default: `modal` or `$MODAL_CLI`)
- `--skip-download`: Use cached files, skip Modal download

## Output Format

Each line contains document-level record:
```json
{
  "document_id": "doc1",
  "text": "combined page text...",
  "pages": [...],
  "page_count": 5,
  "success": true,
  "total_latency": 12.5,
  "usage": {"prompt_tokens": 100, "completion_tokens": 200},
  "rotation_applied": {"1": 0, "2": 90}
}
```
