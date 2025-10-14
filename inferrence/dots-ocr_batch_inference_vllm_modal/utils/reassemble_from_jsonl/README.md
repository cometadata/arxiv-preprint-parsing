# JSONL to Markdown Reassembler

Converts batch inference results into assembled markdown documents and optionally simpler document (vs. page-level) JSONL with just the extracted text.

## Usage

```bash
python reassemble_from_jsonl.py -i input.jsonl -o output_dir/
```

## Input Format

JSONL file with one page per line:
```json
{"document_id": "doc_123", "page": 1, "success": true, "content": "[{\"category\": \"Title\", \"text\": \"...\"}, ...]"}
```

## Options

- `-i, --input_file`: Input JSONL file (required)
- `-o, --output_dir`: Output directory for markdown files (required)
- `--no-page-markers`: Omit `<!-- Page N -->` markers
- `--skip-headers-footers`: Exclude page headers/footers
- `--output-jsonl`: Also output JSONL format (for downstream processing)

## Output

- One `.md` file per document in output directory
- `processing_report.txt` with success/failure statistics
- Optional JSONL with assembled documents


## Examples

```bash
# Basic
python reassemble_from_jsonl.py -i results.jsonl -o markdown/

# Clean output without markers/headers
python reassemble_from_jsonl.py -i results.jsonl -o clean/ \
  --no-page-markers --skip-headers-footers

# With JSONL output
python reassemble_from_jsonl.py -i results.jsonl -o output/ \
  --output-jsonl assembled.jsonl
```
