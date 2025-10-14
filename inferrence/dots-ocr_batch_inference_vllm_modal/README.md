# dots.ocr Batch Inference

Batch OCR and layout analysis for PDFs using [dots.ocr](https://github.com/rednote-hilab/dots.ocr) model via [vLLM](https://github.com/vllm-project/vllm) on Modal](https://modal.com/).


## Installation

Set up a [Modal](https://modal.com/) account, then:

```bash
pip install modal
```

## Usage

```bash
modal run dots-ocr_batch_inference_vllm_modal.py --manifest manifest.jsonl
```

### Basic Usage

```bash
modal run dots-ocr_batch_inference_vllm_modal.py \
  --manifest /data/manifest.jsonl \
  --prompt-mode layout-all \
  --workers 4
```

### Process Specific PDFs

```bash
modal run dots-ocr_batch_inference_vllm_modal.py \
  --manifest manifest.jsonl \
  --pdf-filenames "doc1.pdf,doc2.pdf,doc3.pdf"
```


## Manifest Format

JSONL file with entries:
```json
{"document_id": "doc1", "pdf_path": "s3://bucket/doc.pdf", "pages": [1, 2, 3]}
{"document_id": "doc2", "pdf_path": "/local/path.pdf"}
```

Fields:
- `document_id` or `id`: Unique document identifier
- `pdf_path`: Path to PDF (S3 URI or local path)
- `pages`: Optional list of page numbers to process (processes all pages if omitted)


## Prompt Modes

The script supports three modes via the `--prompt-mode` argument:

- `layout-all` (default): Outputs layout information with bounding boxes, categories, and text content.

- `layout-only`: Outputs only layout bounding boxes and categories without text content.

- `ocr`: Simple OCR mode that converts document images to markdown format.


## Arguments

### Main Arguments

- `--manifest`: Input manifest file (default: `/data/manifest.jsonl`)
- `--prompt-mode`: Processing mode - `layout-all`, `layout-only`, or `ocr` (default: `layout-all`)
- `--workers`: Concurrent document workers (default: 4)
- `--max-concurrent-docs`: Max simultaneous documents (default: min(workers, 8))
- `--max-inflight-requests`: Max concurrent API requests (default: 128)
- `--pdf-filenames`: Comma or space-separated list of specific PDF filenames to process

### Additional Arguments

- `output_path`: Successful predictions (default: `/data/dots_predictions.jsonl`)
- `error_path`: Failed pages (default: `/data/dots_errors.jsonl`)
- `model_name`: HuggingFace model ID (default: `rednote-hilab/dots.ocr`)
- `tensor_parallel_size`: GPUs for tensor parallelism (default: 1)
- `gpu_memory_utilization`: GPU memory utilization fraction
- `max_model_len`: Maximum model context length (default: 131072)
- `max_pages_per_doc`: Limit pages per document
- `request_timeout`: API request timeout in seconds (default: 180.0)
- `max_retries`: Maximum retry attempts per page (default: 3)
- `batch_id`: Optional batch identifier for tracking


## Output Format

### Success File (`dots_predictions.jsonl`)

Each line contains:
```json
{
  "document_id": "doc1",
  "page": 1,
  "page_index": 1,
  "pdf_path": "s3://bucket/doc.pdf",
  "model": "dots-ocr",
  "success": true,
  "latency": 2.34,
  "usage": {"prompt_tokens": 1024, "completion_tokens": 512, "total_tokens": 1536},
  "content": "extracted content here",
  "error": null
}
```

### Error File (`dots_errors.jsonl`)

Failed pages with error information:
```json
{
  "document_id": "doc1",
  "page": 2,
  "page_index": 2,
  "pdf_path": "s3://bucket/doc.pdf",
  "model": "dots-ocr",
  "success": false,
  "latency": 1.23,
  "usage": {},
  "content": null,
  "error": "render_failed: ..."
}
```
