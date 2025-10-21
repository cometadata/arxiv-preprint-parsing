# dots.ocr Batch Inference

Batch OCR and layout analysis for PDFs using the [dots.ocr](https://github.com/rednote-hilab/dots.ocr) model and [vLLM](https://github.com/vllm-project/vllm) on [Modal](https://modal.com/).


## Prerequisites

1. Modal account with volumes created for data, model weights, and the vLLM cache.
2. Set the environment variables before invoking `modal run`, e.g.:
   ```bash
   export DOTS_MODAL_APP_NAME="dots-ocr-vllm-batch"
   export DOTS_MODAL_DATA_VOLUME="dots-ocr-data"
   export DOTS_MODAL_MODEL_VOLUME="dots-ocr-models"
   export DOTS_MODAL_VLLM_CACHE_VOLUME="dots-ocr-vllm-cache"
   ```
3. Install runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Running The Batch Pipeline

```bash
modal run dots-ocr_batch_inference_vllm_modal.py \
  --manifest /data/manifest.jsonl \
  --output-path /data/dots_predictions.jsonl \
  --error-path /data/dots_errors.jsonl \
  --model-name rednote-hilab/dots.ocr \
  --download-dir /tmp/pdf_files \
  --log-path /data/dots_batch.log \
  --model-cache-dir /models \
  --vllm-port 30024 \
  --prompt-mode layout-all \
  --workers 4
```

All paths must exist inside the Modal container (i.e. via mounted volumes).

### Processing A Subset Of PDFs

```bash
modal run dots-ocr_batch_inference_vllm_modal.py \
  --manifest /data/manifest.jsonl \
  --output-path /data/dots_predictions.jsonl \
  --error-path /data/dots_errors.jsonl \
  --model-name rednote-hilab/dots.ocr \
  --download-dir /tmp/pdf_files \
  --log-path /data/dots_batch.log \
  --model-cache-dir /models \
  --vllm-port 30024 \
  --pdf-filenames "doc1.pdf,doc2.pdf"
```

### Optional Concurrency and Retry Arguments

- `--max-concurrent-docs`: hard cap on concurrent documents (default logic: min(workers, 8))
- `--max-inflight-requests`: cap on simultaneous API calls (default: 128)
- `--max-page-concurrency`: limit pages per document processed in parallel (default: 2)
- `--max-pages-per-doc`: truncate documents to the first *n* pages
- `--request-timeout`: timeout in seconds for completion requests (default: 180)
- `--max-retries`: retries per page on failure (default: 3)
- `--batch-id`: annotate metrics with a custom identifier
- `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`: pass-through tuning knobs for vLLM


## Manifest Format

The input manifest is JSONL, one record per document:

```json
{"document_id": "doc1", "pdf_path": "s3://bucket/doc.pdf", "pages": [1, 2, 3]}
{"document_id": "doc2", "pdf_path": "/local/path.pdf"}
```

- `document_id` or `id`: unique identifier for the document
- `pdf_path`: S3 URI, HTTP URL, or local/volume path
- `pages`: optional page numbers to process; the entire document is used if omitted


## Prompt Modes

Choose with `--prompt-mode`:

- `layout-all`: layout categories, bounding boxes, and associated text
- `layout-only`: layout categories and bounding boxes without text
- `ocr`: Markdown OCR output without layout annotations


## Outputs

Two JSONL files are generated: one for successful pages, the other for failures.

### Success File (`--output-path`)

Each record contains page-level metadata and the generated content:

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

### Error File (`--error-path`)

Failures record the exception string:

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
