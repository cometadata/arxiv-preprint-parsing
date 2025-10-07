# RolmOCR Batch Inference (vLLM + Modal)

Batch OCR processing for PDFs using [RolmOCR](https://huggingface.co/reducto/RolmOCR) model via [vLLM](https://github.com/vllm-project/vllm) on Modal.


## Installation

Set up a [Modal](https://modal.com/) then

```bash
pip install modal
```


## Usage

```bash
modal run rolmocr_batch_inference_vllm_modal.py --manifest manifest.jsonl
```

## Manifest Format

JSONL file with entries:
```json
{"document_id": "doc1", "pdf_path": "s3://bucket/doc.pdf", "pages": [1, 2, 3]}
{"document_id": "doc2", "pdf_path": "/local/path.pdf"}
```

## Arguments

- `manifest_path`: Input manifest file (default: `/data/manifest.jsonl`)
- `output_path`: Successful predictions (default: `/data/rolmocr_predictions.jsonl`)
- `error_path`: Failed pages (default: `/data/rolmocr_errors.jsonl`)
- `model_name`: HuggingFace model ID (default: `reducto/RolmOCR`)
- `tensor_parallel_size`: GPUs for tensor parallelism (default: 1)
- `max_pages_per_doc`: Limit pages per document
- `workers`: Concurrent document workers (default: 4)
- `max_concurrent_docs`: Max simultaneous documents (default: 8)
- `system_prompt`: Custom system prompt or path to file
