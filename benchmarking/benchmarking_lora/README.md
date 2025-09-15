# Author-Affiliation Parsing Benchmark Inference

Inference pipeline for extracting/benchmarking author affiliations from arXiv papers using vLLM on Modal.

## Usage

### Upload data files
```bash
modal run benchmark_author-affiliation_parsing_vllm_modal.py --upload \
  --input-file merged_output.json \
  --prompt-file prompt.txt
```

### Run inference
```bash
modal run benchmark_author-affiliation_parsing_vllm_modal.py \
  --batch-size 4 \
  --num-samples 100
```

### Download results
```bash
modal run benchmark_author-affiliation_parsing_vllm_modal.py --download
```

### Clear model cache
```bash
modal run benchmark_author-affiliation_parsing_vllm_modal.py --clear-cache
```

## Configuration

Environment variables:
- `BASE_MODEL`: Base model ID (default: "Qwen/Qwen3-4B")
- `LORA_PATH`: LoRA adapter path (default: "cometadata/affiliation-parsing-lora-Qwen3-4B")
- `MAX_LORA_RANK`: Maximum LoRA rank (default: 64)
- `TENSOR_PARALLEL_SIZE`: GPU parallelism (default: 1)
-
## Input/Output

- **Input**: `merged_output.json` - JSON with `arxiv_id` and `markdown_text` fields
- **Prompt**: `prompt.txt` - System prompt template
- **Output**: `predictions.jsonl` - Line-delimited JSON with predictions

## Notes

- Configuration is a H100 GPU with 32GB memory.