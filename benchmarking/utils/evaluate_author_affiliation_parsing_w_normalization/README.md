# Author-Affiliation Evaluation with Normalization

Evaluates author-affiliation extraction performance using text normalization.

## Usage

```bash
python evaluate_author_affiliation_parsing_w_normalization.py \
  -g ground_truth.json \
  -p predictions.jsonl \
  -o evaluation_results.json \
  --analyze
```

### Arguments
- `-g, --ground-truth`: Path to ground truth JSON file (required)
- `-p, --predictions`: Path to predictions JSONL file (required)
- `-o, --output`: Output evaluation results file (default: `evaluation_aggressive_norm.json`)
- `-a, --analyze`: Show normalization impact analysis


## Input Format

### Ground Truth (JSON)
```json
[
  {
    "arxiv_id": "1234.5678",
    "authors": [
      {
        "name": "John Doe",
        "affiliations": ["MIT", "Harvard"]
      }
    ]
  }
]
```

### Predictions (JSONL)
```json
{"arxiv_id": "1234.5678", "predicted_authors": [{"name": "John Doe", "affiliations": ["MIT", "Harvard"]}]}
```

##  Metrics

- Aggregate precision/recall/F1 for authors and affiliations
- Per-document evaluation
- Documents with 100% accuracy
- Best/worst performing documents