# Text Extraction Evaluation Framework

Evaluation framework for text‑extraction tasks with strict and fuzzy matching.


## Installation

`pip install -U numpy scipy rapidfuzz pyyaml`

## Quick Start
1) Create an eval YAML config and JSON schema for the task.
2) Run the CLI:
- `python run_evaluation.py -g ground_truth.json -p predictions.json -c eval_config.yaml -o evaluation_results.json -v`

## Configuration Example
```yaml
task_name: author_affiliation_v1
entity_schema_path: author_schema.json
reporting_modes: [strict, fuzzy]
key_field: name
field_eval_rules:
  name:
    match_type: fuzzy        # strict | fuzzy
    normalization: true      # normalize before matching
    similarity_threshold: 0.95  # used only when fuzzy
  affiliations:
    match_type: fuzzy
    normalization: true
    similarity_threshold: 0.95
combined_eval:
  harsh_penalty: true        # partial matches add both FP and FN
```
Notes
- Reporting modes select which precomputed buckets to display; they do not alter field logic.
- If a field uses `match_type: strict`, its fuzzy bucket equals the strict bucket.

## Schema Example
```json
{
  "entity_name": "Author",
  "doc_id_field": "arxiv_id",
  "entities_field": "authors",
  "fields": {
    "name": { "type": "string" },
    "affiliations": { "type": "array[string]" }
  }
}
```
- `doc_id_field` identifies documents.
- `entities_field` points to the list to evaluate.
- `fields.type` supports `string` and `array[string]` here.

## Matching Semantics
- Key field (entity matching)
  - Strict: normalized equality on the key; duplicates collapse by normalized value (i.e. last wins).
  - Fuzzy: attempted only for keys left unmatched after strict, using the key’s `similarity_threshold`.
- Attributes
  - `array[string]`: strict uses normalized set intersection/differences; fuzzy tries to match remaining unique strings by threshold. Duplicates collapse by normalized value.
  - `string`: strict uses normalized equality; fuzzy applies threshold if enabled.

## Output
- Aggregated reports are grouped by evaluation mode and metric category:
```json
{
  "category_labels": {
    "entity:author": "author_identification",
    "field:affiliations": "affiliations_matching",
    "combined": "combined_authors"
  },
  "reports": {
    "strict": {
      "entity:author": {"true_positives": 10, "false_positives": 2, "false_negatives": 3, ...},
      "field:affiliations": {"true_positives": 12, ...},
      "combined": {"true_positives": 8, ...}
    },
    "fuzzy": {
      "entity:author": {...},
      "field:affiliations": {...},
      "combined": {...}
    }
  },
  "document_results": [
    {
      "doc_id": "arXiv:2003.03151",
      "status": "success",
      "metrics": {
        "entity:author": {"strict": {...}, "fuzzy": {...}},
        "field:affiliations": {"strict": {...}, "fuzzy": {...}},
        "combined": {"strict": {...}, "fuzzy": {...}}
      },
      "details": {
        "entity_matches": [...],
        "array_field_details": {"affiliations": [...]}
      }
    }
  ]
}
```
- `category_labels` translate metric keys (`entity:*`, `field:*`, `combined`) into task-specific names.
- `document_results` always include the document identifier, per-mode metrics, and optional detail payloads.
- Documents with missing predictions are recorded as `{status: "error", error: "Missing prediction"}` and are excluded from aggregation. A `predicted_*` list of `null` is recorded as `{status: "null_prediction"}`.


## Development
- Run tests with: `pip install pytest && pytest -q`
