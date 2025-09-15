# Convert to Inference Ooutput Evaluation Format

Converts author-affiliation inference results from JSONL format to evaluation-ready JSON.

## Usage

```bash
python convert_inferrence_output_to_evaluation_format.py -i input.jsonl -o predictions.json [-g ground_truth.json] [-v]
```

## Arguments

- `-i, --input`: Input JSONL file with inference results (required)
- `-o, --output`: Output JSON file for predictions (default: predictions.json)
- `-g, --ground-truth`: Optional ground truth file to match arXiv ID formats
- `-v, --verbose`: Enable verbose logging

## Input Format

JSONL file with entries containing:
```json
{
  "arxiv_id": "arXiv:1234.5678",
  "predicted_authors": [
    {
      "name": "Author Name",
      "affiliations": ["Institution 1", "Institution 2"]
    }
  ]
}
```

## Output Format

JSON dictionary mapping arXiv IDs to author lists:
```json
{
  "arXiv:1234.5678": [
    {
      "name": "Author Name",
      "affiliations": ["Institution 1", "Institution 2"]
    }
  ]
}
```

## Testing

```bash
python test_convert_inference_output.py
```