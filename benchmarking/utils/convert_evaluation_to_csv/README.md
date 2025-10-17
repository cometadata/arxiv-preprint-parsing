# Convert Evaluation JSON to CSV

Converts author affiliation extraction evaluation JSON into CSV files for easier analysis.

## Usage

```bash
python convert_evaluation_to_csv.py -i evaluation.json -o results/run1
```

## Output Files

- `{prefix}_correct.csv` - Documents with perfect author extraction (no false positives/negatives)
- `{prefix}_errors.csv` - Documents with extraction issues, including detailed notes
- `{prefix}_errors_details.csv` - Row-per-issue breakdown of all mismatches

## Options

- `-i, --input` - Input evaluation JSON file (required)
- `-o, --output-prefix` - Output path prefix for CSV files (required)
- `--correct-suffix` - Suffix for correct predictions CSV (default: `_correct.csv`)
- `--error-suffix` - Suffix for error summaries CSV (default: `_errors.csv`)
