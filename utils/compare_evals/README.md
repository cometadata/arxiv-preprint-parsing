# Compare Evals

Compares precision, recall, and F-scores between two evaluation reports.

## Usage

```bash
python compare_evals.py -b baseline.json -c candidate.json [-o output.json]
```

## Arguments

- `-b, --baseline`: Path to baseline eval JSON file
- `-c, --candidate`: Path to candidate eval JSON file
- `-o, --output`: Optional path to write JSON output (prints to stdout if omitted)

## Output

JSON with baseline/candidate values and deltas for each metric across all modes and categories.