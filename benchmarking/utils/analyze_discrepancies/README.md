# Discrepancy Analysis Tool

Analyzes evaluation results to identify and categorize extraction errors in author-affiliation parsing, distinguishing between formatting differences and real content mismatches.

## Usage

```bash
python analyze_discrepancies.py -e evaluation_results_detailed.json -o output_prefix
```

### Arguments
- `-e, --evaluation`: Path to evaluation results JSON (default: `evaluation_results_detailed.json`)
- `-o, --output-prefix`: Prefix for output files (default: `normalized_discrepancy_analysis`)
- `-v, --verbose`: Print detailed analysis summary

## Output Files

- `{prefix}_complete.json`: Full analysis with all discrepancies
- `{prefix}_summary.json`: Statistical summary
- `{prefix}_real_mismatches.json`: Real content differences only (excluding formatting)
- `{prefix}_error_patterns.json`: Most common error patterns

## Metrics

- Documents with perfect author/affiliation extraction
- Formatting-only vs. real content mismatches
- Most commonly missed/extra affiliations
- Per-document discrepancy counts