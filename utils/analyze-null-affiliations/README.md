# analyze null arXiv affiliations

Sample COMET's arXiv author affiliations dataset and analyze null affiliation rates.

## Usage

```bash
# Full pipeline (download, sample, analyze)
uv run analyze_null_affiliations.py run

# With options
uv run analyze_null_affiliations.py run cometadata/arxiv-author-affiliations-matched-ror-ids -n 50000 --seed 42

# Individual commands
uv run analyze_null_affiliations.py download <repo_id>
uv run analyze_null_affiliations.py sample <data_file> -n 10000 -o samples/sample.jsonl
uv run analyze_null_affiliations.py analyze <sample_file> -o results/
```

## Output

- `works_all_missing.jsonl` - all authors lack affiliations
- `works_some_missing.jsonl` - partial affiliation coverage
- `works_complete.jsonl` - all authors have affiliations
- `works_no_authors.jsonl` - no author data
- `stats.csv` - summary statistics

## Options

| Flag | Description |
|------|-------------|
| `-n, --num` | Sample size (default: 10000) |
| `--seed` | Random seed for reproducibility |
| `--data-dir` | Download directory (default: data/) |
| `--samples-dir` | Sample output directory (default: samples/) |
| `--skip-download` | Use existing data only |
| `--force` | Re-download even if data exists |
