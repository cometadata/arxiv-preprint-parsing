# Converter OpenAlex Works to Test Format

Retrieves OpenAlex works records converts author affiliation data to test.json format.

## Requirements

```bash
pip install requests
```

## Usage

Basic:
```bash
./convert_openalex_to_test_format.py -i input.json -o output.json
```

With polite pool access (recommended):
```bash
./convert_openalex_to_test_format.py -i input.json -o output.json -e your@email.com
```

## Options

- `-i, --input`: Input JSON file (default: test.json)
- `-o, --output`: Output JSON file (default: openalex_converted.json)
- `-e, --email`: Email for OpenAlex polite pool (recommended)
- `-r, --rate-limit`: Requests per second (default: 8, max: 10)
- `-v, --verbose`: Enable debug logging


## Input Format

Expects JSON with records containing `doi`, `arxiv_id`, and `filename` fields.

## Output Format

Produces JSON with author names and raw affiliation strings extracted from OpenAlex.
