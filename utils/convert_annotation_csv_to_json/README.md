# Convert Annotation CSV to JSON

Converts arXiv annotation CSV file to a structured JSON format.

## Usage

```bash
python convert_annotation_csv_to_json.py -i input.csv -o output.json
```

## Arguments

- `-i, --input` (required): Path to the input CSV file
- `-o, --output` (required): Path for the output JSON file

## Input CSV Format

Expected columns:
- `PDF File Link` - PDF filename (e.g., `1234.5678.pdf`)
- `title` - Publication title
- `title_lang` - Title language
- `authorName` - Author name
- `author_affiliation` - Author affiliation

## Output JSON Format

```json
[
  {
    "arxiv_id": "arXiv:1234.5678",
    "doi": "https://doi.org/10.48550/arXiv.1234.5678",
    "title": {
      "text": "Paper Title",
      "lang": "en"
    },
    "authors": [
      {
        "name": "Author Name",
        "affiliations": ["Institution"]
      }
    ]
  }
]
```