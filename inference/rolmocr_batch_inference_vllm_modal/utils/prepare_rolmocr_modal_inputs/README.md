# RolmOCR Modal Inputs

Uploads PDFs to Modal volume and generates a batch inference manifest.

## Installation

Set up a [Modal](https://modal.com/) account, then:

```bash
pip install modal
```

## Usage

```bash
python prepare_rolmocr_modal_inputs.py /path/to/pdfs manifest.jsonl
```

## Arguments

- `pdf_dir`: Directory containing PDFs
- `manifest`: Local output path for manifest

## Options

- `--volume`: Modal volume name (default: `rolmocr-data`)
- `--manifest-remote`: Remote manifest path (default: `manifest.jsonl`)
- `--pdf-prefix`: Remote PDF directory (default: `pdfs`)
- `--recursive`: Include PDFs from subdirectories
- `--prompt`: Upload custom system prompt file
- `--prompt-prefix`: Remote prompt directory (default: `prompts`)
- `--modal-env`: Modal environment (e.g., `main`, `staging`)

## Output

Logs JSON summary and command to run batch inference to the console.

## Example

```bash
python prepare_rolmocr_modal_inputs.py docs/ manifest.jsonl --recursive --prompt custom.txt
```
