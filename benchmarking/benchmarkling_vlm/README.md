# VLM Benchmarking for Author-Affiliation Parsing

Modal-based benchmarking scripts for extracting author and affiliation information from arXiv preprints using vision language Models with vLLM.

## Model test scripts
- MiniCPM-V-4.5 (`openbmb/MiniCPM-V-4_5`)
- RolmOCR (`reducto/RolmOCR`)

## Usage
```bash
# Upload data
modal run script.py --upload --image-dir /path/to/images --input-file metadata.json --prompt-file prompt.txt

# Run inference
modal run script.py --batch-size 1 --num-samples 100

# Download results
modal run script.py --download

# Clear data/cache
modal run script.py --clear
modal run script.py --clear-cache --model-name model/name
```

## Default Configuration
- GPU: H100
- Timeout: 4 hours
- Max image dimensions: 2000px side, 3.5MP total