# Check PDF FIles

Scans a specified directory (and its subdirectories) for `.pdf` files and verifies their integrity.

## Installation

```bash
pip install pikepdf
````

## Usage

```bash
python check_pdf_files -d path/to/your/pdf_directory/ [-o path/to/output_results.csv]
```

### Arguments

  * `-d DIRECTORY`, `--directory DIRECTORY`
      * **Required.** Path to the directory containing PDF files to verify.
  * `-o OUTPUT_CSV_PATH`, `--output_csv OUTPUT_CSV_PATH`
      * *Optional.* Path to save the detailed verification results as a CSV file.
      * If not provided, the CSV file will be named based on the input directory (e.g., `input_dir_name_verification_results.csv`) and saved in the current working directory.

### Example

To check PDFs in a folder named `my_article_pdfs` and save the CSV to the default location:

```bash
python check_pdf_files -d my_article_pdfs/
```

To specify a name and location for the output CSV:

```bash
python check_pdf_files -d my_article_pdfs/ -o reports/verification_data.csv
```

## Output

A CSV file is generated containing detailed results for each processed PDF. The columns are:

  * `File Path`: The full path to the PDF file.
  * `Status`: The verification status category (see below).
  * `Details`: A message providing more information about the status (e.g., page count, error message).

### Status Categories

  * **VALID:** The PDF opened successfully and contains one or more pages.
  * **CORRUPTED:** The PDF could not be opened or parsed correctly by `pikepdf`, or the file is empty (0 bytes).
  * **PASSWORD\_PROTECTED:** The PDF is encrypted and requires a password to open; its content cannot be verified.
  * **ZERO\_PAGES:** The PDF has a valid structure but contains no pages, which might indicate an issue.
  * **ERROR:** An unexpected issue occurred while trying to access or verify the file (e.g., file not found during processing, other exceptions).