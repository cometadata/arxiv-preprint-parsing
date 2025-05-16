import os
import argparse
import csv
import pikepdf


def verify_pdf_file(file_path):
    if not os.path.exists(file_path):
        return "error", "File does not exist."
    if os.path.getsize(file_path) == 0:
        return "corrupted", "File is empty (0 bytes)."

    try:
        with pikepdf.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages > 0:
                return "valid", f"Valid PDF, {num_pages} pages."
            else:
                return "zero_pages", "Valid PDF structure, but 0 pages."
    except pikepdf.PasswordError:
        return "password_protected", "PDF is password-protected."
    except pikepdf.PdfError as e:
        return "corrupted", f"Corrupted or invalid PDF (pikepdf error: {str(e)[:100]}...)."
    except FileNotFoundError:
        return "error", "File not found during pikepdf.open (should not happen)."
    except Exception as e:
        return "error", f"Unexpected error verifying PDF: {str(e)[:100]}..."


def main():
    parser = argparse.ArgumentParser(
        description="Verifies PDF files in a given directory (and subdirectories) "
                    "to check for corruption or other issues. Writes results to a CSV file."
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="Path to the directory containing PDF files to verify.", dest="input_directory"
    )
    parser.add_argument(
        "-o", "--output_csv",
        help="Path to save the detailed verification results as a CSV file. "
             "If not provided, defaults to a name based on the input directory, saved in the current working directory (e.g., 'input_dir_name_verification_results.csv').", default=None, metavar="OUTPUT_CSV_PATH"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_directory):
        print(f"Error: Input path '{args.input_directory}' is not a valid directory.")
        return

    print(f"Starting PDF verification in directory: {args.input_directory}\n")

    summary = {
        "total_scanned": 0,
        "valid": 0,
        "corrupted": 0,
        "password_protected": 0,
        "zero_pages": 0,
        "other_errors": 0,
    }

    csv_results_data = []
    csv_fieldnames = ['File Path', 'Status', 'Details']

    pdf_files_found = []
    for root, _, files in os.walk(args.input_directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files_found.append(os.path.join(root, file))

    if not pdf_files_found:
        print("No PDF files found in the specified directory.")
        return

    summary["total_scanned"] = len(pdf_files_found)

    for pdf_path in pdf_files_found:
        print(f"Verifying: {pdf_path}")
        category, message = verify_pdf_file(pdf_path)

        result_for_csv = {'File Path': pdf_path,
                          'Status': category.upper(), 'Details': message}
        csv_results_data.append(result_for_csv)

        if category == "valid":
            summary["valid"] += 1
            print(f"  Status: VALID - {message}")
        elif category == "corrupted":
            summary["corrupted"] += 1
            print(f"  Status: CORRUPTED - {message}")
        elif category == "password_protected":
            summary["password_protected"] += 1
            print(f"  Status: PASSWORD PROTECTED - {message}")
        elif category == "zero_pages":
            summary["zero_pages"] += 1
            print(f"  Status: ZERO PAGES - {message}")
        elif category == "error":
            summary["other_errors"] += 1
            print(f"  Status: ERROR - {message}")
        print("-" * 30)

    print("\n--- Verification Summary (Console) ---")
    print(f"Total PDF files scanned: {summary['total_scanned']}")
    print(f"  Valid PDFs (pages > 0): {summary['valid']}")
    print(f"  Corrupted or Invalid PDFs: {summary['corrupted']}")
    print(f"  Password-Protected PDFs: {summary['password_protected']}")
    print(f"  PDFs with 0 Pages: {summary['zero_pages']}")
    if summary["other_errors"] > 0:
        print(f"  Files with other verification errors: {summary['other_errors']}")
    print("------------------------------------")

    output_csv_file_path_to_use = ""
    if args.output_csv is None:
        abs_input_dir = os.path.abspath(args.input_directory)
        input_dir_basename = os.path.basename(os.path.normpath(abs_input_dir))
        default_csv_filename = f"{input_dir_basename}_verification_results.csv"
        output_csv_file_path_to_use = os.path.join(
            os.getcwd(), default_csv_filename)
    else:
        output_csv_file_path_to_use = os.path.abspath(args.output_csv)

    try:
        with open(output_csv_file_path_to_use, 'w', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_results_data)
        print(f"\nDetailed verification results saved to: {output_csv_file_path_to_use}")
    except IOError as e:
        print(f"\nError writing CSV results to '{output_csv_file_path_to_use}': {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while writing CSV results: {e}")


if __name__ == "__main__":
    main()
